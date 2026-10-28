"""Run each Command Labs agent against a LangSmith dataset and grade the answers with Jev."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate

from llm_judge.config import load_config
from llm_judge.jev_judge import build_jev_judge_evaluator
from clients.cmdlabs import call_cmdlabs_completion, fetch_cmdlabs_agent_config


def _parse_agent_ids(raw_agent_ids: str | None) -> list[str]:
    agent_ids: list[str] = []
    for item in (raw_agent_ids or "").split(","):
        agent_id = item.strip()
        if agent_id and agent_id not in agent_ids:
            agent_ids.append(agent_id)
    return agent_ids


def _fetch_agent_config(*, args: Any, api_key: str | None, agent_id: str) -> dict[str, Any] | None:
    try:
        return fetch_cmdlabs_agent_config(
            api_url=args.cmdlabs_api_url,
            api_key=api_key,
            agent_id=agent_id,
            cmdlabs_api_timeout_seconds=args.cmdlabs_api_timeout_seconds,
        )
    except Exception as exc:
        print(f"Unable to fetch config for agent {agent_id}: {exc}")
        return None


def _describe_agent(agent_config: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the agent name and model out of its config, for easy reading in LangSmith."""
    agent_config = agent_config or {}
    model = ((agent_config.get("config") or {}).get("data") or {}).get("model") or {}
    return {
        "agent_name": agent_config.get("name"),
        "model_provider": model.get("provider"),
        "model": model.get("model"),
    }


def main() -> None:
    args = load_config()

    agent_ids = _parse_agent_ids(args.agent_ids)
    if not agent_ids:
        raise ValueError("Missing agent id. Set CMDLABS_AGENT_ID in .env or pass --agent-id (e.g. 39 or 39,40).")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("Missing LANGSMITH_API_KEY in .env (required for LangSmith evaluation).")
    jev_api_key = os.getenv("TYPESAFE_API_KEY")
    if not jev_api_key:
        raise ValueError("Missing TYPESAFE_API_KEY in .env (required for the Jev judge).")
    cmdlabs_api_key = os.getenv("CMDLABS_API_KEY")
    # Shared by every experiment in this run, so the group can be found in LangSmith.
    run_group = args.run_group or f"{datetime.now():%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"

    # Fetch the examples once so every agent is evaluated on exactly the same examples.
    client = Client()
    if not client.has_dataset(dataset_name=args.dataset_name):
        raise ValueError(
            f"LangSmith dataset {args.dataset_name} not found. Upload it first with scripts/upload_langsmith_dataset.py."
        )
    examples = list(client.list_examples(dataset_name=args.dataset_name, limit=args.max_examples))
    if not examples:
        raise ValueError(f"LangSmith dataset {args.dataset_name} has no examples.")

    jev_judge = build_jev_judge_evaluator(
        model_name=args.jev_judge_model,
        api_key=jev_api_key,
        timeout_seconds=args.judge_timeout_seconds,
    )

    print(f"Dataset {args.dataset_name}: {len(examples)} examples per agent")
    print(f"Agents: {', '.join(agent_ids)} | Judge: {args.jev_judge_model} | Run group: {run_group}")
    print(f"Running agents in parallel, {args.max_concurrency} examples at a time per agent")

    # One experiment per agent: same examples, same judge, different agent.
    def run_agent(agent_id: str) -> Any:
        # Record each agent's config so the experiment shows what it ran against.
        agent_config = _fetch_agent_config(args=args, api_key=cmdlabs_api_key, agent_id=agent_id)
        agent = _describe_agent(agent_config)

        def target(inputs: dict[str, str]) -> dict[str, str]:
            answer = call_cmdlabs_completion(
                api_url=args.cmdlabs_api_url,
                api_key=cmdlabs_api_key,
                agent_id=agent_id,
                session_id=str(uuid.uuid4()),
                prompt=inputs["question"],
                cmdlabs_api_timeout_seconds=args.cmdlabs_api_timeout_seconds,
                cmdlabs_api_retries=args.cmdlabs_api_retries,
            )
            return {"answer": answer}

        experiment_name = f"{args.experiment_name}-{run_group}-agent-{agent_id}"
        if agent["model"]:
            experiment_name += f"-{agent['model']}"
        label = f"agent {agent_id} ({agent['agent_name']})" if agent["agent_name"] else f"agent {agent_id}"
        print(f"\nRunning LangSmith evaluation for {label}: {experiment_name}...")
        return evaluate(
            target,
            data=examples,
            evaluators=[jev_judge],
            experiment_prefix=experiment_name,
            description=f"{agent['agent_name']}: {agent['model_provider']}/{agent['model']}" if agent["model"] else None,
            max_concurrency=args.max_concurrency,
            client=client,
            metadata={
                "run_group": run_group,
                "agent_id": agent_id,
                **agent,
                "agent_config": agent_config,
                "dataset_name": args.dataset_name,
                "jev_judge_model": args.jev_judge_model,
            },
        )

    with ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
        experiments = list(zip(agent_ids, executor.map(run_agent, agent_ids)))

    print(f"\nEvaluation finished. Run group: {run_group}")
    for agent_id, results in experiments:
        print(f"Agent {agent_id}: {getattr(results, 'experiment_url', None) or getattr(results, 'experiment_name', '')}")
    if len(experiments) > 1:
        print(f"Compare them in LangSmith: open dataset {args.dataset_name}, select the experiments with run_group {run_group}, then Compare.")
