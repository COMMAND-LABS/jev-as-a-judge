# TLDR

This project demonstrates how to implement Jev-as-a-Judge evaluation on the LangSmith platform.

It runs Command Labs agents that share a configuration but use different LLMs against the same LangSmith dataset, has Jev grade each answer against the reference answer, and records one LangSmith experiment per agent so you can compare them side by side.

## Project structure

- `scripts/upload_langsmith_dataset.py` - upload a Q&A CSV to LangSmith as a dataset
- `scripts/run_llm_as_a_judge.py` - run the agents and grade their answers with Jev
- `llm_judge/config.py` - CLI + environment configuration
- `llm_judge/jev_judge.py` - the Jev judge and its scoring rubric
- `llm_judge/orchestrator.py` - one LangSmith experiment per agent, run in parallel
- `clients/cmdlabs.py` - Command Labs API client (agent answers + agent config)

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install dependencies (this creates `.venv` automatically):

```bash
uv sync
```

2. Copy `.env.example` to `.env` and fill in `LANGSMITH_API_KEY`, `CMDLABS_API_KEY`, `TYPESAFE_API_KEY` (Jev) and `CMDLABS_AGENT_ID` (the agents to compare, e.g. `54,55`).

## Run

1. Upload the dataset. The CSV needs a header row with columns `q` (question) and `a` (reference answer). The `data/` folder is git-ignored, so put your CSV there yourself:

```bash
uv run scripts/upload_langsmith_dataset.py \
  --dataset-file data/ai-masterclass-dataset.csv \
  --dataset-name ai-masterclass-dataset
```

Add `--replace-existing` to replace the dataset's examples after editing the CSV.

2. Run the agents (one experiment per agent in `CMDLABS_AGENT_ID`):

```bash
uv run scripts/run_llm_as_a_judge.py \
  --experiment-name ai-masterclass \
  --dataset-name ai-masterclass-dataset
```

- `--max-examples 2` - quick test run on the first examples. Every agent gets the same examples.
- `--agent-id 54,55` - run different agents than the ones in `.env`.
- `--max-concurrency 5` - examples evaluated at the same time per agent (default 5, or `MAX_CONCURRENCY` in `.env`). The agents also run in parallel; lower this if you see rate-limit errors.
- `--run-group <id>` - choose the run group ID yourself, e.g. to add a re-run of one agent to an existing group.

3. In LangSmith, open the dataset, select the experiments from the same run group and click Compare.

## What you see in LangSmith

Every run gets a run group ID (e.g. `20261028-1532-a1b2`), so experiment names look like:

```
ai-masterclass-20261028-1532-a1b2-agent-54-gpt-5.4-<langsmith suffix>
```

Each experiment's description shows the agent's name and model (e.g. `AMA (w/ GPT 5.4): openai/gpt-5.4`), and its metadata has `run_group`, `agent_id`, `agent_name`, `model_provider`, `model` and the agent's full `agent_config` (tools, knowledge base, system prompt).

## How Jev scores answers

Jev compares each agent answer with the reference answer and scores it from 0 to 4:

0 = Poor, 1 = Bad, but not the worst, 2 = Somewhat correct, 3 = Okay, 4 = Great.

The score (`jev_judge_score`) is probability-weighted, so it can fall between levels (e.g. 3.6). The LangSmith comment shows Jev's confidence and the probability of each level.
