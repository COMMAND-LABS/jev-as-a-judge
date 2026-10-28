"""Configuration and argument parsing for the runner."""

import argparse
import os

from dotenv import load_dotenv


def load_config() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run Kalygo agents against a LangSmith dataset and grade their answers with Jev."
    )
    parser.add_argument(
        "--agent-id",
        dest="agent_ids",
        default=os.getenv("KALYGO_AGENT_ID"),
        help="Kalygo agent ID, or a comma-separated list (e.g. 39,40) to run one experiment per agent",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Existing LangSmith dataset to evaluate on (upload it first with scripts/upload_langsmith_dataset.py)",
    )
    parser.add_argument(
        "--experiment-name",
        default="kalygo",
        help="LangSmith experiment name; with several agents, '-agent-<id>' is appended for each one",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of dataset examples to evaluate",
    )
    parser.add_argument(
        "--kalygo-completion-api-url",
        default=os.getenv("KALYGO_COMPLETION_API_URL", "https://completion.kalygo.io"),
        help="Base URL for Kalygo API (completion + agent config)",
    )
    parser.add_argument(
        "--kalygo-api-timeout-seconds",
        type=int,
        default=int(os.getenv("KALYGO_API_TIMEOUT_SECONDS", "120")),
        help="HTTP timeout for each Kalygo API completion request",
    )
    parser.add_argument(
        "--kalygo-api-retries",
        type=int,
        default=int(os.getenv("KALYGO_API_RETRIES", "2")),
        help="Number of retries for transient Kalygo streaming timeouts/connection errors",
    )
    parser.add_argument(
        "--jev-judge-model",
        default=os.getenv("JEV_JUDGE_MODEL", "jev-latest"),
        help="Jev model used to grade answers",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=int,
        default=int(os.getenv("JUDGE_TIMEOUT_SECONDS", "60")),
        help="HTTP timeout for each Jev API call",
    )
    return parser.parse_args()
