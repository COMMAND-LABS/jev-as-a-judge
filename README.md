# TLDR

This project demonstrates how to implement Jev-as-a-Judge evaluation on the LangSmith platform.

It runs Command Labs agents that share a configuration but use different LLMs against the same LangSmith dataset, has Jev grade each answer against the reference answer, and records one LangSmith experiment per agent so you can compare them side by side.

## Project structure

- `scripts/upload_langsmith_dataset.py` - upload a Q&A CSV to LangSmith as a dataset
- `scripts/run_llm_as_a_judge.py` - run the agents and grade their answers with Jev
- `llm_judge/config.py` - CLI + environment configuration
- `clients/cmdlabs.py` - Command Labs API call + stream parsing
- `llm_judge/jev_judge.py` - the Jev judge and its scoring rubric
- `llm_judge/orchestrator.py` - one LangSmith experiment per agent

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install dependencies (this creates `.venv` automatically):

```bash
uv sync
```

2. Copy `.env.example` to `.env` and fill in `LANGSMITH_API_KEY`, `CMDLABS_API_KEY` and `TYPESAFE_API_KEY` (Jev).

## Run

1. Upload the dataset. The CSV needs a header row with columns `q` (question) and `a` (reference answer):

```bash
uv run scripts/upload_langsmith_dataset.py \
  --dataset-file data/ai-masterclass-dataset.csv \
  --dataset-name ai-masterclass-dataset
```

Add `--replace-existing` to replace the dataset's examples after editing the CSV.

2. Run the agents (one experiment per agent ID):

```bash
uv run scripts/run_llm_as_a_judge.py \
  --experiment-name ai-masterclass \
  --dataset-name ai-masterclass-dataset \
  --agent-id 39,40
```

Add `--max-examples 5` for a quick test run. Every agent gets the same examples.

3. In LangSmith, open the dataset, select the experiments and click Compare.

## How Jev scores answers

Jev compares each agent answer with the reference answer and scores it from 0 to 4:

0 = Poor, 1 = Bad, but not the worst, 2 = Somewhat correct, 3 = Okay, 4 = Great.

The score (`jev_judge_score`) is probability-weighted, so it can fall between levels (e.g. 3.6). The LangSmith comment shows Jev's confidence and the probability of each level.
