# TLDR

Notes from preparing the "Jev as a Judge" DEMO

##

uv run scripts/upload_langsmith_dataset.py \
  --dataset-file data/sample-3-examples.csv \
  --dataset-name sample-3-examples-test

##

uv run scripts/upload_langsmith_dataset.py \
  --dataset-file data/ai-masterclass-dataset.csv \
  --dataset-name ai-masterclass-dataset \
  --replace-existing

##

uv run scripts/run_llm_as_a_judge.py \
  --experiment-name ama-rag \
  --dataset-name ai-masterclass-dataset

##

uv run scripts/run_llm_as_a_judge.py \
  --experiment-name ama-rag \
  --dataset-name ai-masterclass-dataset \
  --max-concurrency 5
