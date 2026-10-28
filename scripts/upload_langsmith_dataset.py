import argparse
import csv

from dotenv import load_dotenv
from langsmith import Client
from langsmith import schemas as ls_schemas

def load_examples(csv_path: str, max_examples: int | None = None) -> list[dict[str, dict[str, str]]]:
    """Read a CSV with columns q and a into LangSmith examples, skipping rows missing either."""
    examples: list[dict[str, dict[str, str]]] = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            question = (row.get("q") or "").strip()
            answer = (row.get("a") or "").strip()
            if not question or not answer:
                continue
            examples.append({"inputs": {"question": question}, "outputs": {"answer": answer}})
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local CSV dataset to LangSmith.")
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Path to the local CSV file (expects columns q and a).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="LangSmith dataset name to create/update.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of rows to upload.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="If dataset exists, delete its examples before uploading new ones.",
    )
    return parser.parse_args()


def ensure_dataset_exists(client: Client, dataset_name: str) -> None:
    if client.has_dataset(dataset_name=dataset_name):
        return
    client.create_dataset(
        dataset_name=dataset_name,
        description="Uploaded via upload_langsmith_dataset.py",
        data_type=ls_schemas.DataType.kv,
    )


def maybe_clear_dataset_examples(client: Client, dataset_name: str, replace_existing: bool) -> None:
    if not replace_existing:
        return
    existing_ids = [example.id for example in client.list_examples(dataset_name=dataset_name)]
    if not existing_ids:
        return
    chunk_size = 100
    for idx in range(0, len(existing_ids), chunk_size):
        client.delete_examples(existing_ids[idx : idx + chunk_size])


def main() -> None:
    load_dotenv()
    args = parse_args()

    examples = load_examples(args.dataset_file, max_examples=args.max_examples)
    if not examples:
        raise ValueError(f"No valid examples found in {args.dataset_file}")

    client = Client()
    ensure_dataset_exists(client, args.dataset_name)
    maybe_clear_dataset_examples(client, args.dataset_name, args.replace_existing)
    client.create_examples(dataset_name=args.dataset_name, examples=examples)

    print(f"Uploaded {len(examples)} examples to LangSmith dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
