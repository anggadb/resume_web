"""Delete every record from the Pinecone index configured in .env."""

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete all records from every namespace in the Pinecone index "
            "configured by PINECONE_INDEX."
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def vector_count(namespace_stats: Any) -> int:
    if isinstance(namespace_stats, dict):
        return int(
            namespace_stats.get("vector_count")
            or namespace_stats.get("record_count")
            or 0
        )
    return int(
        getattr(namespace_stats, "vector_count", None)
        or getattr(namespace_stats, "record_count", None)
        or 0
    )


def confirm(index_name: str, total_records: int) -> bool:
    expected = f"DELETE {index_name}"
    print(
        f"This will permanently delete {total_records} record(s) from "
        f"Pinecone index '{index_name}'."
    )
    print(f"Type '{expected}' to continue:")
    return input("> ").strip() == expected


def main() -> int:
    args = parse_args()
    load_dotenv()

    api_key = require_env("PINECONE_API_KEY")
    index_name = require_env("PINECONE_INDEX")

    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    namespaces = (
        stats.get("namespaces", {})
        if isinstance(stats, dict)
        else stats.namespaces or {}
    )
    total_records = sum(vector_count(item) for item in namespaces.values())

    if not namespaces or total_records == 0:
        print(f"Pinecone index '{index_name}' contains no records.")
        return 0

    print(f"Index: {index_name}")
    for namespace, namespace_stats in namespaces.items():
        display_name = namespace or "__default__"
        print(f"- {display_name}: {vector_count(namespace_stats)} record(s)")

    if not args.yes and not confirm(index_name, total_records):
        print("Deletion cancelled.")
        return 1

    for namespace in namespaces:
        target_namespace = namespace or "__default__"
        index.delete(delete_all=True, namespace=target_namespace)
        print(f"Deletion requested for namespace '{target_namespace}'.")

    print(
        "Deletion requests completed. Pinecone is eventually consistent, "
        "so record counts may take a short time to update."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
