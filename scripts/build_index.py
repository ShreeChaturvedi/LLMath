#!/usr/bin/env python3
"""Build FAISS index for NaturalProofs dataset.

Usage:
    python scripts/build_index.py [--rebuild]

Options:
    --rebuild   Force rebuild even if index exists
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmath.config import load_config
from llmath.retrieval import NaturalProofsRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for NaturalProofs"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/naturalproofs_faiss.index",
        help="Output path for FAISS index",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/naturalproofs_meta.json",
        help="Output path for metadata JSON",
    )
    args = parser.parse_args()

    # Ensure data directory exists
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    retriever_config = config.retriever.model_copy(
        update={
            "index_path": Path(args.index_path),
            "meta_path": Path(args.meta_path),
        }
    )

    print(f"Building index with embedding model: {config.embedding.model_name}")
    print(f"Index path: {args.index_path}")
    print(f"Metadata path: {args.meta_path}")
    print(f"Rebuild: {args.rebuild}")
    print()

    retriever = NaturalProofsRetriever(
        config=retriever_config,
        embedding_config=config.embedding,
        rebuild_index=args.rebuild,
    )

    # Test the index
    print("\nTesting index with sample query...")
    results = retriever.search("continuous function differentiable", k=3)
    print(f"Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r.score:.3f}, idx={r.idx}")

    print("\nIndex built successfully!")


if __name__ == "__main__":
    main()
