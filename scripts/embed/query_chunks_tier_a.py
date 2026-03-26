#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-index", default="chunks_tier_a.faiss")
    parser.add_argument("--metadata-jsonl", default="chunk_metadata_tier_a.jsonl")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    print("Loading metadata...")
    metadata = load_jsonl(Path(args.metadata_jsonl))

    print("Loading FAISS index...")
    index = faiss.read_index(args.faiss_index)

    print(f"Loading embedding model on {args.device}...")
    model = SentenceTransformer(args.model_name, device=args.device)

    print("Embedding query...")
    q = model.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("Searching...\n")
    scores, ids = index.search(q, args.top_k)

    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        row = metadata[int(idx)]
        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {row.get('chunk_id')}")
        print(f"Source: {row.get('source_path')}")
        print(f"Pages: {row.get('pages')}")
        print()
        print(row.get("text", "")[:1800])
        print()


if __name__ == "__main__":
    main()