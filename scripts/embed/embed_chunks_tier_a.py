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


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="chunks_tier_a_filtered.jsonl")
    parser.add_argument("--output-embeddings-jsonl", default="embedded_chunks_tier_a.jsonl")
    parser.add_argument("--output-metadata-jsonl", default="chunk_metadata_tier_a.jsonl")
    parser.add_argument("--faiss-index", default="chunks_tier_a.faiss")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    if args.limit is not None:
        rows = rows[:args.limit]

    texts = [r["text"] for r in rows]
    print(f"Loaded {len(texts)} chunks")

    print(f"Loading embedding model: {args.model_name} on {args.device}")
    model = SentenceTransformer(args.model_name, device=args.device)

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, args.faiss_index)

    print("Writing metadata JSONL...")
    metadata_rows = []
    embedded_rows = []

    for i, (row, emb) in enumerate(zip(rows, embeddings)):
        metadata_rows.append({
            "faiss_id": i,
            "chunk_id": row.get("chunk_id"),
            "source_id": row.get("source_id"),
            "source_path": row.get("source_path"),
            "file_type": row.get("file_type"),
            "pages": row.get("pages"),
            "text": row.get("text"),
        })

        out = dict(row)
        out["embedding"] = emb.tolist()
        embedded_rows.append(out)

    write_jsonl(Path(args.output_metadata_jsonl), metadata_rows)
    write_jsonl(Path(args.output_embeddings_jsonl), embedded_rows)

    print("Done.")
    print(f"FAISS index: {args.faiss_index}")
    print(f"Metadata JSONL: {args.output_metadata_jsonl}")
    print(f"Embeddings JSONL: {args.output_embeddings_jsonl}")


if __name__ == "__main__":
    main()