#!/usr/bin/env python3
"""
RAG chat script that uses:
- SentenceTransformers + FAISS for retrieval
- Meta's native `llama` Python package for generation from ORIGINAL checkpoint format

This is intended for original checkpoint directories such as:
    /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct

Requirements:
1. A working FAISS index + chunk metadata JSONL
2. sentence-transformers, faiss-cpu, numpy
3. Meta's native `llama` package installed from the original repo, e.g.
       git clone https://github.com/meta-llama/llama.git
       cd llama
       python -m pip install -e .

Example:
    python rag_chat_llama_native.py \
      --faiss-index chunks_tier_a.faiss \
      --metadata-jsonl chunk_metadata_tier_a.jsonl \
      --ckpt-dir /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct \
      --tokenizer-path /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model \
      --query "What injector issues have appeared before?" \
      --top-k 5 \
      --show-context
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from llama import Llama
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import Meta's native `llama` package.\n"
        "Install it from the original repo in the environment where you run this script:\n"
        "  git clone https://github.com/meta-llama/llama.git\n"
        "  cd llama\n"
        "  python -m pip install -e .\n"
        f"\nOriginal import error:\n{exc}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG chat using original Meta Llama checkpoints")
    p.add_argument("--faiss-index", required=True, help="Path to FAISS index")
    p.add_argument("--metadata-jsonl", required=True, help="Path to metadata JSONL used with the FAISS index")
    p.add_argument("--ckpt-dir", required=True, help="Original checkpoint directory")
    p.add_argument("--tokenizer-path", default=None, help="Path to tokenizer.model (defaults to <ckpt-dir>/tokenizer.model)")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--embedding-device", default="cuda", help="cuda or cpu")
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-batch-size", type=int, default=1)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--show-context", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(query: str, retrieved_rows: List[Dict[str, Any]], max_context_chars: int) -> str:
    header = (
        "You are an engineering assistant for the Liquid Rocketry at Illinois knowledge base.\n"
        "Answer using only the retrieved context below.\n"
        "If the sources conflict, say so explicitly.\n"
        "Cite source paths inline in parentheses.\n"
        "Be concise but technically precise.\n\n"
        "Retrieved context:\n"
    )

    context_parts: List[str] = []
    total_chars = 0
    for i, row in enumerate(retrieved_rows, start=1):
        source = row.get("source_path", "")
        pages = row.get("pages", [])
        page_label = f" pages={pages}" if pages else ""
        text = row.get("text", "")
        block = f"[Chunk {i} | source={source}{page_label}]\n{text}\n"
        if total_chars + len(block) > max_context_chars:
            break
        context_parts.append(block)
        total_chars += len(block)

    context = "\n".join(context_parts)
    prompt = (
        f"{header}{context}\n"
        f"User question:\n{query}\n\n"
        "Answer:\n"
    )
    return prompt


def main() -> None:
    args = parse_args()

    faiss_index_path = Path(args.faiss_index)
    metadata_path = Path(args.metadata_jsonl)
    ckpt_dir = Path(args.ckpt_dir)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else ckpt_dir / "tokenizer.model"

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    print("Loading metadata...", flush=True)
    metadata = load_jsonl(metadata_path)

    print("Loading FAISS index...", flush=True)
    index = faiss.read_index(str(faiss_index_path))

    print(f"Loading embedding model on {args.embedding_device}: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model, device=args.embedding_device)

    print("Embedding query...", flush=True)
    q = embedder.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"Searching top {args.top_k} chunks...", flush=True)
    scores, ids = index.search(q, args.top_k)

    retrieved_rows: List[Dict[str, Any]] = []
    print("\nRetrieved chunks:\n", flush=True)
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        row = metadata[int(idx)]
        retrieved_rows.append(row)
        print("=" * 100, flush=True)
        print(f"Rank: {rank}", flush=True)
        print(f"Score: {score:.4f}", flush=True)
        print(f"Source: {row.get('source_path')}", flush=True)
        print(f"Pages: {row.get('pages')}", flush=True)
        if args.show_context:
            print("", flush=True)
            print(row.get("text", "")[:2000], flush=True)
            print("", flush=True)

    prompt = build_prompt(args.query, retrieved_rows, args.max_context_chars)

    print(f"\nLoading native Meta Llama generator from {ckpt_dir}...", flush=True)
    generator = Llama.build(
        ckpt_dir=str(ckpt_dir),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    print("Generating answer...\n", flush=True)
    dialogs = [[{"role": "user", "content": prompt}]]

    results = generator.chat_completion(
        dialogs,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    result = results[0]
    generation = result.get("generation", {})
    content = generation.get("content", "")

    print("=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(content)
    print("=" * 100)


if __name__ == "__main__":
    main()
