#!/usr/bin/env python3
"""
Persistent terminal RAG chat loop for local Llama checkpoint + FAISS retrieval.

This script is meant to be launched ONCE and then kept open:
- loads FAISS + metadata once
- loads embedding model once
- loads native Meta Llama generator once
- then lets you ask repeated questions in a terminal chat loop

Recommended launch:
PYTHONPATH=/home/jerich-lee/Documents/llama3 \
torchrun --nproc_per_node 1 rag_chat_loop.py \
  --faiss-index chunks_tier_a.faiss \
  --metadata-jsonl chunk_metadata_tier_a.jsonl \
  --ckpt-dir /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct \
  --tokenizer-path /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model \
  --max-seq-len 4096 \
  --max-batch-size 4 \
  --top-k 5

Commands inside chat:
  /quit
  /exit
  /show
  /hide
  /topk N
  /context N
  /help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer

from llama import Llama


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent local RAG chat loop")
    p.add_argument("--faiss-index", required=True)
    p.add_argument("--metadata-jsonl", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)

    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--embedding-device", default="cuda")

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-batch-size", type=int, default=4)
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
        "Use only the retrieved context below when answering.\n"
        "If sources conflict, say so explicitly.\n"
        "Cite source paths inline in parentheses.\n"
        "Be technically precise and concise.\n\n"
        "Retrieved context:\n"
    )

    parts: List[str] = []
    total_chars = 0
    for i, row in enumerate(retrieved_rows, start=1):
        source = row.get("source_path", "")
        pages = row.get("pages", [])
        pages_str = f" pages={pages}" if pages else ""
        text = row.get("text", "")
        block = f"[Chunk {i} | source={source}{pages_str}]\n{text}\n"
        if total_chars + len(block) > max_context_chars:
            break
        parts.append(block)
        total_chars += len(block)

    context = "\n".join(parts)
    return (
        f"{header}{context}\n"
        f"User question:\n{query}\n\n"
        "Answer:\n"
    )


def retrieve(
    *,
    query: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(q, top_k)

    rows: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        row = dict(metadata[int(idx)])
        row["_rank"] = rank
        row["_score"] = float(score)
        rows.append(row)
    return rows


def print_retrieved(rows: List[Dict[str, Any]], show_context: bool) -> None:
    print("\nRetrieved chunks:\n", flush=True)
    for row in rows:
        print("=" * 100, flush=True)
        print(f"Rank: {row.get('_rank')}", flush=True)
        print(f"Score: {row.get('_score'):.4f}", flush=True)
        print(f"Source: {row.get('source_path')}", flush=True)
        print(f"Pages: {row.get('pages')}", flush=True)
        if show_context:
            print("", flush=True)
            print((row.get("text") or "")[:2000], flush=True)
            print("", flush=True)


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /quit or /exit      Exit chat\n"
        "  /show               Show retrieved chunk text each turn\n"
        "  /hide               Hide retrieved chunk text each turn\n"
        "  /topk N             Change number of retrieved chunks\n"
        "  /context N          Change max context chars\n"
        "  /help               Show this help\n",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    faiss_index_path = Path(args.faiss_index)
    metadata_path = Path(args.metadata_jsonl)
    ckpt_dir = Path(args.ckpt_dir)
    tokenizer_path = Path(args.tokenizer_path)

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    print("Loading metadata...", flush=True)
    metadata = load_jsonl(metadata_path)

    print("Loading FAISS index...", flush=True)
    index = faiss.read_index(str(faiss_index_path))

    print(f"Loading embedding model on {args.embedding_device}: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model, device=args.embedding_device)

    print(f"Loading native Meta Llama generator from {ckpt_dir}...", flush=True)
    generator = Llama.build(
        ckpt_dir=str(ckpt_dir),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    top_k = args.top_k
    max_context_chars = args.max_context_chars
    show_context = args.show_context

    print("\nRAG chat ready.")
    print("Type /help for commands. Type /quit to exit.\n", flush=True)

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", flush=True)
            break

        if not query:
            continue

        if query in {"/quit", "/exit"}:
            print("Exiting.", flush=True)
            break

        if query == "/help":
            print_help()
            continue

        if query == "/show":
            show_context = True
            print("Retrieved chunk text will be shown.", flush=True)
            continue

        if query == "/hide":
            show_context = False
            print("Retrieved chunk text will be hidden.", flush=True)
            continue

        if query.startswith("/topk "):
            try:
                top_k = max(1, int(query.split(maxsplit=1)[1]))
                print(f"top_k set to {top_k}", flush=True)
            except Exception:
                print("Usage: /topk N", flush=True)
            continue

        if query.startswith("/context "):
            try:
                max_context_chars = max(1000, int(query.split(maxsplit=1)[1]))
                print(f"max_context_chars set to {max_context_chars}", flush=True)
            except Exception:
                print("Usage: /context N", flush=True)
            continue

        retrieved = retrieve(
            query=query,
            embedder=embedder,
            index=index,
            metadata=metadata,
            top_k=top_k,
        )

        # print_retrieved(retrieved, show_context=show_context)

        prompt = build_prompt(
            query=query,
            retrieved_rows=retrieved,
            max_context_chars=max_context_chars,
        )

        dialogs = [[{"role": "user", "content": prompt}]]

        print("\nAssistant>\n", flush=True)
        results = generator.chat_completion(
            dialogs,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        result = results[0]
        generation = result.get("generation", {})
        content = generation.get("content", "")
        print(content, flush=True)
        print("\n" + "-" * 100 + "\n", flush=True)


if __name__ == "__main__":
    main()
