#!/usr/bin/env python3
"""
RAG chat wrapper that:
1. retrieves top-k chunks from a FAISS index
2. builds a grounded prompt
3. invokes a working local Llama 3/3.1 backend through torchrun

This is designed for the setup where Meta's example_chat_completion.py already
works on your machine with the original checkpoint directory.

Example:
    python rag_chat_llama.py \
      --faiss-index chunks_tier_a.faiss \
      --metadata-jsonl chunk_metadata_tier_a.jsonl \
      --llama-repo-dir /home/jerich-lee/Documents/llama3 \
      --torchrun-cmd /home/jerich-lee/Documents/llama3/.venv/bin/torchrun \
      --ckpt-dir /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct \
      --tokenizer-path /home/jerich-lee/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model \
      --query "What injector issues have appeared before?" \
      --top-k 5 \
      --show-context
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer


ANSWER_BEGIN = "__RAG_ANSWER_BEGIN__"
ANSWER_END = "__RAG_ANSWER_END__"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG chat via FAISS + local torchrun llama backend")
    p.add_argument("--faiss-index", required=True)
    p.add_argument("--metadata-jsonl", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=5)

    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--embedding-device", default="cuda")

    p.add_argument("--llama-repo-dir", required=True, help="Repo root containing the working llama example/backend")
    p.add_argument("--torchrun-cmd", default="torchrun", help="Full path to torchrun in the working llama env, if needed")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)

    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-batch-size", type=int, default=4)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument("--show-context", action="store_true")
    p.add_argument("--nproc-per-node", type=int, default=1)
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
        "If the sources conflict, say so explicitly.\n"
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


def write_helper_script(path: Path) -> None:
    helper = '''#!/usr/bin/env python3
import argparse
from pathlib import Path
from llama import Llama

ANSWER_BEGIN = "__RAG_ANSWER_BEGIN__"
ANSWER_END = "__RAG_ANSWER_END__"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-batch-size", type=int, default=4)
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

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

    print(ANSWER_BEGIN)
    print(content)
    print(ANSWER_END)

if __name__ == "__main__":
    main()
'''
    path.write_text(helper, encoding="utf-8")
    path.chmod(0o755)


def run_generator(
    *,
    torchrun_cmd: str,
    llama_repo_dir: Path,
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str,
    max_seq_len: int,
    max_batch_size: int,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    nproc_per_node: int,
) -> str:
    with tempfile.TemporaryDirectory(prefix="rag_llama_") as td:
        td_path = Path(td)
        helper_path = td_path / "llama_helper.py"
        prompt_path = td_path / "prompt.txt"

        write_helper_script(helper_path)
        prompt_path.write_text(prompt, encoding="utf-8")

        cmd = [
            torchrun_cmd,
            "--nproc_per_node", str(nproc_per_node),
            str(helper_path),
            "--ckpt-dir", ckpt_dir,
            "--tokenizer-path", tokenizer_path,
            "--prompt-file", str(prompt_path),
            "--max-seq-len", str(max_seq_len),
            "--max-batch-size", str(max_batch_size),
            "--max-gen-len", str(max_gen_len),
            "--temperature", str(temperature),
            "--top-p", str(top_p),
        ]

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(llama_repo_dir) + (os.pathsep + existing_pp if existing_pp else "")

        print("\nLaunching local generator:\n  " + " ".join(cmd) + "\n", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(llama_repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        collected: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            collected.append(line)

        ret = proc.wait()
        output = "".join(collected)

        if ret != 0:
            raise RuntimeError(f"Generator subprocess failed with exit code {ret}")

        m = re.search(
            re.escape(ANSWER_BEGIN) + r"\n(.*?)\n" + re.escape(ANSWER_END),
            output,
            flags=re.DOTALL,
        )
        if not m:
            raise RuntimeError("Could not parse answer markers from generator output")

        return m.group(1).strip()


def main() -> None:
    args = parse_args()

    faiss_index_path = Path(args.faiss_index)
    metadata_path = Path(args.metadata_jsonl)
    llama_repo_dir = Path(args.llama_repo_dir)

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")
    if not llama_repo_dir.exists():
        raise FileNotFoundError(f"Llama repo dir not found: {llama_repo_dir}")
    if not Path(args.ckpt_dir).exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {args.ckpt_dir}")
    if not Path(args.tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer path not found: {args.tokenizer_path}")

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

    answer = run_generator(
        torchrun_cmd=args.torchrun_cmd,
        llama_repo_dir=llama_repo_dir,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        prompt=prompt,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
        nproc_per_node=args.nproc_per_node,
    )

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(answer)
    print("=" * 100)


if __name__ == "__main__":
    main()
