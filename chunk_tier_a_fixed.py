#!/usr/bin/env python3
"""
Chunk extracted Tier A documents into retrieval-sized JSONL chunks.

Input:
    extracted_documents_tier_a.jsonl

Outputs:
    1. chunks_tier_a.jsonl         -> one JSON object per chunk
    2. chunk_report_tier_a.csv     -> per-document chunking summary

Default chunking strategy:
    - paragraph / page-aware block chunking
    - target ~700 words per chunk
    - max ~900 words per chunk
    - 1 overlapping block between adjacent chunks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PAGE_MARKER_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk extracted Tier A documents into retrieval-sized chunks."
    )
    parser.add_argument("--input-jsonl", default="extracted_documents_tier_a.jsonl")
    parser.add_argument("--output-jsonl", default="chunks_tier_a.jsonl")
    parser.add_argument("--report-csv", default="chunk_report_tier_a.csv")
    parser.add_argument("--target-words", type=int, default=700)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--overlap-blocks", type=int, default=1)
    parser.add_argument("--min-chunk-words", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--chunk-log-every", type=int, default=1)
    parser.add_argument("--block-log-every", type=int, default=200)
    return parser.parse_args()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_spaces(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc


def split_lines_preserve_tables(block: str) -> List[str]:
    lines = [normalize_spaces(x) for x in block.split("\n")]
    return [x for x in lines if x]


def regroup_units(units: Sequence[str], max_words: int) -> List[str]:
    groups: List[str] = []
    current: List[str] = []
    current_words = 0

    for unit in units:
        unit = normalize_spaces(unit)
        if not unit:
            continue
        unit_words = count_words(unit)

        if unit_words > max_words:
            if current:
                groups.append(
                    "\n".join(current) if any("|" in x for x in current) else " ".join(current)
                )
                current = []
                current_words = 0
            groups.extend(split_block_into_small_pieces(unit, max_words))
            continue

        if current and current_words + unit_words > max_words:
            groups.append(
                "\n".join(current) if any("|" in x for x in current) else " ".join(current)
            )
            current = [unit]
            current_words = unit_words
        else:
            current.append(unit)
            current_words += unit_words

    if current:
        groups.append(
            "\n".join(current) if any("|" in x for x in current) else " ".join(current)
        )

    return groups


def split_block_into_small_pieces(block: str, max_words: int) -> List[str]:
    block = normalize_spaces(block)
    if not block:
        return []

    if count_words(block) <= max_words:
        return [block]

    if "|" in block or "\n" in block:
        lines = split_lines_preserve_tables(block)
        if len(lines) > 1:
            return regroup_units(lines, max_words)

    sentences = [normalize_spaces(s) for s in SENTENCE_SPLIT_RE.split(block)]
    sentences = [s for s in sentences if s]
    if len(sentences) > 1:
        return regroup_units(sentences, max_words)

    words = block.split()
    out: List[str] = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i : i + max_words]))
    return out


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def estimate_eta(elapsed_seconds: float, completed_units: int, total_units: int) -> float:
    if completed_units <= 0:
        return 0.0
    rate = completed_units / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if rate <= 0:
        return 0.0
    remaining = max(0, total_units - completed_units)
    return remaining / rate


def live_prefix(
    doc_index: Optional[int],
    total_docs: Optional[int],
    start_time: Optional[float],
) -> str:
    elapsed = format_seconds(time.time() - start_time) if start_time is not None else "00:00:00"
    if doc_index is not None and total_docs is not None:
        return f"[doc {doc_index}/{total_docs} | elapsed {elapsed}]"
    return f"[elapsed {elapsed}]"


def split_text_into_blocks(
    text: str,
    max_words_per_block: int,
    *,
    source_path: str = "",
    doc_index: Optional[int] = None,
    total_docs: Optional[int] = None,
    start_time: Optional[float] = None,
    block_log_every: int = 0,
) -> List[str]:
    text = normalize_newlines(text)
    raw_blocks = re.split(r"\n\s*\n", text)
    raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

    if not raw_blocks:
        raw_blocks = [x.strip() for x in text.split("\n") if x.strip()]

    total_raw_blocks = len(raw_blocks)

    print(
        f"    {live_prefix(doc_index, total_docs, start_time)} "
        f"START block prep -> {source_path} | raw_blocks={total_raw_blocks:,}",
        flush=True,
    )

    blocks: List[str] = []
    for raw_idx, raw in enumerate(raw_blocks, start=1):
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        pieces = split_block_into_small_pieces(raw, max_words_per_block)
        pieces = [normalize_spaces(b) for b in pieces if normalize_spaces(b)]
        blocks.extend(pieces)

        if block_log_every > 0 and (
            raw_idx == 1
            or raw_idx == total_raw_blocks
            or raw_idx % block_log_every == 0
        ):
            print(
                f"    {live_prefix(doc_index, total_docs, start_time)} "
                f"block_prep {raw_idx:,}/{total_raw_blocks:,} | "
                f"normalized_blocks_so_far={len(blocks):,}",
                flush=True,
            )

    print(
        f"    {live_prefix(doc_index, total_docs, start_time)} "
        f"DONE block prep -> {source_path} | normalized_blocks={len(blocks):,}",
        flush=True,
    )

    return blocks


def join_blocks(blocks: Sequence[str]) -> str:
    return "\n\n".join(blocks).strip()


def unique_pages_in_text(text: str) -> List[int]:
    pages = [int(x) for x in PAGE_MARKER_RE.findall(text)]
    return sorted(set(pages))


def overlap_would_immediately_refinalize(
    overlap_words: int,
    next_block_words: int,
    target_words: int,
    max_words: int,
    min_chunk_words: int,
) -> bool:
    would_exceed = overlap_words + next_block_words > max_words
    reached_target = overlap_words >= target_words
    would_trigger_target_boundary = (
        reached_target
        and overlap_words + next_block_words > target_words + max(50, target_words // 8)
    )
    return (would_exceed and overlap_words >= min_chunk_words) or would_trigger_target_boundary


def build_chunk_objects(
    doc: Dict[str, Any],
    target_words: int,
    max_words: int,
    overlap_blocks: int,
    min_chunk_words: int,
    *,
    doc_index: Optional[int] = None,
    total_docs: Optional[int] = None,
    start_time: Optional[float] = None,
    chunk_log_every: int = 0,
    block_log_every: int = 0,
) -> List[Dict[str, Any]]:
    text = str(doc.get("text", "") or "").strip()
    if not text:
        return []

    source_path = str(doc.get("source_path", "") or "")

    blocks = split_text_into_blocks(
        text,
        max_words_per_block=max_words,
        source_path=source_path,
        doc_index=doc_index,
        total_docs=total_docs,
        start_time=start_time,
        block_log_every=block_log_every,
    )
    if not blocks:
        return []

    print(
        f"    {live_prefix(doc_index, total_docs, start_time)} "
        f"START chunk build -> {source_path} | "
        f"doc_words={count_words(text):,} | blocks={len(blocks):,}",
        flush=True,
    )

    chunks_blocks: List[List[str]] = []
    current_blocks: List[str] = []
    current_words = 0

    i = 0
    total_blocks = len(blocks)
    safety_counter = 0
    max_iterations = max(10000, total_blocks * 20)

    while i < len(blocks):
        safety_counter += 1
        if safety_counter > max_iterations:
            raise RuntimeError(
                f"Chunking loop exceeded safety limit for document: {source_path}. "
                f"Likely no forward progress near block {i+1}/{total_blocks}."
            )

        block = blocks[i]
        block_words = count_words(block)

        if not current_blocks:
            current_blocks = [block]
            current_words = block_words
            i += 1
            continue

        would_exceed = current_words + block_words > max_words
        reached_target = current_words >= target_words

        should_finalize = (
            (would_exceed and current_words >= min_chunk_words)
            or (
                reached_target
                and current_words + block_words > target_words + max(50, target_words // 8)
            )
        )

        if should_finalize:
            finalized_words = current_words
            chunks_blocks.append(current_blocks)
            chunk_num = len(chunks_blocks)

            if chunk_log_every > 0 and (
                chunk_num == 1 or chunk_num % chunk_log_every == 0
            ):
                print(
                    f"    {live_prefix(doc_index, total_docs, start_time)} "
                    f"chunk_created {chunk_num:,} | finalized_words={finalized_words:,} | "
                    f"next_block={i+1:,}/{total_blocks:,}",
                    flush=True,
                )

            overlap = current_blocks[-overlap_blocks:] if overlap_blocks > 0 else []

            while overlap:
                overlap_words = sum(count_words(x) for x in overlap)
                if overlap_would_immediately_refinalize(
                    overlap_words=overlap_words,
                    next_block_words=block_words,
                    target_words=target_words,
                    max_words=max_words,
                    min_chunk_words=min_chunk_words,
                ):
                    overlap = overlap[1:]
                else:
                    break

            current_blocks = list(overlap)
            current_words = sum(count_words(x) for x in current_blocks)
            continue

        current_blocks.append(block)
        current_words += block_words
        i += 1

    if current_blocks:
        tail_words = sum(count_words(x) for x in current_blocks)

        if chunks_blocks and tail_words < min_chunk_words:
            merged = chunks_blocks[-1] + current_blocks
            chunks_blocks[-1] = merged
            print(
                f"    {live_prefix(doc_index, total_docs, start_time)} "
                f"merged_tiny_tail -> {source_path} | tail_words={tail_words:,}",
                flush=True,
            )
        else:
            chunks_blocks.append(current_blocks)
            chunk_num = len(chunks_blocks)

            if chunk_log_every > 0 and (
                chunk_num == 1 or chunk_num % chunk_log_every == 0
            ):
                print(
                    f"    {live_prefix(doc_index, total_docs, start_time)} "
                    f"chunk_created {chunk_num:,} | finalized_words={tail_words:,} | "
                    f"next_block={total_blocks:,}/{total_blocks:,}",
                    flush=True,
                )

    print(
        f"    {live_prefix(doc_index, total_docs, start_time)} "
        f"DONE chunk build -> {source_path} | total_chunks={len(chunks_blocks):,}",
        flush=True,
    )

    chunk_texts = [join_blocks(cb) for cb in chunks_blocks if join_blocks(cb)]

    source_id = str(doc.get("source_id", "") or "")
    doc_hash = str(doc.get("text_sha256", "") or sha256_text(text))
    metadata = doc.get("metadata", {}) or {}

    out: List[Dict[str, Any]] = []
    total_chunks = len(chunk_texts)

    for idx, chunk_text in enumerate(chunk_texts, start=1):
        pages = unique_pages_in_text(chunk_text)
        chunk_hash = sha256_text(chunk_text)
        chunk_id_base = source_id or source_path or doc_hash[:12]
        chunk_id = f"{chunk_id_base}::chunk_{idx:04d}"

        out.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_name": doc.get("source_name", ""),
            "source_path": source_path,
            "local_path": doc.get("local_path", ""),
            "relative_local_path": doc.get("relative_local_path", ""),
            "file_type": doc.get("file_type", ""),
            "mime_type": doc.get("mime_type", ""),
            "web_view_link": doc.get("web_view_link", ""),
            "download_mode": doc.get("download_mode", ""),
            "manifest_status": doc.get("manifest_status", ""),
            "document_text_sha256": doc_hash,
            "document_char_count": doc.get("char_count", len(text)),
            "document_word_count": doc.get("word_count", count_words(text)),
            "chunk_index": idx,
            "chunk_count_for_document": total_chunks,
            "chunk_char_count": len(chunk_text),
            "chunk_word_count": count_words(chunk_text),
            "approx_token_count": approx_token_count(chunk_text),
            "pages": pages,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "chunk_text_sha256": chunk_hash,
            "text": chunk_text,
            "metadata": metadata,
        })

    return out


def main() -> None:
    args = parse_args()

    if args.overlap_blocks < 0:
        raise ValueError("--overlap-blocks must be >= 0")
    if args.min_chunk_words <= 0:
        raise ValueError("--min-chunk-words must be > 0")
    if args.target_words <= 0 or args.max_words <= 0:
        raise ValueError("--target-words and --max-words must be > 0")
    if args.max_words < args.target_words:
        raise ValueError("--max-words must be >= --target-words")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    if args.chunk_log_every < 0:
        raise ValueError("--chunk-log-every must be >= 0")
    if args.block_log_every < 0:
        raise ValueError("--block-log-every must be >= 0")

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_jsonl = Path(args.output_jsonl)
    report_csv = Path(args.report_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading documents...", flush=True)
    docs = list(iter_jsonl(input_path))
    if args.limit is not None:
        docs = docs[: args.limit]

    total_docs = len(docs)

    for doc in docs:
        if not doc.get("word_count"):
            text = str(doc.get("text", "") or "")
            doc["word_count"] = count_words(text)
        if not doc.get("char_count"):
            text = str(doc.get("text", "") or "")
            doc["char_count"] = len(text)

    total_input_words = sum(int(doc.get("word_count", 0) or 0) for doc in docs)

    print(f"Loaded {total_docs} documents", flush=True)
    print(f"Total input words: {total_input_words:,}", flush=True)

    total_chunks = 0
    skipped_docs = 0
    processed_words = 0
    report_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, doc in enumerate(docs, start=1):
            source_path = str(doc.get("source_path", "") or "")
            file_type = str(doc.get("file_type", "") or "")
            doc_word_count = int(doc.get("word_count", 0) or 0)
            doc_char_count = int(doc.get("char_count", 0) or 0)

            print(
                f"\n[START doc {idx}/{total_docs}] {source_path} | doc_words={doc_word_count:,}",
                flush=True,
            )

            chunks = build_chunk_objects(
                doc=doc,
                target_words=args.target_words,
                max_words=args.max_words,
                overlap_blocks=args.overlap_blocks,
                min_chunk_words=args.min_chunk_words,
                doc_index=idx,
                total_docs=total_docs,
                start_time=start_time,
                chunk_log_every=args.chunk_log_every,
                block_log_every=args.block_log_every,
            )

            if not chunks:
                skipped_docs += 1
                report_rows.append({
                    "index": idx,
                    "source_id": doc.get("source_id", ""),
                    "source_path": source_path,
                    "file_type": file_type,
                    "document_char_count": doc_char_count,
                    "document_word_count": doc_word_count,
                    "chunk_count": 0,
                    "avg_chunk_words": 0,
                    "max_chunk_words": 0,
                    "min_chunk_words": 0,
                    "status": "skipped_no_chunks",
                })
            else:
                for chunk in chunks:
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                chunk_word_counts = [int(c["chunk_word_count"]) for c in chunks]
                total_chunks += len(chunks)

                report_rows.append({
                    "index": idx,
                    "source_id": doc.get("source_id", ""),
                    "source_path": source_path,
                    "file_type": file_type,
                    "document_char_count": doc_char_count,
                    "document_word_count": doc_word_count,
                    "chunk_count": len(chunks),
                    "avg_chunk_words": round(sum(chunk_word_counts) / len(chunk_word_counts), 2),
                    "max_chunk_words": max(chunk_word_counts),
                    "min_chunk_words": min(chunk_word_counts),
                    "status": "ok",
                })

            processed_words += doc_word_count
            elapsed = time.time() - start_time

            docs_pct = safe_pct(idx, total_docs)
            words_pct = safe_pct(processed_words, total_input_words)
            eta_seconds = estimate_eta(elapsed, processed_words, total_input_words)

            should_print = (
                idx == 1
                or idx == total_docs
                or idx % args.progress_every == 0
                or len(chunks) >= 50
                or doc_word_count >= 25000
            )

            if should_print:
                status = "SKIP" if not chunks else "OK"
                chunk_count = len(chunks)
                print(
                    f"[{idx}/{total_docs} | docs {docs_pct:6.2f}% | words {words_pct:6.2f}% | "
                    f"elapsed {format_seconds(elapsed)} | eta {format_seconds(eta_seconds)}] "
                    f"{status} -> {source_path} | {chunk_count} chunks | "
                    f"doc_words={doc_word_count:,} | total_chunks={total_chunks:,}",
                    flush=True,
                )

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "source_id",
                "source_path",
                "file_type",
                "document_char_count",
                "document_word_count",
                "chunk_count",
                "avg_chunk_words",
                "max_chunk_words",
                "min_chunk_words",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    total_elapsed = time.time() - start_time

    print("\nDone.", flush=True)
    print(f"  Input JSONL:   {input_path}", flush=True)
    print(f"  Output JSONL:  {output_jsonl}", flush=True)
    print(f"  Report CSV:    {report_csv}", flush=True)
    print(f"  Documents:     {total_docs}", flush=True)
    print(f"  Chunks:        {total_chunks:,}", flush=True)
    print(f"  Skipped docs:  {skipped_docs}", flush=True)
    print(f"  Total words:   {total_input_words:,}", flush=True)
    print(f"  Runtime:       {format_seconds(total_elapsed)}", flush=True)
    print(
        f"  Settings:      target_words={args.target_words}, "
        f"max_words={args.max_words}, overlap_blocks={args.overlap_blocks}, "
        f"progress_every={args.progress_every}, chunk_log_every={args.chunk_log_every}, "
        f"block_log_every={args.block_log_every}",
        flush=True,
    )


if __name__ == "__main__":
    main()
