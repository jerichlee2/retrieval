#!/usr/bin/env python3
"""
Filter extracted JSONL before chunking.

What it does:
- skips obvious heavy hitters by path substring
- skips documents over a max word count
- deduplicates by normalized text hash
- supports force-keeping specific documents via --keep-substring

Typical use:
    python filter_extracted_for_chunking_whitelist.py \
      --input-jsonl extracted_documents_tier_a.jsonl \
      --output-jsonl extracted_documents_tier_a_filtered.jsonl \
      --report-csv filter_extracted_for_chunking_report.csv \
      --keep-substring "Liquid Rocketry at Illinois Technical Handbook" \
      --keep-substring "Injector sizing.pdf"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_SKIP_SUBSTRINGS = [
    "textbook",
    "handbook",
    "asme",
    "catalog",
    "pirated textbooks",
    "raw data",
    "test csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter extracted JSONL before chunking.")
    parser.add_argument(
        "--input-jsonl",
        default="extracted_documents_tier_a.jsonl",
        help="Input extracted JSONL path",
    )
    parser.add_argument(
        "--output-jsonl",
        default="extracted_documents_tier_a_filtered.jsonl",
        help="Filtered output JSONL path",
    )
    parser.add_argument(
        "--report-csv",
        default="filter_extracted_for_chunking_report.csv",
        help="CSV report path",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50000,
        help="Skip documents whose word_count exceeds this threshold unless force-kept.",
    )
    parser.add_argument(
        "--skip-substring",
        action="append",
        default=None,
        help="Additional lowercase-insensitive path substring to skip. Repeatable.",
    )
    parser.add_argument(
        "--keep-substring",
        action="append",
        default=None,
        help="Lowercase-insensitive path substring to force-keep. Repeatable. Force-kept docs bypass skip, max-words, and duplicate filters.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable text-hash deduplication.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N records. Default: 50",
    )
    return parser.parse_args()


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


def normalize_spaces(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_text_hash(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def path_matches_any(path_lower: str, substrings: List[str]) -> Optional[str]:
    for s in substrings:
        if s and s in path_lower:
            return s
    return None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    report_path = Path(args.report_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    skip_substrings = list(DEFAULT_SKIP_SUBSTRINGS)
    if args.skip_substring:
        skip_substrings.extend(args.skip_substring)
    skip_substrings = [s.lower() for s in skip_substrings if s]

    keep_substrings = [s.lower() for s in (args.keep_substring or []) if s]

    seen_hashes = set()
    kept = 0
    skipped = 0
    processed = 0
    report_rows: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            processed += 1

            row = json.loads(line)
            source_path = str(row.get("source_path", "") or "")
            path_lower = source_path.lower()
            text = str(row.get("text", "") or "")
            word_count = int(row.get("word_count", 0) or 0)
            if word_count <= 0 and text:
                word_count = len(text.split())
                row["word_count"] = word_count

            keep_match = path_matches_any(path_lower, keep_substrings)
            skip_match = path_matches_any(path_lower, skip_substrings)

            decision = "keep"
            reason = "kept"

            if keep_match is not None:
                decision = "keep"
                reason = f"force_keep:{keep_match}"
            elif skip_match is not None:
                decision = "skip"
                reason = f"path_contains:{skip_match}"
            elif args.max_words > 0 and word_count > args.max_words:
                decision = "skip"
                reason = f"word_count={word_count} > {args.max_words}"
            elif not args.no_dedupe:
                text_hash = normalized_text_hash(text)
                if text_hash in seen_hashes:
                    decision = "skip"
                    reason = "duplicate_text_hash"
                else:
                    seen_hashes.add(text_hash)

            report_rows.append({
                "index": idx,
                "source_id": row.get("source_id", ""),
                "source_path": source_path,
                "file_type": row.get("file_type", ""),
                "word_count": word_count,
                "decision": decision,
                "reason": reason,
            })

            if decision == "keep":
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1
                print(f"[{idx}] SKIP {source_path} ({reason})", flush=True)

            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"Progress: processed={processed} kept={kept} skipped={skipped}",
                    flush=True,
                )

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "source_id",
                "source_path",
                "file_type",
                "word_count",
                "decision",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nDone.", flush=True)
    print(f"  Input JSONL:   {input_path}", flush=True)
    print(f"  Output JSONL:  {output_path}", flush=True)
    print(f"  Report CSV:    {report_path}", flush=True)
    print(f"  Processed:     {processed}", flush=True)
    print(f"  Kept:          {kept}", flush=True)
    print(f"  Skipped:       {skipped}", flush=True)
    print(f"  Max words:     {args.max_words}", flush=True)
    print(f"  Dedupe:        {'off' if args.no_dedupe else 'on'}", flush=True)
    print(f"  Keep rules:    {keep_substrings if keep_substrings else 'none'}", flush=True)


if __name__ == "__main__":
    main()
