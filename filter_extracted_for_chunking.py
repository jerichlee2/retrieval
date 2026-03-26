import argparse
import csv
import hashlib
import json
import os
from typing import Iterable, List, Tuple

DEFAULT_SKIP_SUBSTRINGS = [
    "textbook",
    "handbook",
    "asme",
    "catalog",
    "pirated textbooks",
    "raw data",
    "test csv",
]


def guess_file_type(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def should_skip_path(path_lower: str, skip_substrings: List[str]) -> Tuple[bool, str]:
    for s in skip_substrings:
        if s and s in path_lower:
            return True, f"path_contains:{s}"
    return False, ""


def process_file(
    input_jsonl: str,
    output_jsonl: str,
    report_csv: str,
    skip_substrings: List[str],
    max_words: int,
    dedupe_by_text: bool,
    limit: int | None,
) -> None:
    seen_hashes = set()
    kept = 0
    skipped = 0
    processed = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout, \
         open(report_csv, "w", encoding="utf-8", newline="") as freport:

        writer = csv.writer(freport)
        writer.writerow([
            "index",
            "source_id",
            "source_path",
            "file_type",
            "char_count",
            "word_count",
            "action",
            "reason",
        ])

        for idx, line in enumerate(fin, start=1):
            if limit is not None and processed >= limit:
                break

            line = line.strip()
            if not line:
                continue

            processed += 1
            row = json.loads(line)
            source_id = row.get("source_id", "")
            source_path = row.get("source_path", "")
            file_type = row.get("file_type") or guess_file_type(source_path)
            text = row.get("text", "") or ""
            char_count = len(text)
            word_count = len(text.split())
            path_lower = source_path.lower()

            skip, reason = should_skip_path(path_lower, skip_substrings)
            if skip:
                skipped += 1
                writer.writerow([
                    idx,
                    source_id,
                    source_path,
                    file_type,
                    char_count,
                    word_count,
                    "skipped",
                    reason,
                ])
                print(f"[{processed}] SKIP {source_path} ({reason})", flush=True)
                continue

            if word_count > max_words:
                skipped += 1
                writer.writerow([
                    idx,
                    source_id,
                    source_path,
                    file_type,
                    char_count,
                    word_count,
                    "skipped",
                    f"word_count_gt:{max_words}",
                ])
                print(
                    f"[{processed}] SKIP {source_path} (word_count={word_count} > {max_words})",
                    flush=True,
                )
                continue

            if dedupe_by_text:
                digest = text_hash(normalize_whitespace(text))
                if digest in seen_hashes:
                    skipped += 1
                    writer.writerow([
                        idx,
                        source_id,
                        source_path,
                        file_type,
                        char_count,
                        word_count,
                        "skipped",
                        "duplicate_text_hash",
                    ])
                    print(f"[{processed}] SKIP {source_path} (duplicate_text_hash)", flush=True)
                    continue
                seen_hashes.add(digest)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            writer.writerow([
                idx,
                source_id,
                source_path,
                file_type,
                char_count,
                word_count,
                "kept",
                "",
            ])

            if processed % 50 == 0:
                print(
                    f"Progress: processed={processed} kept={kept} skipped={skipped}",
                    flush=True,
                )

    print("\nDone.", flush=True)
    print(f"  Input JSONL:  {input_jsonl}", flush=True)
    print(f"  Output JSONL: {output_jsonl}", flush=True)
    print(f"  Report CSV:   {report_csv}", flush=True)
    print(f"  Processed:    {processed}", flush=True)
    print(f"  Kept:         {kept}", flush=True)
    print(f"  Skipped:      {skipped}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter extracted Tier A documents before chunking."
    )
    parser.add_argument(
        "--input-jsonl",
        default="extracted_documents_tier_a.jsonl",
        help="Path to extracted Tier A JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="extracted_documents_tier_a_filtered.jsonl",
        help="Path to write filtered JSONL.",
    )
    parser.add_argument(
        "--report-csv",
        default="filter_extracted_for_chunking_report.csv",
        help="Path to write keep/skip report CSV.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50000,
        help="Skip documents with more than this many words.",
    )
    parser.add_argument(
        "--skip-substring",
        action="append",
        default=None,
        help="Case-insensitive substring to skip if found in source_path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-dedupe-by-text",
        action="store_true",
        help="Disable deduplication by normalized text hash.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_substrings = args.skip_substring if args.skip_substring is not None else DEFAULT_SKIP_SUBSTRINGS
    skip_substrings = [s.lower() for s in skip_substrings]

    process_file(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        report_csv=args.report_csv,
        skip_substrings=skip_substrings,
        max_words=args.max_words,
        dedupe_by_text=not args.no_dedupe_by_text,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
