import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Files that are usually excellent or at least reasonable for text-based RAG
INCLUDE_MIME_EXACT = {
    "application/pdf",
    "application/json",
    "application/rtf",
    "application/xml",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
}

INCLUDE_MIME_PREFIXES = (
    "text/",
)

INCLUDE_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".text", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".rst",
    ".tex", ".xml", ".html", ".htm", ".xhtml", ".svg", ".drawio", ".mmd",
    ".py", ".ipynb", ".m", ".mat", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".java", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".sh", ".bash",
    ".zsh", ".ps1", ".sql", ".r", ".jl", ".f90", ".f95", ".dat",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".rtf",
    ".odt", ".ods", ".odp",
}

# Files that are almost never useful for text-first RAG, or require a different pipeline.
EXCLUDE_EXTENSIONS = {
    # archives / packages
    ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".iso",
    # images / media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic",
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a",
    # CAD / simulation / heavy binaries
    ".stp", ".step", ".igs", ".iges", ".x_t", ".x_b", ".sldprt", ".sldasm", ".slddrw",
    ".f3d", ".ipt", ".iam", ".dwg", ".dxf", ".stl", ".obj", ".3mf", ".blend",
    ".prt", ".asm", ".par", ".psm", ".jt", ".catpart", ".catproduct",
    ".sim", ".odb", ".op2", ".bdf", ".nas", ".inp", ".cae",
    # executables / compiled artifacts / databases
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".o", ".a", ".lib",
    ".db", ".sqlite", ".sqlite3", ".parquet", ".feather", ".h5", ".hdf5", ".pkl",
}

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

GOOD_NAME_KEYWORDS = {
    "report", "analysis", "procedure", "requirement", "requirements", "spec", "specification",
    "review", "notes", "meeting", "minutes", "summary", "postmortem", "test", "checklist",
    "calibration", "manual", "guide", "safety", "bom", "bill of materials", "drawing",
}


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_extension_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return s.where(s.eq("") | s.str.startswith("."), "." + s)


def categorize_row(row: pd.Series) -> Tuple[str, str, int]:
    mime = str(row.get("mimeType", "") or "").strip().lower()
    ext = str(row.get("extension", "") or "").strip().lower()
    name = str(row.get("name", "") or "").strip().lower()
    is_folder = bool(row.get("is_folder", False))

    score = 0

    if is_folder or mime == GOOGLE_FOLDER_MIME:
        return "exclude", "folder", -100

    if mime == GOOGLE_SHORTCUT_MIME:
        return "exclude", "shortcut", -100

    if ext in EXCLUDE_EXTENSIONS:
        return "exclude", f"excluded extension {ext}", -80

    if mime in INCLUDE_MIME_EXACT:
        score += 100

    if any(mime.startswith(prefix) for prefix in INCLUDE_MIME_PREFIXES):
        score += 90

    if ext in INCLUDE_EXTENSIONS:
        score += 70

    if any(keyword in name for keyword in GOOD_NAME_KEYWORDS):
        score += 10

    # likely useful for RAG
    if score >= 90:
        return "candidate", "text-bearing mime/extension", score

    # ambiguous but worth manual review
    if score >= 60:
        return "review", "possibly text-bearing by extension", score

    # very small unknown file types can be manually reviewed later if needed
    return "exclude", "not a strong text-RAG candidate", score


def summarize(df: pd.DataFrame, out_prefix: str) -> str:
    candidate_df = df[df["rag_status"] == "candidate"].copy()
    review_df = df[df["rag_status"] == "review"].copy()
    excluded_df = df[df["rag_status"] == "exclude"].copy()

    lines: List[str] = []
    lines.append("RAG candidate filtering summary")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Candidates: {len(candidate_df):,}")
    lines.append(f"Review:     {len(review_df):,}")
    lines.append(f"Excluded:   {len(excluded_df):,}")
    lines.append("")

    if not candidate_df.empty:
        lines.append("Top candidate MIME types:")
        mime_counts = candidate_df["mimeType"].fillna("<missing>").value_counts().head(20)
        for mime, count in mime_counts.items():
            lines.append(f"  {mime}: {count:,}")
        lines.append("")

        lines.append("Top candidate extensions:")
        ext_counts = candidate_df["extension"].fillna("<missing>").replace("", "<none>").value_counts().head(20)
        for ext, count in ext_counts.items():
            lines.append(f"  {ext}: {count:,}")
        lines.append("")

        if "path" in candidate_df.columns:
            top_level = candidate_df["path"].fillna("").astype(str).str.split("/").str[0].replace("", "<ROOT>")
            top_level_counts = top_level.value_counts().head(20)
            lines.append("Top top-level candidate folders:")
            for folder, count in top_level_counts.items():
                lines.append(f"  {folder}: {count:,}")
            lines.append("")

    reason_counts = df["rag_reason"].value_counts().head(20)
    lines.append("Top filter reasons:")
    for reason, count in reason_counts.items():
        lines.append(f"  {reason}: {count:,}")

    summary_text = "\n".join(lines)
    summary_path = f"{out_prefix}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a Google Drive inventory CSV into likely RAG candidates."
    )
    parser.add_argument(
        "--input",
        default="drive_inventory_all.csv",
        help="Path to the full inventory CSV",
    )
    parser.add_argument(
        "--out-prefix",
        default="rag_filter",
        help="Prefix for output files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"name", "extension", "is_folder", "mimeType"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["is_folder"] = normalize_bool_series(df["is_folder"])
    df["extension"] = normalize_extension_series(df["extension"])

    categories = df.apply(categorize_row, axis=1, result_type="expand")
    categories.columns = ["rag_status", "rag_reason", "rag_score"]
    df = pd.concat([df, categories], axis=1)

    candidate_df = df[df["rag_status"] == "candidate"].copy()
    review_df = df[df["rag_status"] == "review"].copy()
    excluded_df = df[df["rag_status"] == "exclude"].copy()

    candidate_path = f"{args.out_prefix}_candidates.csv"
    review_path = f"{args.out_prefix}_review.csv"
    excluded_path = f"{args.out_prefix}_excluded.csv"

    candidate_df.to_csv(candidate_path, index=False)
    review_df.to_csv(review_path, index=False)
    excluded_df.to_csv(excluded_path, index=False)

    summary_path = summarize(df, args.out_prefix)

    print(f"Loaded: {input_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Candidates: {len(candidate_df):,} -> {candidate_path}")
    print(f"Review:     {len(review_df):,} -> {review_path}")
    print(f"Excluded:   {len(excluded_df):,} -> {excluded_path}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()
