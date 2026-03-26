import argparse
import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    # Best-effort text-first exports for RAG prep
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    # Keep all sheets together; later pipeline can extract text/tables
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    # PDF works well for later text extraction and preserves slide structure
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
}

UNSUPPORTED_GOOGLE_MIME_PREFIX = "application/vnd.google-apps."


def authenticate():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def sanitize_component(name: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    cleaned = str(name)
    for ch in bad:
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or "unnamed"


def safe_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes"}


def maybe_int(v) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def choose_download_plan(name: str, mime_type: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (mode, export_mime, output_extension)
    mode in {"download", "export", "skip"}
    """
    if mime_type == GOOGLE_FOLDER_MIME:
        return "skip", None, None

    if mime_type == GOOGLE_SHORTCUT_MIME:
        return "skip", None, None

    if mime_type in EXPORT_MAP:
        export_mime, ext = EXPORT_MAP[mime_type]
        return "export", export_mime, ext

    if mime_type.startswith(UNSUPPORTED_GOOGLE_MIME_PREFIX):
        return "skip", None, None

    return "download", None, None


def build_output_path(base_dir: Path, row: pd.Series, mirror_paths: bool, force_suffix_id: bool, ext_override: Optional[str]) -> Path:
    name = sanitize_component(row.get("name", "unnamed"))
    file_id = str(row.get("id", "")).strip()

    if ext_override:
        if not name.lower().endswith(ext_override.lower()):
            name = f"{name}{ext_override}"

    if force_suffix_id and file_id:
        stem, suffix = os.path.splitext(name)
        name = f"{stem}__{file_id}{suffix}"

    if mirror_paths:
        path_value = str(row.get("path", "")).strip()
        if path_value:
            parts = [sanitize_component(part) for part in path_value.split("/") if part.strip()]
            if parts:
                parts[-1] = name
                return base_dir.joinpath(*parts)

    return base_dir / name


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def download_or_export_file(service, row: pd.Series, out_path: Path, chunk_size: int) -> Tuple[str, Optional[int]]:
    file_id = str(row.get("id", "")).strip()
    mime_type = str(row.get("mimeType", "")).strip()
    name = str(row.get("name", "")).strip()

    mode, export_mime, _ = choose_download_plan(name, mime_type)
    if mode == "skip":
        return "skipped", None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "export":
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)

    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    size_bytes = out_path.stat().st_size if out_path.exists() else None
    return mode, size_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/export likely RAG candidate files from a Drive inventory CSV.")
    parser.add_argument("--input", default="rag_filter_candidates.csv", help="Input CSV of candidate files")
    parser.add_argument("--output-dir", default="rag_downloads", help="Directory to save downloaded/exported files")
    parser.add_argument("--manifest", default="download_manifest.csv", help="CSV manifest written during run")
    parser.add_argument("--mirror-paths", action="store_true", help="Recreate the Drive relative path locally")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N rows")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files whose output path already exists")
    parser.add_argument("--suffix-id", action="store_true", help="Append Drive file ID to local filenames to avoid collisions")
    parser.add_argument("--min-size-bytes", type=int, default=None, help="Only process files at least this large (when size is known)")
    parser.add_argument("--max-size-bytes", type=int, default=None, help="Only process files at most this large (when size is known)")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="Download chunk size in bytes")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between file transfers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without transferring data")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV has no rows.")

    if "rag_status" in df.columns:
        df = df[df["rag_status"].fillna("").astype(str).str.lower() == "candidate"].copy()

    if "is_folder" in df.columns:
        df = df[~df["is_folder"].map(safe_bool)].copy()

    if args.min_size_bytes is not None and "size_bytes" in df.columns:
        df = df[df["size_bytes"].map(maybe_int).fillna(args.min_size_bytes) >= args.min_size_bytes].copy()

    if args.max_size_bytes is not None and "size_bytes" in df.columns:
        df = df[df["size_bytes"].map(maybe_int).fillna(0) <= args.max_size_bytes].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    total = len(df)
    if total == 0:
        raise ValueError("No rows left to process after filtering.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)

    print(f"Input rows to process: {total}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")

    service = None if args.dry_run else authenticate()

    manifest_rows = []
    downloaded = 0
    skipped = 0
    errored = 0
    start = time.time()

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        name = str(row.get("name", ""))
        mime_type = str(row.get("mimeType", ""))
        file_id = str(row.get("id", ""))
        drive_path = str(row.get("path", ""))

        mode, export_mime, ext_override = choose_download_plan(name, mime_type)
        out_path = build_output_path(
            base_dir=output_dir,
            row=row,
            mirror_paths=args.mirror_paths,
            force_suffix_id=args.suffix_id,
            ext_override=ext_override,
        )

        if not args.skip_existing:
            out_path = ensure_unique_path(out_path)

        status = "pending"
        size_written = None
        error_message = ""

        if mode == "skip":
            status = "skipped"
            skipped += 1
        elif args.skip_existing and out_path.exists():
            status = "skipped_existing"
            skipped += 1
        elif args.dry_run:
            status = f"would_{mode}"
        else:
            try:
                status, size_written = download_or_export_file(
                    service=service,
                    row=row,
                    out_path=out_path,
                    chunk_size=args.chunk_size,
                )
                downloaded += 1
            except HttpError as e:
                status = "error"
                errored += 1
                error_message = f"HTTP {getattr(e, 'status_code', 'error')}: {e}"
            except Exception as e:
                status = "error"
                errored += 1
                error_message = str(e)

        manifest_rows.append({
            "index": idx,
            "id": file_id,
            "name": name,
            "path": drive_path,
            "mimeType": mime_type,
            "webViewLink": row.get("webViewLink", ""),
            "mode": mode,
            "export_mime": export_mime or "",
            "local_path": str(out_path),
            "status": status,
            "size_written_bytes": size_written,
            "error": error_message,
        })

        if idx % 25 == 0 or idx == total:
            elapsed = time.time() - start
            pct = 100.0 * idx / total
            print(
                f"[{idx}/{total} | {pct:6.2f}%] downloaded={downloaded} skipped={skipped} errors={errored} | current={drive_path or name}",
                flush=True,
            )
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    elapsed = time.time() - start
    print("\nDone.")
    print(f"Downloaded/exported: {downloaded}")
    print(f"Skipped:             {skipped}")
    print(f"Errors:              {errored}")
    print(f"Elapsed seconds:     {elapsed:.1f}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
