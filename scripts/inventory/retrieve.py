import os
import time
from typing import List, Dict, Optional

import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Change this to the folder ID you want to inventory.
ROOT_FOLDER_ID = "1x79jNLrhtBN5bxteRxDDYenyFC8-rW8K"

# Output file: contains EVERYTHING found under the root folder
OUTPUT_CSV = "drive_inventory_all.csv"

FOLDER_MIME = "application/vnd.google-apps.folder"
PAGE_SIZE = 1000

# Set this to 0.0 for max speed. Increase slightly only if you hit rate limits.
FOLDER_SLEEP_SECONDS = 0.0

FIELDS = (
    "nextPageToken, files("
    "id, name, mimeType, size, createdTime, modifiedTime, "
    "webViewLink, parents, driveId, owners(displayName,emailAddress)"
    ")"
)


def authenticate():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def safe_get_owner_string(owners: Optional[List[Dict]]) -> str:
    if not owners:
        return ""

    vals = []
    for owner in owners:
        display = owner.get("displayName", "")
        email = owner.get("emailAddress", "")
        if display and email:
            vals.append(f"{display} <{email}>")
        elif display:
            vals.append(display)
        elif email:
            vals.append(email)

    return "; ".join(vals)


def get_extension(name: str) -> str:
    _, ext = os.path.splitext(name)
    return ext.lower()


def elapsed_str(start_time: float) -> str:
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def list_children(service, folder_id: str, folder_path: str, stats: Dict) -> List[Dict]:
    all_files = []
    page_token = None
    page_num = 0

    while True:
        page_num += 1
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=PAGE_SIZE,
            pageToken=page_token,
            fields=FIELDS,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        batch = response.get("files", [])
        all_files.extend(batch)

        stats["api_pages"] += 1

        print(
            f"[{elapsed_str(stats['start_time'])}] "
            f"Fetched page {page_num} for "
            f"'{folder_path or '<ROOT>'}' -> {len(batch)} items "
            f"(API pages so far: {stats['api_pages']})",
            flush=True,
        )

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return all_files


def walk_folder(service, folder_id: str, path: str = "", stats: Optional[Dict] = None) -> List[Dict]:
    if stats is None:
        stats = {
            "start_time": time.time(),
            "folders_seen": 0,
            "files_seen": 0,
            "rows_seen": 0,
            "api_pages": 0,
        }

    rows = []

    children = list_children(service, folder_id, path, stats)

    stats["folders_seen"] += 1
    print(
        f"[{elapsed_str(stats['start_time'])}] "
        f"Scanning folder #{stats['folders_seen']}: '{path or '<ROOT>'}' "
        f"with {len(children)} direct children "
        f"(files so far: {stats['files_seen']}, rows so far: {stats['rows_seen']})",
        flush=True,
    )

    for f in children:
        file_id = f.get("id", "")
        name = f.get("name", "")
        mime_type = f.get("mimeType", "")
        size = f.get("size", "")
        created = f.get("createdTime", "")
        modified = f.get("modifiedTime", "")
        link = f.get("webViewLink", "")
        parents = ",".join(f.get("parents", []))
        drive_id = f.get("driveId", "")
        owners = safe_get_owner_string(f.get("owners"))

        current_path = f"{path}/{name}" if path else name
        is_folder = mime_type == FOLDER_MIME

        rows.append({
            "path": current_path,
            "name": name,
            "extension": get_extension(name),
            "is_folder": is_folder,
            "id": file_id,
            "mimeType": mime_type,
            "size_bytes": size,
            "createdTime": created,
            "modifiedTime": modified,
            "owners": owners,
            "parents": parents,
            "driveId": drive_id,
            "webViewLink": link,
        })

        stats["rows_seen"] += 1

        if is_folder:
            if FOLDER_SLEEP_SECONDS > 0:
                time.sleep(FOLDER_SLEEP_SECONDS)

            rows.extend(walk_folder(service, file_id, current_path, stats))
        else:
            stats["files_seen"] += 1

            if stats["files_seen"] % 500 == 0:
                print(
                    f"[{elapsed_str(stats['start_time'])}] "
                    f"Progress: {stats['files_seen']} files, "
                    f"{stats['folders_seen']} folders, "
                    f"{stats['rows_seen']} total rows",
                    flush=True,
                )

    return rows


def main():
    start_time = time.time()
    print("Authenticating with Google Drive API...", flush=True)
    service = authenticate()
    print("Authentication successful.", flush=True)
    print(f"Starting crawl from ROOT_FOLDER_ID = {ROOT_FOLDER_ID}", flush=True)

    rows = walk_folder(service, ROOT_FOLDER_ID)

    print(
        f"[{elapsed_str(start_time)}] Crawl complete. Building DataFrame...",
        flush=True,
    )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["size_bytes"] = pd.to_numeric(df["size_bytes"], errors="coerce")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[{elapsed_str(start_time)}] Wrote {len(df)} rows to {OUTPUT_CSV}", flush=True)

    files_only = df[df["is_folder"] == False].copy()
    print(f"Files:   {len(files_only)}", flush=True)
    print(f"Folders: {len(df) - len(files_only)}", flush=True)

    if not files_only.empty:
        print("\nTop 25 largest files:", flush=True)
        cols = ["path", "mimeType", "size_bytes", "modifiedTime", "webViewLink"]
        print(
            files_only.sort_values("size_bytes", ascending=False)[cols]
            .head(25)
            .to_string(index=False),
            flush=True,
        )

    print(f"\nTotal runtime: {elapsed_str(start_time)}", flush=True)


if __name__ == "__main__":
    main()