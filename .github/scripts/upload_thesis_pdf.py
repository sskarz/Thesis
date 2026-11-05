#!/usr/bin/env python3
"""Replace the contents of an existing Google Drive file with a local PDF.

Used by .github/workflows/thesis-drive-upload.yml. The Drive file keeps its id,
name, sharing and link; only its contents change (Drive API files.update with a
media upload). Afterwards the script checks that Drive's md5Checksum matches the
local file and exits non-zero if it does not.

Environment:
  GDRIVE_SA_KEY    JSON key of a Google service account (from a repo secret).
                   The Drive file must be shared with that account as Editor.
  GDRIVE_FILE_ID   Id of the Drive file to overwrite.
  PDF_PATH         Local file to upload (default thesis/Thesis_CSULA_Format.pdf).
"""
import hashlib
import json
import os
import sys

SCOPES = ["https://www.googleapis.com/auth/drive"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_contents(service, file_id, path, media_cls):
    """Overwrite file_id's contents with path; return Drive's file metadata."""
    before = service.files().get(fileId=file_id, fields="id,name,mimeType,md5Checksum").execute()
    local_md5 = md5_of(path)
    if before.get("md5Checksum") == local_md5:
        print(f"{before['name']} ({file_id}) already matches {path}; nothing to upload.")
        return before
    media = media_cls(path, mimetype="application/pdf", resumable=True)
    after = (
        service.files()
        .update(fileId=file_id, media_body=media, fields="id,name,md5Checksum,size,modifiedTime")
        .execute()
    )
    if after.get("md5Checksum") != local_md5:
        raise SystemExit(
            f"Drive md5 {after.get('md5Checksum')} does not match local {local_md5} after upload"
        )
    print(f"Replaced {after['name']} ({after['id']}): {after.get('size')} bytes, "
          f"md5 {local_md5}, modified {after.get('modifiedTime')}.")
    return after


def main():
    key = os.environ.get("GDRIVE_SA_KEY", "").strip()
    file_id = os.environ.get("GDRIVE_FILE_ID", "").strip()
    path = os.environ.get("PDF_PATH", "thesis/Thesis_CSULA_Format.pdf")
    if not key:
        sys.exit("GDRIVE_SA_KEY is not set (add the service account JSON key as a repo secret).")
    if not file_id:
        sys.exit("GDRIVE_FILE_ID is not set.")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found.")

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = service_account.Credentials.from_service_account_info(json.loads(key), scopes=SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    replace_contents(service, file_id, path, MediaFileUpload)


if __name__ == "__main__":
    main()
