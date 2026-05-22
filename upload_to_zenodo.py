#!/usr/bin/env python3
"""
upload_to_zenodo.py — Upload a directory of files to a Zenodo deposition.

Workflow:
    1. export ZENODO_TOKEN=...   (get from https://zenodo.org/account/settings/applications/)
    2. python upload_to_zenodo.py create   (creates draft + reserves DOI)
    3. python upload_to_zenodo.py upload --record-id <id> --dir <bundle_dir>
    4. Review draft + publish at https://zenodo.org/me/uploads

Uses streaming uploads via the bucket API. Resumable on interruption (re-running
skips files already in the deposition with matching size).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")

if not TOKEN:
    sys.exit("ERROR: set ZENODO_TOKEN (https://zenodo.org/account/settings/applications/)")


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def create_draft(title: str, description: str, upload_type: str, license_id: str) -> dict:
    """Create an empty deposition draft and return the API response."""
    keywords = [
        "GNSS", "RINEX", "u-blox ZED-F9P", "shipborne",
        "Antarctica", "R/V Laura Bassi", "multi-constellation",
        "GPS", "Galileo", "GLONASS", "BeiDou",
    ]
    if upload_type == "software":
        keywords += ["Python", "Snakemake", "data pipeline", "Hatanaka", "PRIDE PPP-AR"]

    metadata = {
        "metadata": {
            "title": title,
            "upload_type": upload_type,
            "description": description,
            "creators": [{"name": "Bertalanic, Bertrand"}],
            "access_right": "open",
            "license": license_id,
            "keywords": keywords,
        }
    }
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions",
        headers={**headers(), "Content-Type": "application/json"},
        data=json.dumps(metadata),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def get_deposition(record_id: int) -> dict:
    r = requests.get(f"{ZENODO_API}/deposit/depositions/{record_id}",
                     headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def existing_files(record_id: int) -> dict:
    """Return {filename: size} for files already uploaded."""
    r = requests.get(f"{ZENODO_API}/deposit/depositions/{record_id}/files",
                     headers=headers(), timeout=30)
    r.raise_for_status()
    return {f["filename"]: f["filesize"] for f in r.json()}


def upload_file(bucket_url: str, path: Path) -> None:
    """Stream-upload one file to the bucket URL."""
    size = path.stat().st_size
    print(f"  uploading {path.name} ({human(size)})...", flush=True)
    with path.open("rb") as fh:
        r = requests.put(
            f"{bucket_url}/{path.name}",
            data=fh,
            headers=headers(),
            timeout=None,
        )
    r.raise_for_status()
    print(f"  done: {path.name}", flush=True)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


DEFAULTS = {
    "dataset": {
        "title": ("R/V Laura Bassi shipborne GNSS dataset, Trieste–Antarctica–"
                  "Trieste (2025-09-26 to 2026-04-29) — raw UBX + RINEX 3.04"),
        "description": (
            "Multi-constellation, multi-band GNSS dataset from a 216-day "
            "shipborne campaign (R/V Laura Bassi, Trieste–Antarctica–Trieste, "
            "2025-09-26 to 2026-04-29). u-blox ZED-F9P-15B receiver, 2 Hz raw "
            "observables, broadcast navigation, RF spectrum (MON-SPAN), and "
            "receiver health telemetry. Includes raw u-blox UBX binaries and "
            "Hatanaka-compressed RINEX 3.04. Derived products in companion "
            "record (see Related identifiers)."
        ),
        "license": "cc-by-4.0",
    },
    "derived": {
        "title": ("R/V Laura Bassi shipborne GNSS dataset — derived products "
                  "(Level 2)"),
        "description": (
            "Derived products from the R/V Laura Bassi shipborne GNSS campaign "
            "(2025-09-26 to 2026-04-29): cruise track, multipath M1/M2, RF "
            "spectrogram Zarr, TEC time series, scintillation proxy, daily QC "
            "statistics, milestone crossings, and figures. Produced from the "
            "raw UBX dataset in the companion record."
        ),
        "license": "cc-by-4.0",
    },
    "software": {
        "title": ("antarctica-gnss-pipeline: processing code for the "
                  "R/V Laura Bassi shipborne GNSS dataset"),
        "description": (
            "Python + Snakemake pipeline that processes u-blox ZED-F9P-15B UBX "
            "binaries into Parquet, Zarr, and ESSD-ready figures. Includes a "
            "custom NumPy fast-path UBX parser, RINEX 3.04 conversion via "
            "convbin, Hatanaka compression, kinematic PRIDE PPP-AR wrapper, "
            "and analysis modules for trajectory, multipath, RF spectrum, TEC, "
            "and QC. Used to produce the companion data record."
        ),
        "license": "mit",
    },
}


def cmd_create(args):
    if args.type not in DEFAULTS:
        sys.exit(f"ERROR: --type must be one of {list(DEFAULTS)}")
    d = DEFAULTS[args.type]
    title = args.title or d["title"]
    description = args.description or d["description"]
    license_id = args.license or d["license"]
    upload_type = "software" if args.type == "software" else "dataset"

    dep = create_draft(title, description, upload_type, license_id)
    rec_id = dep["id"]
    doi = dep["metadata"].get("prereserve_doi", {}).get("doi", "(reserved on publish)")
    print(f"Created {args.type} draft record {rec_id}")
    print(f"  Title:          {title}")
    print(f"  Type:           {upload_type}")
    print(f"  License:        {license_id}")
    print(f"  DOI (reserved): {doi}")
    print(f"  Edit at:        https://zenodo.org/uploads/{rec_id}")
    print(f"  Bucket URL:     {dep['links']['bucket']}")
    print()
    print("To upload files:")
    print(f"  python upload_to_zenodo.py upload --record-id {rec_id} --dir <bundle_dir>")


def cmd_upload(args):
    dep = get_deposition(args.record_id)
    if dep["state"] != "unsubmitted":
        sys.exit(f"ERROR: record {args.record_id} state is '{dep['state']}', not 'unsubmitted'")
    bucket = dep["links"]["bucket"]
    have = existing_files(args.record_id)

    bundle_dir = Path(args.dir).resolve()
    files = sorted(p for p in bundle_dir.glob("*") if p.is_file())
    files += sorted(p for p in (bundle_dir / "metadata").glob("*") if p.is_file())

    print(f"Record {args.record_id} bucket: {bucket}")
    print(f"Bundle dir:                    {bundle_dir}")
    print(f"Files to consider:             {len(files)}")
    print(f"Already uploaded:              {len(have)}")
    print()

    for p in files:
        sz = p.stat().st_size
        if p.name in have and have[p.name] == sz:
            print(f"  skip (already uploaded): {p.name} ({human(sz)})")
            continue
        upload_file(bucket, p)

    print()
    print("Upload complete.")
    print(f"Review and publish: https://zenodo.org/uploads/{args.record_id}")


def cmd_status(args):
    dep = get_deposition(args.record_id)
    files = dep.get("files", [])
    print(f"Record:  {args.record_id}")
    print(f"State:   {dep['state']}")
    print(f"Title:   {dep['metadata'].get('title', '(none)')}")
    print(f"DOI:     {dep['metadata'].get('prereserve_doi', {}).get('doi', dep.get('doi', '(unset)'))}")
    print(f"Files:   {len(files)}")
    total = 0
    for f in files:
        total += f["filesize"]
        print(f"  {human(f['filesize']):>10}  {f['filename']}")
    print(f"  ───────────────────")
    print(f"  {human(total):>10}  total")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create an empty draft deposition")
    p_create.add_argument("--type", default="dataset",
                          choices=["dataset", "derived", "software"],
                          help="Record type (default: dataset = Record A)")
    p_create.add_argument("--title", help="Override default record title")
    p_create.add_argument("--description", help="Override default record description")
    p_create.add_argument("--license", help="Override default license id (e.g. cc-by-4.0, mit, apache-2.0)")
    p_create.set_defaults(func=cmd_create)

    p_upload = sub.add_parser("upload", help="Upload files from a bundle directory")
    p_upload.add_argument("--record-id", type=int, required=True)
    p_upload.add_argument("--dir", required=True, help="Bundle directory")
    p_upload.set_defaults(func=cmd_upload)

    p_status = sub.add_parser("status", help="Show deposition status + file list")
    p_status.add_argument("--record-id", type=int, required=True)
    p_status.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
