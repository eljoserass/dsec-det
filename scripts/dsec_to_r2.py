#!/usr/bin/env python3
"""
Downloads DSEC zips, extracts them, and uploads the raw files to Cloudflare R2.
Runs multiple (split * modality) pipelines in parallel.
Each pipeline: download zip → extract to tmpdir → upload files → clean up.

Writes progress.txt so you can check status while it runs in the background.
"""

import boto3
import requests
import zipfile
import tempfile
import threading
import time
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
R2_ACCOUNT_ID  = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY  = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY  = os.environ["R2_SECRET_KEY"]
R2_BUCKET      = os.environ["R2_BUCKET"]
R2_PREFIX      = "dsec"           # root prefix inside the bucket

# What to download.
# det  → events | images | calibration | object_detections | left_images_distorted
# main → events | images | disparity   | optical_flow      | calibration
DATASET    = "main"
MODALITIES = ["events", "calibration", "images"]
SPLITS     = ["train", "test"]

# Parallelism
PIPELINE_WORKERS = 3   # how many (split, modality) pairs run simultaneously
UPLOAD_WORKERS   = 8   # parallel file uploads per extracted zip
CHUNK_MB         = 32   # streaming download chunk size in MB
# ---------------------------------------------------------------------------

DATASET_URLS = {
    "det":  "https://download.ifi.uzh.ch/rpg/DSEC/{split}_object_detection_coarse/{split}_{mod}.zip",
    "main": "https://download.ifi.uzh.ch/rpg/DSEC/{split}_coarse/{split}_{mod}.zip",
}

PROGRESS_FILE = Path(__file__).parent / "progress.txt"
SUMMARY_FILE  = Path(__file__).parent / "download_summary.txt"

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto",
)

# Thread-safe state for progress tracking
_lock = threading.Lock()
_pipeline_stats: list[dict] = []
_progress: dict[str, str] = {}  # pipeline_name -> status string


def preflight_check():
    errors = []
    warnings = []

    # 1. Bucket name — underscores are not valid in R2/S3
    if "_" in R2_BUCKET:
        errors.append(f"R2_BUCKET='{R2_BUCKET}' contains underscores — use hyphens instead")

    # 2. Required env vars are non-empty
    for var, val in [("R2_ACCOUNT_ID", R2_ACCOUNT_ID), ("R2_ACCESS_KEY", R2_ACCESS_KEY),
                     ("R2_SECRET_KEY", R2_SECRET_KEY), ("R2_BUCKET", R2_BUCKET)]:
        if not val:
            errors.append(f"{var} is not set")

    # 3. Bucket reachable
    try:
        s3.head_bucket(Bucket=R2_BUCKET)
    except Exception as e:
        errors.append(f"Cannot reach bucket '{R2_BUCKET}': {e}")

    # 4. Disk space on /workspace (need at least ~50 GB free for parallel downloads)
    stat = shutil.disk_usage("/workspace")
    free_gb = stat.free / 1e9
    if free_gb < 20:
        errors.append(f"Low disk space on /workspace: {free_gb:.1f} GB free (need ≥ 20 GB)")
    elif free_gb < 60:
        warnings.append(f"Disk space on /workspace: {free_gb:.1f} GB free — may be tight with {PIPELINE_WORKERS} parallel workers")

    # 5. First download URL is reachable
    sample_url = DATASET_URLS[DATASET].format(split=SPLITS[0], mod=MODALITIES[0])
    try:
        r = requests.head(sample_url, timeout=10, allow_redirects=True)
        if r.status_code >= 400:
            errors.append(f"Download URL returned HTTP {r.status_code}: {sample_url}")
    except Exception as e:
        errors.append(f"Download URL unreachable ({e}): {sample_url}")

    for w in warnings:
        print(f"  [WARN]  {w}")
    for e in errors:
        print(f"  [ERROR] {e}")

    if errors:
        raise SystemExit("Preflight failed — fix the above errors before running.")
    print("  Preflight OK\n")


def _write_progress():
    """Overwrite progress.txt with current snapshot. Call with _lock held."""
    done = sum(1 for v in _progress.values() if v.startswith("done"))
    total = len(_progress)
    lines = [f"=== DSEC → R2 progress ({done}/{total} pipelines done) ===", ""]
    for name in sorted(_progress):
        lines.append(f"  {name:<35} {_progress[name]}")
    lines.append("")
    PROGRESS_FILE.write_text("\n".join(lines) + "\n")


def _key_exists(r2_key: str) -> bool:
    """Check if an object already exists in R2."""
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=r2_key)
        return True
    except s3.exceptions.ClientError:
        return False


def upload_one(local_path: Path, r2_key: str):
    if _key_exists(r2_key):
        print(f"    — {r2_key} (exists, skipped)")
        return False
    s3.upload_file(str(local_path), R2_BUCKET, r2_key)
    print(f"    ✓ {r2_key}")
    return True


def pipeline(split: str, mod: str):
    url = DATASET_URLS[DATASET].format(split=split, mod=mod)
    name = f"{split}/{mod}"
    tag = f"[{name}]"
    print(f"{tag} starting — {url}")

    stat = {
        "pipeline":          name,
        "download_bytes":    0,
        "download_secs":     0.0,
        "extract_secs":      0.0,
        "upload_file_count": 0,
        "upload_secs":       0.0,
        "total_secs":        0.0,
    }

    t_total_start = time.monotonic()

    with _lock:
        _progress[name] = "downloading..."
        _write_progress()

    with tempfile.TemporaryDirectory(dir="/workspace") as tmpdir:
        zip_path = Path(tmpdir) / f"{split}_{mod}.zip"

        # 1. Stream download to disk (with resume on connection drop)
        MAX_RETRIES = 10
        RETRY_DELAY = 5  # seconds between retries
        t0 = time.monotonic()
        downloaded = 0
        last_progress_gb = 0
        for attempt in range(1, MAX_RETRIES + 1):
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
            file_mode = "ab" if downloaded > 0 else "wb"
            try:
                with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                    if downloaded > 0 and r.status_code == 206:
                        print(f"\n{tag} resuming from {downloaded/1e9:.2f} GB (attempt {attempt})")
                    elif downloaded > 0 and r.status_code == 200:
                        # Server doesn't support range requests; restart
                        print(f"\n{tag} server doesn't support resume, restarting (attempt {attempt})")
                        downloaded = 0
                        last_progress_gb = 0
                        file_mode = "wb"
                    else:
                        r.raise_for_status()
                    with open(zip_path, file_mode) as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_MB * 1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            gb = downloaded / 1e9
                            if gb - last_progress_gb >= 1.0:
                                with _lock:
                                    _progress[name] = f"downloading... {gb:.1f} GB"
                                    _write_progress()
                                last_progress_gb = gb
                            print(f"{tag} downloaded {gb:.2f} GB", end="\r")
                break  # download completed successfully
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as exc:
                if attempt == MAX_RETRIES:
                    raise
                print(f"\n{tag} connection dropped at {downloaded/1e9:.2f} GB "
                      f"({exc.__class__.__name__}), retrying in {RETRY_DELAY}s "
                      f"(attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
        stat["download_bytes"] = downloaded
        stat["download_secs"]  = time.monotonic() - t0
        dl_mb_s = (downloaded / 1e6) / stat["download_secs"] if stat["download_secs"] > 0 else 0
        print(f"\n{tag} download done ({downloaded/1e9:.2f} GB in "
              f"{stat['download_secs']:.1f}s = {dl_mb_s:.1f} MB/s), extracting...")

        # 2. Extract
        with _lock:
            _progress[name] = f"extracting... ({downloaded/1e9:.2f} GB)"
            _write_progress()

        t0 = time.monotonic()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        zip_path.unlink()
        stat["extract_secs"] = time.monotonic() - t0
        print(f"{tag} extraction done ({stat['extract_secs']:.1f}s)")

        # 3. Collect extracted files and upload in parallel
        files = [
            (p, f"{R2_PREFIX}/{split}/{p.relative_to(tmpdir)}" if DATASET == "main" else f"{R2_PREFIX}/{p.relative_to(tmpdir)}")
            for p in Path(tmpdir).rglob("*")
            if p.is_file()
        ]
        print(f"{tag} uploading {len(files)} files...")

        with _lock:
            _progress[name] = f"uploading {len(files)} files..."
            _write_progress()

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
            futures = {pool.submit(upload_one, lp, key): key for lp, key in files}
            for fut in as_completed(futures):
                fut.result()
        stat["upload_file_count"] = len(files)
        stat["upload_secs"]       = time.monotonic() - t0

    stat["total_secs"] = time.monotonic() - t_total_start
    with _lock:
        _pipeline_stats.append(stat)
        _progress[name] = f"done ({downloaded/1e9:.2f} GB, {stat['total_secs']:.0f}s)"
        _write_progress()

    print(f"{tag} done (total {stat['total_secs']:.1f}s)")


def _fmt_duration(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def write_summary(job_start: float, job_end: float):
    job_secs = job_end - job_start
    lines = []
    lines.append("=" * 60)
    lines.append("DSEC → R2 DOWNLOAD SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Dataset:          {DATASET}")
    lines.append(f"Splits:           {', '.join(SPLITS)}")
    lines.append(f"Modalities:       {', '.join(MODALITIES)}")
    lines.append(f"Pipeline workers: {PIPELINE_WORKERS}")
    lines.append(f"Upload workers:   {UPLOAD_WORKERS}")
    lines.append(f"Total wall time:  {_fmt_duration(job_secs)} ({job_secs:.1f}s)")
    lines.append("")

    total_bytes = 0
    total_files = 0

    lines.append("-" * 60)
    lines.append(f"{'Pipeline':<22} {'Data':>10} {'DL speed':>10} {'DL time':>10} {'Extract':>9} {'Upload files':>13} {'Total':>10}")
    lines.append("-" * 60)

    for s in sorted(_pipeline_stats, key=lambda x: x["pipeline"]):
        dl_gb   = s["download_bytes"] / 1e9
        dl_mb_s = (s["download_bytes"] / 1e6) / s["download_secs"] if s["download_secs"] > 0 else 0
        total_bytes += s["download_bytes"]
        total_files += s["upload_file_count"]
        lines.append(
            f"{s['pipeline']:<22} {dl_gb:>8.2f}GB "
            f"{dl_mb_s:>8.1f}MB/s {_fmt_duration(s['download_secs']):>10} "
            f"{_fmt_duration(s['extract_secs']):>9} "
            f"{s['upload_file_count']:>12}  {_fmt_duration(s['total_secs']):>9}"
        )

    lines.append("-" * 60)
    avg_dl_mb_s = (total_bytes / 1e6) / job_secs if job_secs > 0 else 0
    lines.append(f"{'TOTAL':<22} {total_bytes/1e9:>8.2f}GB {avg_dl_mb_s:>8.1f}MB/s {'':>10} {'':>9} {total_files:>12}")
    lines.append("=" * 60)

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    with SUMMARY_FILE.open("a") as f:
        f.write(f"\n# Run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(text)
    print(f"Summary appended to {SUMMARY_FILE}")


if __name__ == "__main__":
    pairs = [(s, m) for s in SPLITS for m in MODALITIES]
    print(f"Running {len(pairs)} pipelines with {PIPELINE_WORKERS} workers: {pairs}")
    print("Running preflight checks...")
    preflight_check()

    # Init progress file
    with _lock:
        for s, m in pairs:
            _progress[f"{s}/{m}"] = "pending"
        _write_progress()

    job_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=PIPELINE_WORKERS) as pool:
        futures = [pool.submit(pipeline, s, m) for s, m in pairs]
        for fut in as_completed(futures):
            fut.result()

    job_end = time.monotonic()

    write_summary(job_start, job_end)
    PROGRESS_FILE.write_text("ALL DONE. See download_summary.txt for details.\n")
    print("All done.")
