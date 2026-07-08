#!/usr/bin/env python3
"""
Compress and archive log files older than 24 hours.
Skips files still being written to (modified within the last 60s).
"""
import os
import glob
import gzip
import time
import logging

MONITOR_DIR = "/caches/logs/"
ARCHIVE_DIR = os.path.join(MONITOR_DIR, "archive")
LOG_FILE = os.path.join(MONITOR_DIR, "rotator.log")

AGE_THRESHOLD_SEC = 24 * 60 * 60
WRITE_GRACE_SEC = 60

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def skip_reason(path):
    """Return a reason string to skip this file, or None if it's eligible."""
    age = time.time() - os.path.getmtime(path)
    if age < WRITE_GRACE_SEC:
        return f"modified within last {WRITE_GRACE_SEC}s (likely still being written)"
    if age < AGE_THRESHOLD_SEC:
        return f"not yet {AGE_THRESHOLD_SEC // 3600}h old"
    return None


def compress_and_delete(path):
    filename = os.path.basename(path)
    output_path = os.path.join(ARCHIVE_DIR, filename + ".gz")
    try:
        with open(path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(path)
        logging.info("Compressed and removed %s -> %s", path, output_path)
    except OSError as e:
        logging.error("Failed to process %s: %s", path, e)
        if os.path.exists(output_path):
            os.remove(output_path)  # don't leave a partial/corrupt archive behind


def find_and_process_logs():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(MONITOR_DIR, "*.log")):
        reason = skip_reason(path)
        if reason:
            logging.info("Skipping %s: %s", path, reason)
        else:
            compress_and_delete(path)


if __name__ == "__main__":
    find_and_process_logs()