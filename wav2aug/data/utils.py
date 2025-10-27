import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("wav2aug.data")
log.addHandler(logging.NullHandler())


def _human_mb(n: int) -> str:
    return f"{n/1e6:6.1f}MB"


def _progress(done: int, total: int):
    width = max(10, min(40, shutil.get_terminal_size(fallback=(80, 20)).columns - 30))
    if total > 0:
        pct = done / total
        bar = "#" * int(pct * width) + "." * (width - int(pct * width))
        sys.stdout.write(
            f"\rProgress [{bar}] {pct*100:5.1f}%  {_human_mb(done)}/{_human_mb(total)}"
        )
    else:
        sys.stdout.write(f"\rProgress {_human_mb(done)}")
    sys.stdout.flush()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha256(path: str, expected: str | None) -> bool:
    if not expected:
        return True
    try:
        return sha256_file(path).lower() == expected.lower()
    except FileNotFoundError:
        return False


def download_url(
    url: str,
    dst: str,
    *,
    show_progress: bool = True,
    sha256: str | None = None,
    user_agent: str = "wav2aug/1.0",
):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Cached and verified
    if os.path.isfile(dst) and check_sha256(dst, sha256):
        log.info("Using cached file: %s", dst)
        return dst

    file_name = Path(urlparse(url).path).name
    sys.stdout.write(f"Downloading: {file_name}\n")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with (
        urllib.request.urlopen(req) as r,
        tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp,
    ):
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = time.monotonic()
        tty = show_progress and sys.stdout.isatty()
        if tty:
            _progress(0, total)
        while True:
            buf = r.read(1 << 20)  # 1 MiB
            if not buf:
                break
            tmp.write(buf)
            done += len(buf)
            if tty and time.monotonic() - last >= 0.05:
                _progress(done, total)
                last = time.monotonic()
        if tty:
            _progress(done, total)
            sys.stdout.write("\n")
            sys.stdout.flush()
        tmp_path = tmp.name

    if sha256 and not check_sha256(tmp_path, sha256):
        os.remove(tmp_path)
        raise RuntimeError("SHA256 mismatch for downloaded file")

    os.replace(tmp_path, dst)
    return dst


def safe_extract_targz(src_tgz: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(src_tgz, "r:gz") as tar:
        for m in tar.getmembers():
            p = os.path.join(dst_dir, m.name)
            if not os.path.realpath(p).startswith(os.path.realpath(dst_dir) + os.sep):
                raise RuntimeError(f"Blocked path traversal in tar member: {m.name}")
        tar.extractall(dst_dir)
