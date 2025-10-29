from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from .packs import DEFAULT_PACK, REGISTRY

log = logging.getLogger("wav2aug.data")

DATA_ROOT = os.environ.get(
    "WAV2AUG_DATA_DIR", os.path.join(pathlib.Path.home(), ".cache", "wav2aug")
)


def _pack_dir(spec) -> str:
    return os.path.join(DATA_ROOT, "noise", spec.name, spec.version)


def _ready_marker(root: str) -> str:
    return os.path.join(root, ".ready")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_tar_gz(tgz_path: str, dest_dir: str) -> None:
    log.info("extracting archive -> %s", dest_dir)
    with tarfile.open(tgz_path, "r:gz") as tar:
        for m in tar.getmembers():
            p = os.path.join(dest_dir, m.name)
            if not os.path.realpath(p).startswith(os.path.realpath(dest_dir) + os.sep):
                raise RuntimeError(f"Blocked path traversal in tar member: {m.name}")
        tar.extractall(dest_dir)
    log.info("extraction complete")


def _download(url: str, out_path: str) -> None:
    """Download with simple progress bar to stderr."""
    show = os.environ.get("WAV2AUG_PROGRESS", "1") != "0"

    def _progress(done: int, total: int):
        w = max(10, min(40, shutil.get_terminal_size(fallback=(80, 20)).columns - 30))
        if total > 0:
            pct = done / total
            fill = int(pct * w)
            bar = "#" * fill + "." * (w - fill)
            sys.stderr.write(
                f"\rwav2aug - Progress [{bar}] {pct*100:5.1f}%  {done/1e6:6.1f}MB/{total/1e6:6.1f}MB"
            )
        else:
            sys.stderr.write(f"\rwav2aug - Progress {done/1e6:6.1f}MB")
        sys.stderr.flush()

    name = Path(urlparse(url).path).name or "download"

    sys.stderr.write(f"wav2aug - Downloading: {name}\n")
    sys.stderr.flush()

    req = urllib.request.Request(url, headers={"User-Agent": "wav2aug/1.0"})
    start = time.monotonic()
    with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        chunk = 1 << 20
        done = 0
        last = time.monotonic()
        tty = show and sys.stderr.isatty()
        if tty:
            _progress(0, total)

        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if tty and (time.monotonic() - last) >= 0.05:
                _progress(done, total)
                last = time.monotonic()

        if tty:
            _progress(done, total)
            sys.stderr.write("\n")
            sys.stderr.flush()

    elapsed = max(1e-6, time.monotonic() - start)
    log.info(
        "downloaded %s (%.1f MB) in %.1fs (%.1f MB/s)",
        name,
        done / 1e6,
        elapsed,
        (done / 1e6) / elapsed,
    )


def ensure_pack(name: str | None = None) -> str:
    """Download and cache noise pack if not already available.

    Verifies pack exists in local cache, downloading and extracting
    if missing. Includes SHA256 verification for integrity.
    Uses file locking to prevent race conditions in multiprocessing.

    Args:
        name: Noise pack identifier. Uses default pack if None.

    Returns:
        Path to directory containing noise audio files.

    Raises:
        RuntimeError: If SHA256 verification fails after download.
    """
    spec = REGISTRY.get(name or DEFAULT_PACK.name, DEFAULT_PACK)
    root = _pack_dir(spec)
    marker = _ready_marker(root)

    # if already downloaded, return immediately
    if os.path.isfile(marker):
        return root

    # file locking to prevent race conditions
    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".download.lock")

    if HAS_FCNTL:
        # unix file locking
        with open(lock_path, "w") as lock_file:
            try:
                # acquire exclusive lock (blocks until available)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                # check again after acquiring lock, another process might have completed download
                if os.path.isfile(marker):
                    return root

                _perform_download(spec, root, marker)

            finally:
                # lock is automatically released when file is closed
                pass

        # clean up lock file
        try:
            os.unlink(lock_path)
        except OSError:
            pass  # ignore if already removed
    else:
        # fallback for windows
        lock_dir = os.path.join(root, ".download.lock.dir")
        max_wait = 300  # 5 mins
        wait_time = 0

        while wait_time < max_wait:
            try:
                os.makedirs(lock_dir, exist_ok=False)
                # we got the lock
                try:
                    # check again after acquiring lock
                    if os.path.isfile(marker):
                        return root
                    _perform_download(spec, root, marker)
                finally:
                    # release lock
                    try:
                        os.rmdir(lock_dir)
                    except OSError:
                        pass
                break
            except FileExistsError:
                # another process has the lock, wait a bit
                time.sleep(0.1)
                wait_time += 0.1
                # check if download completed while we were waiting
                if os.path.isfile(marker):
                    return root
        else:
            # timeout waiting for lock, try to proceed anyway
            log.warning("Timeout waiting for download lock, proceeding anyway")
            if not os.path.isfile(marker):
                _perform_download(spec, root, marker)

    return root


def _perform_download(spec, root: str, marker: str) -> None:
    """Perform the actual download and extraction."""
    log.info("downloading pack %s to %s", spec.name, root)
    with tempfile.TemporaryDirectory(dir=root) as tmpd:
        tgz = os.path.join(tmpd, "pack.tgz")
        _download(spec.url, tgz)
        if spec.sha256:
            digest = _sha256_file(tgz)
            if digest.lower() != spec.sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {spec.url}")
        _safe_extract_tar_gz(tgz, root)

    manifest = {
        "name": spec.name,
        "version": spec.version,
        "source_url": spec.url,
        "sha256": spec.sha256,
    }
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # mark as complete, this will release other waiting processes
    pathlib.Path(marker).touch()
    log.info("pack %s download complete", spec.name)
