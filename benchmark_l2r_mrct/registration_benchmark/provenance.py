"""Small provenance helpers used by run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    return _sha256_file_version(path, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=256)
def _sha256_file_version(path: Path, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(repo: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    def command(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {
        "path": str(repo),
        "commit": command("rev-parse", "HEAD"),
        "remote": command("remote", "get-url", "origin"),
        "dirty": bool(command("status", "--porcelain")),
    }


def environment_identity() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        result["torch"] = None
    return result
