"""Shared paths — honour GOING_DB so Windows dev and Lucy prod use one knob."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_DB = "/mnt/nvme/going/db/going.db"


def repo() -> Path:
    return Path(os.getenv("GOING_REPO", str(REPO)))


def log_dir() -> Path:
    """Writable log directory — defaults to repo/logs (not NAS)."""
    raw = os.getenv("GOING_LOG_DIR", "").strip()
    if raw:
        return Path(raw)
    return repo() / "logs"


def db_path() -> Path:
    return Path(os.getenv("GOING_DB", DEFAULT_DB))


def db_exists() -> bool:
    return db_path().exists()
