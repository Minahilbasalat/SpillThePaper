"""Shared helpers used by all Phase 2 data scripts."""
import hashlib
from datetime import datetime
from pathlib import Path

import yaml

# Project folders (works no matter where the project is stored)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"


def load_yaml(file_name):
    """Read a YAML config file from the config/ folder."""
    with open(CONFIG_DIR / file_name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_file_id(repo_path):
    """Short, stable ID for a file, made from its path in the repository.
    We save downloads as <file_id>.<ext> to avoid Windows problems with long
    or strange file names. The manifest maps file_id back to the real path."""
    return hashlib.sha1(repo_path.encode("utf-8")).hexdigest()[:12]


def log(message, log_name="collection_log.txt"):
    """Print a message and also append it to logs/<log_name> (our evidence)."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / log_name, "a", encoding="utf-8") as f:
        f.write(line + "\n")