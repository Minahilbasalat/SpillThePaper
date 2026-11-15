"""Download and unpack the runtime bundle (trained models, topic models, question bank).

    python scripts/get_runtime.py <URL-of-runtime_bundle.zip>
    python scripts/get_runtime.py                 # uses config/runtime_url.txt or $SPILL_RUNTIME_URL

Best home for the zip: a GitHub Release of this repository (drag the file in; the asset
link works here directly). A Google Drive link also works if `pip install gdown`.
Why it is not in Git: the trained weights are hosted externally (course rule) and the question
bank is derived from exam papers that we do not redistribute inside the repository."""
import io
import os
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    url = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SPILL_RUNTIME_URL", "")).strip()
    cfg = ROOT / "config" / "runtime_url.txt"
    if not url and cfg.exists():
        url = cfg.read_text(encoding="utf-8").strip()
    if not url:
        sys.exit("Give the URL of runtime_bundle.zip (argument, $SPILL_RUNTIME_URL or config/runtime_url.txt).")
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            sys.exit("Google Drive link: run `pip install gdown` first.")
        out = ROOT / "runtime_bundle.zip"
        gdown.download(url, str(out), quiet=False, fuzzy=True)
        data = out.read_bytes()
    else:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        data = r.content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(ROOT)
        print(f"unpacked {len(z.namelist())} files into {ROOT}")
    print("Check it with:  python -m pytest tests -q")


if __name__ == "__main__":
    main()
