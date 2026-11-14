"""Step 4b - OCR: read the scanned PDFs and photographs that step 4 flagged "needs_ocr".

Reads : data/processed/extraction_results.csv   (rows with needs_ocr = True)
        data/raw/<file>                          (the downloaded files)
Writes: data/interim/text/<file_id>.txt          (accepted OCR text, same folder as normal text)
        data/interim/ocr_rejected/<file_id>.txt  (low-quality OCR, kept only for inspection)
        data/processed/ocr_results.csv           (one row per file: pages, words, confidence, status)
Updates: data/processed/extraction_results.csv   (accepted files: method "ocr", needs_ocr False)

A file is accepted only if OCR found at least MIN_WORDS words with a mean confidence of at
least MIN_CONF. Handwritten or very poor scans are rejected instead of polluting the data.
The script can be stopped and restarted: finished files are skipped.

Usage:
    python "data scripts/04b_ocr.py" --limit 20            # pilot run, check the output first
    python "data scripts/04b_ocr.py" --workers 6           # full run
    python "data scripts/04b_ocr.py" --merge-only          # only update extraction_results.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common import PROCESSED_DIR, RAW_DIR, ROOT, log

sys.path.insert(0, str(ROOT))                     # so that "src" can be imported

TEXT_DIR = ROOT / "data" / "interim" / "text"
REJECT_DIR = ROOT / "data" / "interim" / "ocr_rejected"
OCR_RESULTS = PROCESSED_DIR / "ocr_results.csv"
EXTRACTION = PROCESSED_DIR / "extraction_results.csv"
LOG = "extraction_log.txt"

MIN_WORDS = 30
MIN_CONF = 55.0


def _init_worker():
    os.environ["OMP_THREAD_LIMIT"] = "1"          # one Tesseract thread per worker process


def _work(file_id, local_file, psm):
    from src.data.ocr import ocr_file             # imported here: runs inside the worker
    try:
        r = ocr_file(RAW_DIR / local_file, psm=psm)
        return file_id, r, ""
    except Exception as error:                    # corrupt image, unsupported file, ...
        return file_id, None, f"{type(error).__name__}: {str(error)[:100]}"


def merge_into_extraction():
    """Mark accepted files as 'ocr' in extraction_results.csv."""
    ext = pd.read_csv(EXTRACTION, keep_default_na=False)
    ocr = pd.read_csv(OCR_RESULTS, keep_default_na=False)
    ok = ocr[ocr["status"] == "ok"].set_index("file_id")
    hit = ext["file_id"].isin(ok.index)
    ext.loc[hit, "extraction_method"] = "ocr"
    ext.loc[hit, "needs_ocr"] = False
    ext.loc[hit, "char_count"] = ext.loc[hit, "file_id"].map(ok["chars"]).astype(int)
    ext.loc[hit, "word_count"] = ext.loc[hit, "file_id"].map(ok["words"]).astype(int)
    ext.to_csv(EXTRACTION, index=False, encoding="utf-8")
    log(f"extraction_results.csv updated: {int(hit.sum())} files now have OCR text", LOG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=0, help="only this many files (pilot)")
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page mode (3 = automatic)")
    parser.add_argument("--redo", action="store_true", help="redo files already OCR'd")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge_into_extraction()
        return

    ext = pd.read_csv(EXTRACTION, keep_default_na=False)
    manifest = pd.read_csv(PROCESSED_DIR / "collection_manifest.csv", keep_default_na=False)
    local = dict(zip(manifest["file_id"], manifest["local_file"]))
    need = ext[(ext["needs_ocr"].astype(str) == "True") | (ext["extraction_method"] == "ocr")]

    done = pd.read_csv(OCR_RESULTS, keep_default_na=False) if OCR_RESULTS.exists() else pd.DataFrame()
    if args.redo:
        done = pd.DataFrame()
    skip = set(done["file_id"]) if len(done) else set()
    todo = [f for f in need["file_id"] if f not in skip and f in local]
    if args.limit:
        todo = todo[: args.limit]
    log(f"OCR: {len(need)} files flagged, {len(skip)} already done, {len(todo)} to do "
        f"({args.workers} workers)", LOG)

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_DIR.mkdir(parents=True, exist_ok=True)
    rows = done.to_dict("records") if len(done) else []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_work, f, local[f], args.psm) for f in todo]
        for n, fut in enumerate(as_completed(futures), start=1):
            file_id, r, error = fut.result()
            if r is None:
                rows.append({"file_id": file_id, "pages": 0, "words": 0, "chars": 0,
                             "mean_conf": 0, "status": "error", "error": error})
            else:
                good = r["words"] >= MIN_WORDS and r["mean_conf"] >= MIN_CONF
                folder = TEXT_DIR if good else REJECT_DIR
                (folder / f"{file_id}.txt").write_text(r["text"], encoding="utf-8")
                rows.append({"file_id": file_id, "pages": r["pages"], "words": r["words"],
                             "chars": len(r["text"].strip()), "mean_conf": r["mean_conf"],
                             "status": "ok" if good else "rejected_low_quality", "error": ""})
            if n % 25 == 0 or n == len(todo):
                pd.DataFrame(rows).to_csv(OCR_RESULTS, index=False, encoding="utf-8")
                log(f"  progress: {n}/{len(todo)}", LOG)

    res = pd.DataFrame(rows)
    res.to_csv(OCR_RESULTS, index=False, encoding="utf-8")
    log(f"OCR status: {res['status'].value_counts().to_dict()}", LOG)
    ok = res[res["status"] == "ok"]
    if len(ok):
        log(f"accepted files: median {ok['words'].median():.0f} words, "
            f"median confidence {ok['mean_conf'].astype(float).median():.1f}", LOG)
    merge_into_extraction()


if __name__ == "__main__":
    main()