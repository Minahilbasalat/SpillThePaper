"""Read text from files a student uploads: text PDF, scanned PDF, Word, plain text, photo.

extract_text(filename, data) -> {"text", "method", "warning"}
Scanned PDFs and photos use the same OCR as the data pipeline (needs Tesseract installed)."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_WORDS_PER_PAGE = 25          # fewer words per page: treat a PDF as scanned
MAX_BYTES = 25 * 1024 * 1024


def _ocr_bytes(suffix: str, data: bytes) -> dict:
    try:
        from src.data.ocr import ocr_file
        import pytesseract
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError(f"OCR libraries are not installed ({e}). pip install pytesseract pillow") from e
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            return ocr_file(tmp)
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError("Tesseract OCR is not installed on this computer, so scanned files "
                               "and photos cannot be read. Install it or upload a text PDF/Word "
                               "file.") from e
    finally:
        os.unlink(tmp)


def extract_text(filename: str, data: bytes) -> dict:
    if len(data) > MAX_BYTES:
        raise ValueError("file is larger than 25 MB")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = len(doc)
            text = "\n".join(page.get_text() for page in doc)
        if pages and len(text.split()) / pages >= MIN_WORDS_PER_PAGE:
            return {"text": text, "method": "pdf_text", "warning": ""}
        r = _ocr_bytes(".pdf", data)
        return {"text": r["text"], "method": "ocr",
                "warning": "" if r["mean_conf"] >= 55 else
                f"scanned file read with low confidence ({r['mean_conf']}); check the questions"}
    if suffix == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append(" ".join(c.text for c in row.cells))
        return {"text": "\n".join(parts), "method": "docx", "warning": ""}
    if suffix in {".txt", ".md"}:
        return {"text": data.decode("utf-8", errors="replace"), "method": "text", "warning": ""}
    if suffix in IMAGE_SUFFIXES:
        r = _ocr_bytes(suffix, data)
        return {"text": r["text"], "method": "ocr",
                "warning": "" if r["mean_conf"] >= 55 else
                f"photo read with low confidence ({r['mean_conf']}); handwriting is not supported"}
    raise ValueError(f"unsupported file type '{suffix}' (use PDF, DOCX, TXT, JPG or PNG)")
