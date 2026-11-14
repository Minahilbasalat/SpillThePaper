"""OCR for scanned PDFs and photographs, using Tesseract.

ocr_file(path) -> dict(text, mean_conf, words, pages)

Design notes (tested on synthetic scans; the pilot run on the real archive is the real test):
  * PDF pages are rendered at about 216 dpi with PyMuPDF, then OCR'd page by page.
  * Phone photos carry an EXIF rotation flag and are sometimes sideways: both are fixed
    before OCR (EXIF first, then Tesseract's orientation detection).
  * Text is rebuilt line by line (blank line between blocks) because the question
    segmenter in src/data/segment_questions.py relies on line starts ("Q1", "2.").
  * Tesseract's automatic page mode can return nothing on sparse pages; those pages are
    retried as one text block.
  * Tesseract reports a confidence per word; the mean confidence is returned so that the
    caller can reject unreadable (for example handwritten) files instead of polluting
    the question bank with garbage.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

import pytesseract

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

PDF_ZOOM = 3.0          # 72 dpi * 3 = 216 dpi
MAX_PAGES = 12
MIN_SIDE, MAX_SIDE = 1600, 3600
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPARSE_WORDS = 15       # below this many words a page gets a second attempt


def _prepare(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L")
    longest = max(img.size)
    if longest > MAX_SIDE or longest < MIN_SIDE:
        scale = (MAX_SIDE if longest > MAX_SIDE else MIN_SIDE) / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img, cutoff=1)


def _fix_rotation(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        if angle:
            img = img.rotate(-angle, expand=True, fillcolor=255)
    except Exception:          # too little text for orientation detection: keep as is
        pass
    return img


def _ocr_image(img: Image.Image, psm: int):
    d = pytesseract.image_to_data(img, lang="eng", config=f"--oem 1 --psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    lines, confs = {}, []
    for i, word in enumerate(d["text"]):
        word = word.strip()
        if not word:
            continue
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(word)
    out, last_block = [], None
    for (block, _par, _line), words in lines.items():    # insertion order = reading order
        if last_block is not None and block != last_block:
            out.append("")
        out.append(" ".join(words))
        last_block = block
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "\n".join(out), mean_conf, len(confs)


def _ocr_best(img: Image.Image, psm: int):
    best = _ocr_image(img, psm)
    if best[2] < SPARSE_WORDS and psm != 6:
        alt = _ocr_image(img, 6)
        if alt[2] * alt[1] > best[2] * best[1]:
            best = alt
    return best


def _page_images(path: Path):
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PAGES:
                    break
                pix = page.get_pixmap(matrix=pymupdf.Matrix(PDF_ZOOM, PDF_ZOOM),
                                      colorspace=pymupdf.csGRAY)
                yield Image.frombytes("L", (pix.width, pix.height), pix.samples)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as im:
            yield im.copy()
    else:
        raise ValueError(f"cannot OCR a {path.suffix} file")


def ocr_file(path, psm: int = 3) -> dict:
    texts, confs, words, pages = [], [], 0, 0
    for img in _page_images(Path(path)):
        img = _fix_rotation(_prepare(img))
        text, conf, n = _ocr_best(img, psm)
        pages += 1
        if n:
            texts.append(text)
            confs.append((conf, n))
            words += n
    total = sum(n for _, n in confs)
    mean_conf = sum(c * n for c, n in confs) / total if total else 0.0
    return {"text": "\n\n".join(texts), "mean_conf": round(mean_conf, 1),
            "words": words, "pages": pages}