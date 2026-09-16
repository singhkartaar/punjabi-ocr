"""
A merged OCR book as pages of paragraphs -- the shape lib/writings_pdf.read_pdf returns.

12_ingest_writings.py reads an author's PDFs into paragraph records without
knowing whether the paragraphs came from a text layer or from OCR: both
readers return {"path", "body_size", "pages": [{"page", "marker", "spread",
"columns", "paragraphs": [{"text", "style", "italic", "x0", "size"}]}]}. This
one reads data/ocr/<book>/merged/NNNN.jsonl (22_ocr_merge.py) and adds, on a
quoted paragraph, the line_ids the merge resolved, so that
13_resolve_citations.py can take them as given instead of guessing from an
English translation.

  marker    the printed page number from the running header ("੨੩" -> "23"),
            playing the part "L127.3" plays for the essays
  quote     a paragraph whose lines are Gurbani (matched or bold with verse
            marks); its text is the corpus's where matched
  footnote  the lines under the rule, one paragraph per numbered note
"""
from __future__ import annotations
import glob
import json
import os
import statistics

from lib.ocr_engines import read_page
from lib.ocr_layout import page_paragraphs, quote_meta
from lib.ocr_zones import FOOTNOTE_START

CAN_QUOTE = ("gurbani", "gurbani-unmatched")


def _footnote_paragraphs(lines: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ln in lines:
        text = ln.get("text", "").strip()
        if not text:
            continue
        if out and not FOOTNOTE_START.match(text):
            out[-1]["text"] += " " + text
            out[-1]["lines"].append(ln)
        else:
            out.append({"text": text, "style": "footnote", "italic": False, "x0": ln["bbox"][0],
                        "size": ln["bbox"][3] - ln["bbox"][1], "lines": [ln]})
    return out


def read_ocr_book(book_dir: str) -> dict:
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    pages = []
    sizes: list[float] = []
    for path in sorted(glob.glob(os.path.join(book_dir, "merged", "*.jsonl"))):
        pmeta, lines = read_page(path)
        body = [ln for ln in lines if ln.get("zone") == "body"]
        foot = [ln for ln in lines if ln.get("zone") == "footnote"]
        paras = page_paragraphs(body, pmeta["page_h"], pmeta.get("columns") or None)
        out = []
        for p in paras:
            kinds = [ln.get("kind") for ln in p.get("lines", [])]
            quote = bool(kinds) and all(k in CAN_QUOTE for k in kinds)
            style = "quote" if quote else p["style"]
            rec = {"text": p["text"], "style": style, "italic": quote, "x0": p["x0"], "size": p["size"],
                   "kinds": kinds}
            if quote:
                rec.update(quote_meta(p))
            if any(ln.get("route") for ln in p.get("lines", [])):
                rec["routed"] = True
            agreements = [ln["agreement"] for ln in p.get("lines", []) if ln.get("agreement") is not None]
            if agreements:
                rec["agreement"] = round(min(agreements), 3)
            out.append(rec)
            sizes.append(p["size"])
        for fp in _footnote_paragraphs(foot):
            out.append({k: v for k, v in fp.items() if k != "lines"})
        hints = pmeta.get("hints") or {}
        marker = str(hints["book_page"]) if hints.get("book_page") else None
        pages.append({"page": pmeta["page"], "marker": marker, "spread": False,
                      "columns": len(pmeta.get("columns") or []), "paragraphs": out,
                      "ang_from": hints.get("ang_from"), "ang_to": hints.get("ang_to")})
    return {"path": meta.get("pdf"), "body_size": statistics.median(sizes) if sizes else 10.0, "pages": pages}
