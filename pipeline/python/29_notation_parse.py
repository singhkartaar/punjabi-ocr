"""
Notation books: merged OCR pages -> notation records, with their crops.

  29_notation_parse.py --book gurmat-sangeet-sagar-1                 # every merged page
  29_notation_parse.py --book gurmat-sangeet-sagar-1 --pages 165-176
  29_notation_parse.py --book B --no-grid                             # layout, shabad and crops only

Runs after 22_ocr_merge.py on a book whose manifest says kind: "notation".
Per page: the layout (lib/notation_layout.page_layout) -- heading, shabad
text, reference, section labels, marker rows, grids -- cached under
data/ocr/<book>/notation/pages/NNNN.json so a rerun after a fix in a later
stage does not redo it. Then the pages are strung into notation spans
(link_pages), each span's shabad is resolved against the corpus
(lib/notation_resolve), its grids are read (lib/notation_grid; skipped with
--no-grid, which leaves kind "partial"), its regions are cropped as 1-bit
PNGs, and one record per span goes to data/notations/<book>/notations.jsonl
with images.json beside it and a report in data/raw/.

The scan is the authority: every record carries the crops of its grid,
its shabad text and its heading, and the reviewer (30_notation_gt.py)
judges the record against those.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import notation
from lib.notation import PARSER_VERSION, SCHEMA_VERSION, image_name, make_id, merge_style, notation_slug, validate
from lib.notation_grid import read_region
from lib.notation_layout import link_pages, page_layout, raag_descriptions
from lib.notation_resolve import resolve_shabad
from lib.notation_text import parse_heading
from lib import notation_vocab
from lib.notation_vocab import VERSION as VOCAB_VERSION, raag_key_from_corpus
from lib.ocr_pages import parse_pages
from lib.paths import ARTIFACTS, CORPUS_DB, NOTATIONS_DIR, OCR_DIR, ROOT
CROP_PAD = 24


def load_jsonl(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[0]["_meta"], rows[1:]


def corpus_connection() -> sqlite3.Connection | None:
    """corpus.sqlite if it is there, else the shipped gurbani.sqlite: both hold lines and shabads."""
    for path in (CORPUS_DB, os.path.join(ARTIFACTS, "gurbani.sqlite")):
        if path and os.path.exists(path):
            return sqlite3.connect(path)
    return None


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layouts_for(book_dir: str, pages: list[dict], style: dict, force: bool) -> list[dict]:
    """The layout of every page, from the cache or freshly computed (and cached)."""
    import cv2
    from lib.ocr_grid import binarise
    cache_dir = os.path.join(book_dir, "notation", "pages")
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for rec in pages:
        page = rec["page"]
        cache = os.path.join(cache_dir, "%04d.json" % page)
        merged_path = os.path.join(book_dir, "merged", "%04d.jsonl" % page)
        if not os.path.exists(merged_path):
            continue
        if not force and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(merged_path):
            with open(cache, encoding="utf-8") as fh:
                lay = json.load(fh)
                if lay.get("version") == PARSER_VERSION:
                    out.append(lay)
                    continue
        meta, lines = load_jsonl(merged_path)
        img = cv2.imread(os.path.join(book_dir, "pages", rec["file"]), cv2.IMREAD_GRAYSCALE)
        ink = binarise(img) if img is not None else None
        lay = page_layout(lines, meta.get("page_w") or rec["w"], meta.get("page_h") or rec["h"], style, page,
                          ink=ink, body_h=meta.get("body_h"))
        lay["version"] = PARSER_VERSION
        lay["page_w"], lay["page_h"] = meta.get("page_w") or rec["w"], meta.get("page_h") or rec["h"]
        # the merged lines of the shabad blocks travel with the layout: the resolver needs their matches
        with open(cache, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(lay, fh, ensure_ascii=False)
        out.append(lay)
    return out


_AT_NUMBER = re.compile("(?:ਸ਼ਬਦ|ਸਬਦ)\\s*ਨੰ[:ਃ.]?\\s*([੦-੯0-9]+)")


def shabad_at_number(notes: list[str]) -> str | None:
    """'ਇਹ ਸ਼ਬਦ ਨੰ: ੩ ਤੇ ਲਿਖਿਆ ਹੈ': the shabad text is printed under notation
    number 3 of the same book. The number, as printed."""
    for text in notes:
        m = _AT_NUMBER.search(text or "")
        if m:
            return m.group(1)
    return None


def borrow_shabads(records: list[dict]) -> int:
    """
    A record with no shabad text that points at another notation's number
    takes that notation's shabad (method "book-ref"), one confidence step
    down. Returns how many were filled.
    """
    from lib.notation_vocab import gurmukhi_digits
    by_number = {}
    for rec in records:
        num = (rec.get("heading") or {}).get("number")
        if num is not None and rec["shabad"].get("shabad_id") is not None:
            by_number.setdefault(gurmukhi_digits(str(num)), rec)
    filled = 0
    for rec in records:
        at = rec["layout"].get("shabad_at_number")
        if not at or rec["shabad"].get("shabad_id") is not None:
            continue
        src = by_number.get(gurmukhi_digits(at))
        if not src:
            continue
        sh = dict(src["shabad"])
        sh["method"] = "book-ref"
        sh["confidence"] = round(min(sh.get("confidence") or 0.0, 0.9) - 0.1, 3)
        rec["shabad"] = sh
        rec["raag_shabad"] = src.get("raag_shabad")
        used = (rec.get("heading") or {}).get("raag") or {}
        rec["raag_differs"] = bool(rec["raag_shabad"] and used.get("key") and used.get("key") != rec["raag_shabad"]
                                   and used.get("parent_key") != rec["raag_shabad"])
        flags = set(rec["flags"]) - {"unresolved-shabad", "weak-shabad"}
        flags.add("shabad-by-book-ref")
        if rec["raag_differs"]:
            flags.add("raag-differs")
        rec["flags"] = sorted(flags)
        if rec["kind"] == "non-gurbani":
            rec["kind"] = "partial"
        rec["source"]["content_hash"] = notation.content_hash(rec)
        filled += 1
    return filled


def raag_records(descriptions: list[dict], book: str, book_dir: str, images_dir: str, page_files: dict[int, str]) -> list[dict]:
    """
    What a book says about a raag, cut from the page: one record a description,
    with the crop of every page it runs over and the OCR text as it came.
    """
    import cv2
    from lib.ocr_grid import crop_bilevel
    out = []
    for n, d in enumerate(descriptions, 1):
        images = []
        text_parts = [d["heading"]["text"]]
        for k, p in enumerate(d["pages"], 1):
            boxes = [r["bbox"] for pg, r in d["regions"] if pg == p]
            if d["page"] == p:
                boxes.append(d["heading"]["bbox"])
            if not boxes:
                continue
            img = cv2.imread(os.path.join(book_dir, "pages", page_files[p]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ph, pw = img.shape[:2]
            margin = int(pw * 0.03)
            bbox = [margin, min(b[1] for b in boxes), pw - margin, max(b[3] for b in boxes)]
            fname = "%s-raag-%04d-%d-%d.png" % (book, d["page"], n, k)
            dst = os.path.join(images_dir, fname)
            w, h = crop_bilevel(img, bbox, CROP_PAD, dst)
            images.append({"n": k, "file": "images/" + fname, "role": "block", "page": p, "bbox": [int(v) for v in bbox],
                           "w": w, "h": h, "bytes": os.path.getsize(dst), "sha256": sha256_of(dst)})
        text_parts += [r.get("text") or "" for _, r in d["regions"]]
        raag = d["raag"]
        out.append({"n": n, "book_key": book, "raag": {"printed": raag.get("printed"), "key": raag.get("key"),
                                                        "parent_key": raag.get("parent_key"), "confidence": raag.get("confidence")},
                    "heading": d["heading"]["text"], "page": d["page"], "pages": d["pages"],
                    "text": "\n".join(t for t in text_parts if t), "image": images[0] if images else None, "images": images})
    return out


_INDEX = {}


def corpus_index(con):
    """The corpus matcher's index, built once per run."""
    if con is None:
        return None
    if "index" not in _INDEX:
        from lib.ocr_match import CorpusIndex
        _INDEX["index"] = CorpusIndex.from_sqlite(con)
    return _INDEX["index"]


def bol_witness(sections: list[dict], con, ref: dict | None) -> dict | None:
    """
    The bol rows of the sthai (or the first section read), joined and matched
    against the corpus: {"shabad_id", "score", "line_id"} of the best line
    found, or None. A printed reference narrows the search to its angs.
    """
    from lib.notation_text import bol_text
    from lib.ocr_match import match_text
    index = corpus_index(con)
    if index is None or not sections:
        return None
    best = None
    hints = {"ang_from": ref["ang_from"], "ang_to": ref.get("ang_to") or ref["ang_from"]} if ref and ref.get("ang_from") else None
    for sec in sections[:2]:
        text = " ".join(bol_text(line.get("beats") or []) for line in sec.get("lines") or [])
        text = " ".join(w for w in text.split() if w)
        if len(text.replace(" ", "")) < 8:
            continue
        try:
            hit = match_text(text, index, hints) or match_text(text, index, None)
        except Exception:
            hit = None
        if hit and (best is None or hit["score"] > best["score"]):
            best = {"shabad_id": hit["shabad_id"], "score": float(hit["score"]), "line_id": hit.get("line_id")}
    return best


def matra_check(sections: list[dict], taal_key: str | None) -> str:
    """ok when every avartan row of the grid closes on the taal's last matra (or holds whole avartans); n/a without a taal."""
    info = notation_vocab.taal_info(taal_key)
    if not info or not info.get("matras") or not sections:
        return "n/a"
    if info.get("uncertain"):
        return "n/a"
    matras = int(info["matras"])
    rows = [line for sec in sections for line in sec.get("lines", []) if line.get("kind") == "avartan"]
    if not rows:
        return "n/a"
    bad = 0
    for line in rows:
        last = (line["beats"][-1].get("m") if line["beats"] else None)
        if last is not None and last != matras:
            bad += 1
    return "ok" if bad == 0 else "mismatch"


def span_record(span: dict, book: dict, style: dict, con, seq: int, book_dir: str, images_dir: str,
                page_files: dict[int, str], no_grid: bool, engine=None, body_h_of: dict | None = None) -> tuple[dict, list[dict]]:
    """One notation record from a linked span, plus its image entries."""
    import cv2
    from lib.ocr_grid import binarise, crop_bilevel, thumbnail
    cache: dict[int, object] = {}
    inks: dict[int, object] = {}

    def page_image(p):
        if p not in cache:
            cache[p] = cv2.imread(os.path.join(book_dir, "pages", page_files[p]), cv2.IMREAD_GRAYSCALE)
        return cache[p]

    def page_ink(p):
        if p not in inks:
            img = page_image(p)
            inks[p] = binarise(img) if img is not None else None
        return inks[p]
    first_page = span["pages"][0]
    # the span's first grid page names it; a span with no grid, its heading's page
    grid_pages = [p for sec in span["sections"] for p, _ in sec["grids"]]
    page = grid_pages[0] if grid_pages else (span["heading_page"] or first_page)
    nid = make_id(book["key"], page, seq)
    heading = None
    if span["heading"]:
        h = span["heading"]["parsed"]
        heading = {"raw": span["heading"]["text"], "number": h.get("number"),
                   "raag": h.get("raag"), "taal": h.get("taal"), "reet_of": h.get("reet_of")}
    shabad_lines = [m for _, r in span["shabad"] for m in (r.get("merged") or [])]
    ref = (span["ref"] or {}).get("parsed") if span["ref"] else None
    if ref is None:
        # the reference may sit on the last line of the shabad block
        from lib.notation_text import parse_ref
        for line in reversed(shabad_lines):
            got = parse_ref(line.get("text") or "")
            if got:
                ref = got
                break
    res = resolve_shabad(shabad_lines, ref, con)
    flags = list(res["flags"])
    kind = "notation"
    if not span["shabad"]:
        flags.append("no-shabad-text")
    if heading and heading.get("reet_of") and not grid_pages:
        kind = "reet-ref"
    elif res["kind_hint"] == "non-gurbani" and span["shabad"]:
        kind = "non-gurbani"
    # the grid reader: every grid of every section, its lines into the record
    sections_out: list[dict] = []
    quality = {"cells": 0, "unknown": 0, "conf_sum": 0.0, "conf_n": 0, "marks": {}}
    taal_key = ((heading or {}).get("taal") or {}).get("key") if heading else None
    extents: dict[tuple[int, int], list[int]] = {}
    counts_by_kind: dict[str, int] = {}
    if not no_grid and grid_pages:
        for si, sec in enumerate(span["sections"]):
            sec_taal = ((sec.get("taal") or {}).get("key")) or taal_key
            lines: list[dict] = []
            for p, r in sec["grids"]:
                ink = page_ink(p)
                if ink is None:
                    continue
                # a marker row printed against the grid's top edge is read with the grid, whole
                bbox = list(r["bbox"])
                for mp, mr in sec["markers"]:
                    if mp == p and mr["bbox"][3] >= bbox[1] - 40 and mr["bbox"][1] <= bbox[1]:
                        bbox = [min(bbox[0], mr["bbox"][0]), min(bbox[1], mr["bbox"][1]), max(bbox[2], mr["bbox"][2]), bbox[3]]
                got = read_region(ink, page_image(p), bbox, style, sec_taal, engine, p,
                                  body_h=(body_h_of or {}).get(p), script=style.get("script", "gurmukhi"))
                extents[(si, p)] = got["bbox"]
                lines.extend(got["lines"])
                for f in got["flags"]:
                    if f not in flags:
                        flags.append(f)
                q = got["quality"]
                quality["cells"] += q["cells"]
                quality["unknown"] += q["unknown"]
                if q["cells"]:
                    quality["conf_sum"] += q["conf_mean"] * q["cells"]
                    quality["conf_n"] += q["cells"]
                for k, v in q["marks"].items():
                    quality["marks"][k] = quality["marks"].get(k, 0) + v
                if got.get("taal_inferred") and heading is not None and not sec_taal:
                    if not heading.get("taal"):
                        heading["taal"] = {}
                    if not heading["taal"].get("key"):
                        heading["taal"].update({"key": got["taal_inferred"], "confidence": 0.5,
                                                "matras": (notation_vocab.taal_info(got["taal_inferred"]) or {}).get("matras")})
                        taal_key = got["taal_inferred"]
            if not lines:
                continue
            skind = sec.get("kind") or "other"
            counts_by_kind[skind] = counts_by_kind.get(skind, 0) + 1
            out_sec = {"kind": skind, "n": sec.get("n") or counts_by_kind[skind], "lines": lines}
            if sec.get("label") and sec["label"].get("text"):
                out_sec["label"] = sec["label"]["text"]
            if sec.get("taal") and (sec["taal"] or {}).get("key"):
                out_sec["taal"] = sec["taal"]["key"]
            sections_out.append(out_sec)
    if no_grid or not grid_pages or not sections_out:
        if kind == "notation":
            kind = "partial"
        flags.append("partial-grid")
    # the sung syllables under the swaras are the shabad's own words: a third witness,
    # and the only one when the printed text was not read (or not printed)
    bol = bol_witness(sections_out, con, ref)
    if bol:
        res2 = resolve_shabad(shabad_lines, ref, con, bol_match=bol)
        if res2["shabad"].get("shabad_id") is not None and (res["shabad"].get("shabad_id") is None
                                                            or res2["shabad"]["confidence"] > res["shabad"]["confidence"]):
            if res["shabad"].get("shabad_id") is None:
                flags = [f for f in flags if f not in ("unresolved-shabad", "weak-shabad")] + ["shabad-by-bol"]
                if kind == "non-gurbani":
                    kind = "notation"
            res = res2
            for f in res2["flags"]:
                if f not in flags:
                    flags.append(f)
    if span["continued"]:
        flags.append("continued-from-prev")
    if span.get("inherited"):
        flags.append("shabad-inherited")
    if span.get("capped"):
        flags.append("span-capped")
    if any(sec.get("assumed") for sec in span["sections"]):
        flags.append("no-section-label")
    if heading and heading.get("raag") and heading["raag"].get("key") is None:
        flags.append("raag-unknown")
    if heading and heading.get("taal") and heading["taal"].get("key") is None:
        flags.append("taal-unknown")
    raag_shabad = res["shabad"].get("raag_key")
    raag_used = (heading or {}).get("raag") or {}
    raag_differs = bool(raag_shabad and raag_used.get("key") and raag_used.get("key") != raag_shabad
                        and raag_used.get("parent_key") != raag_shabad)
    if raag_differs:
        flags.append("raag-differs")

    # crops: first the notation as printed on each of its pages -- every region
    # of the span on that page, and whatever the book set between them (a
    # taan, a tihai, a note) -- then each section's grids, the shabad text, the heading
    images = []
    n = 0
    targets: list[tuple[str, int, list[int]]] = []
    for p in sorted(set(span["pages"])):
        # everything the linker put in the span on this page -- the shabad, the
        # heading, the grids, a taan, a tihai, a note, a notation set as text --
        # and the grid extents the reader measured from the ink
        boxes = [span["extent"][p]] if span.get("extent", {}).get(p) else []
        boxes += [r["bbox"] for pg, r in span["shabad"] if pg == p]
        boxes += [r["bbox"] for sec in span["sections"] for pg, r in sec["grids"] + sec["markers"] if pg == p]
        boxes += [ext for (si_, pg), ext in extents.items() if pg == p]
        if not boxes:
            continue
        img = page_image(p)
        if img is None:
            continue
        ph, pw = img.shape[:2]
        margin = int(pw * 0.03)
        targets.append(("block", p, [margin, min(b[1] for b in boxes), pw - margin, max(b[3] for b in boxes)]))
    for si, sec in enumerate(span["sections"]):
        by_page: dict[int, list[list[int]]] = {}
        for p, r in sec["grids"]:
            by_page.setdefault(p, []).append(extents.get((si, p), r["bbox"]))
        for p, r in sec["markers"]:
            by_page.setdefault(p, []).append(r["bbox"])
        if sec.get("label"):
            by_page.setdefault(sec["page"], []).append(sec["label"]["bbox"])
        for p, boxes in by_page.items():
            targets.append(("grid", p, [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                        max(b[2] for b in boxes), max(b[3] for b in boxes)]))
    for p, r in span["shabad"]:
        targets.append(("shabad", p, r["bbox"]))
    if span["heading"]:
        targets.append(("heading", span["heading_page"], span["heading"]["bbox"]))
    for role, p, bbox in targets:
        img = page_image(p)
        if img is None:
            continue
        n += 1
        fname = image_name(nid, n)
        dst = os.path.join(images_dir, fname)
        w, h = crop_bilevel(img, bbox, CROP_PAD, dst)
        entry = {"n": n, "file": "images/" + fname, "role": role, "page": p, "bbox": [int(v) for v in bbox],
                 "w": w, "h": h, "bytes": os.path.getsize(dst), "sha256": sha256_of(dst), "thumb": None}
        if role == "block" and n == 1:
            tname = image_name(nid, n, thumb=True)
            thumbnail(img, [max(0, bbox[0] - CROP_PAD), max(0, bbox[1] - CROP_PAD), bbox[2] + CROP_PAD, bbox[3] + CROP_PAD],
                      os.path.join(images_dir, tname))
            entry["thumb"] = "images/" + tname
        images.append(entry)

    rec = {
        "schema_version": SCHEMA_VERSION, "notation_id": nid, "book_key": book["key"], "page": page, "seq": seq,
        "pages": sorted(set(span["pages"])), "kind": kind, "script": style.get("script", "gurmukhi"),
        "heading": heading,
        "shabad": {k: v for k, v in res["shabad"].items() if k not in ("raag", "raag_key")},
        "raag_shabad": raag_shabad, "raag_differs": raag_differs,
        "sections": sections_out,
        "layout": {"sections": [{"kind": s.get("kind"), "n": s.get("n"), "page": s["page"],
                                 "label": (s["label"] or {}).get("text"),
                                 "taal": s.get("taal"),
                                 "grids": [{"page": p, "bbox": r["bbox"], "lines": r["lines"]} for p, r in s["grids"]],
                                 "markers": [{"page": p, "bbox": r["bbox"], "marks": r["parsed"]["marks"]} for p, r in s["markers"]]}
                                for s in span["sections"]],
                   "notes": [r["text"] for _, r in span["notes"]],
                   "text": [r["text"] for _, r in span.get("text", [])],
                   "extent": {str(p): b for p, b in (span.get("extent") or {}).items()},
                   "shabad_at_number": shabad_at_number([r["text"] for _, r in span["notes"]])},
        "images": images,
        "source": {"book_key": book["key"], "author": book["author"], "author_key": book["author_key"],
                   "title": book["title"], "title_en": book.get("title_en"), "part": book.get("part"),
                   "pages": sorted(set(span["pages"])), "engines": book.get("engines", []),
                   "row_engine": getattr(engine, "key", None) if sections_out else None,
                   "parser": {"name": "notation", "version": PARSER_VERSION},
                   "vocab_version": VOCAB_VERSION, "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "content_hash": "", "gold": None},
        "quality": {"cells": quality["cells"], "unknown": quality["unknown"],
                    "conf_mean": round(quality["conf_sum"] / quality["conf_n"], 3) if quality["conf_n"] else 0.0,
                    "matra_check": matra_check(sections_out, taal_key), "marks": quality["marks"], "watermark": 0.0},
        "flags": sorted(set(flags)), "verified": False,
    }
    rec["source"]["content_hash"] = notation.content_hash(rec)
    rec["shabad"]["raag"] = res["shabad"].get("raag")
    return rec, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="the book key under data/ocr/")
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory")
    ap.add_argument("--notations-dir", default=NOTATIONS_DIR)
    ap.add_argument("--pages", help="'165-176,200' -- only these rendered pages")
    ap.add_argument("--no-grid", action="store_true", help="layout, shabad and crops; leave the grids unread")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cached page layouts")
    args = ap.parse_args()

    book_dir = os.path.join(args.out, args.book)
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        pages_meta = json.load(fh)
    if pages_meta.get("kind") != "notation":
        sys.exit("%s is not a notation book (pages.json says kind %r); set kind: notation in the manifest and rerun 20"
                 % (args.book, pages_meta.get("kind")))
    style = merge_style(pages_meta.get("style"))
    pages = pages_meta["pages"]
    if args.pages:
        wanted = {i + 1 for i in parse_pages(args.pages, max(p["page"] for p in pages))}
        pages = [p for p in pages if p["page"] in wanted]
    page_files = {p["page"]: p["file"] for p in pages}
    author = pages_meta.get("author") or "Unknown"
    book = {"key": args.book, "author": author, "author_key": notation_slug(author),
            "title": pages_meta.get("title") or pages_meta.get("work_title") or pages_meta.get("work") or args.book,
            "part": pages_meta.get("part"), "engines": []}

    layouts = layouts_for(book_dir, pages, style, args.force)
    dropped: list[dict] = []
    spans = link_pages(layouts, style, dropped)
    body_h_of = {lay["page"]: lay.get("body_h") for lay in layouts}
    engine = None
    if not args.no_grid:
        try:
            from lib.ocr_engines import TesseractEngine
            engine = TesseractEngine(psm=7, tess_lang="pan")
            engine.key = "tesseract-pan-psm7"
            try:
                engine.alt = TesseractEngine(psm=7, tess_lang="script/Gurmukhi")   # a second reading of the swar rows
            except SystemExit:
                engine.alt = None
        except SystemExit as err:
            print("grid reader off: %s" % err)
            args.no_grid = True
    con = corpus_connection()
    if con is None:
        print("no corpus database (CORPUS_DB or artifacts/gurbani.sqlite): shabads will not be named")

    out_dir = os.path.join(args.notations_dir, args.book)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    records, all_images, problems = [], [], []
    seq_by_page: dict[int, int] = {}
    for span in spans:
        grid_pages = [p for sec in span["sections"] for p, _ in sec["grids"]]
        page = grid_pages[0] if grid_pages else (span["heading_page"] or span["pages"][0])
        seq_by_page[page] = seq_by_page.get(page, 0) + 1
        rec, images = span_record(span, book, style, con, seq_by_page[page], book_dir, images_dir, page_files,
                                  args.no_grid, engine, body_h_of)
        if rec["shabad"].get("shabad_id") is None and "no-shabad-text" in rec["flags"] and "continued-from-prev" in rec["flags"] \
                and not rec["layout"].get("shabad_at_number"):
            # grids at the edge of the pages read with no shabad before them and none the bol row names: nothing anchors them
            dropped.append({"why": "edge-no-shabad", "pages": rec["pages"], "roles": ["grid"]})
            for im in images:
                try:
                    os.remove(os.path.join(images_dir, os.path.basename(im["file"])))
                    if im.get("thumb"):
                        os.remove(os.path.join(images_dir, os.path.basename(im["thumb"])))
                except OSError:
                    pass
            continue
        errs = validate(rec)
        if errs and rec["sections"] and all(e["path"].startswith("sections") for e in errs):
            # the grid reading is wrong somewhere; the notation (its images, its shabad) stands without it
            problems.append({"notation_id": rec["notation_id"], "errors": errs, "kept": "without the grid"})
            rec["sections"] = []
            if rec["kind"] == "notation":
                rec["kind"] = "partial"
            rec["flags"] = sorted(set(rec["flags"]) | {"partial-grid"})
            rec["source"]["content_hash"] = notation.content_hash(rec)
            errs = validate(rec)
        if errs:
            problems.append({"notation_id": rec["notation_id"], "errors": errs})
            continue
        records.append(rec)
        all_images.extend({**im, "notation_id": rec["notation_id"]} for im in images)

    borrowed = borrow_shabads(records)
    raag_notes = raag_records(raag_descriptions(layouts), args.book, book_dir, images_dir, page_files)
    with open(os.path.join(out_dir, "raags.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"book": args.book, "raag_notes": len(raag_notes)}}, ensure_ascii=False) + "\n")
        for note in raag_notes:
            fh.write(json.dumps(note, ensure_ascii=False) + "\n")
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    meta = notation.meta_for({"book": args.book, "author": author, "title": book["title"], "part": book.get("part"),
                              "language": pages_meta.get("language"), "style": style},
                             pages=len(pages), pages_notation=sum(1 for l in layouts if l["kind"] == "notation"),
                             pages_prose=sum(1 for l in layouts if l["kind"] == "prose"),
                             notations=len(records), kinds=kinds,
                             resolved=sum(1 for r in records if r["shabad"]["shabad_id"] is not None),
                             built=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    notation.write_jsonl(os.path.join(out_dir, "notations.jsonl"), meta, records)
    with open(os.path.join(out_dir, "images.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"book": args.book, "dpi": pages_meta.get("dpi"), "built": meta["built"], "files": all_images},
                  fh, ensure_ascii=False, indent=1)
    report = {"book": args.book, "pages": len(pages), "spans": len(spans), "notations": len(records), "kinds": kinds,
              "resolved": meta["resolved"], "problems": problems,
              "dropped": dropped, "dropped_pages": sorted({p for d in dropped for p in d["pages"]}),
              "flags": _count([f for r in records for f in r["flags"]]),
              "raags_used": _count([((r.get("heading") or {}).get("raag") or {}).get("key") for r in records]),
              "taals": _count([((r.get("heading") or {}).get("taal") or {}).get("key") for r in records]),
              "unknown_raags": sorted({(r["heading"]["raag"] or {}).get("printed") for r in records
                                       if r.get("heading") and r["heading"].get("raag") and r["heading"]["raag"].get("key") is None}),
              "unknown_taals": sorted({(r["heading"]["taal"] or {}).get("printed") for r in records
                                       if r.get("heading") and r["heading"].get("taal") and r["heading"]["taal"].get("key") is None})}
    os.makedirs(os.path.join(ROOT, "data", "raw"), exist_ok=True)
    report_path = os.path.join(ROOT, "data", "raw", "notation-report-%s.json" % args.book)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("%d pages -> %d spans -> %d records (%s), %d with a shabad; %d refused by the validator; %d part(s) without a shabad dropped"
          % (len(pages), len(spans), len(records), ", ".join("%s %d" % kv for kv in kinds.items()),
             meta["resolved"], len(problems), len(dropped)))
    for p in problems[:5]:
        print("  refused %s: %s" % (p["notation_id"], "; ".join("%s at %s" % (e["code"], e["path"]) for e in p["errors"][:3])))
    print("  -> %s  and  %s" % (out_dir, report_path))


def _count(items):
    out: dict = {}
    for it in items:
        out[str(it)] = out.get(str(it), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
