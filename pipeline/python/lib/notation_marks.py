"""
A book cut by hand: the owner's marks (tools/notation-marks.mjs) and what the
pipeline makes of them.

A mark is a start line and an end line drawn on the rendered pages -- a
continuation's end line is the next notation's start line -- with the shabad
the owner found by its first letters and the raag and taal typed. It lives in
<data>/marks/<book>.jsonl, append-only, the last line for a mark_id winning,
a line with "deleted": true a tombstone. The y of a line is in the pixel space
of data/ocr/<book>/pages/NNNN.png, the space every record's bbox and extent
already use, so a mark and a parser record compare directly.

  mark = {"mark_id", "book_key", "kind": notation | description | skip,
          "start": {"page", "y"}, "end": {"page", "y"}, "page_sizes": {"152": [w, h]},
          "shabad": {"source", "shabad_id", "line_id", "text", "query"} | None,
          "raag": {"printed", "key", "parent_key"} | None, "taal": {"printed", "key"} | None,
          "note", "by", "at"}

What the pipeline makes of them:
  - mark_to_record(): a notation record in the notations.jsonl shape, the
    shabad by the owner (method "manual", verified), one whole-width block
    crop per page between the lines;
  - apply_marks(): the parser's records a mark overlaps go, the mark's record
    takes their place -- a marked book is the owner's cut wherever a line was
    drawn and the parser's elsewhere;
  - compare(): the parser's cut against the owner's, per mark and as rates,
    the ground truth a reader change is measured by on a difficult book.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re

from lib.notation import SCHEMA_VERSION, VOCAB_VERSION, content_hash, make_id
from lib.notation_review import _overlap_1d, iou
from lib.notation_vocab import normalise_raag, normalise_taal, raag_key_from_corpus
from lib.paths import MARKS_DIR

MARKS_VERSION = "0.1.0"
KINDS = ("notation", "description", "skip")
CROP_PAD = 24                       # as 29_notation_parse.py cuts
MANUAL_SEQ = 900                    # a mark's record is book:PPPP:9nn -- the parser's seqs are the grids' order on the page, small
REPLACE_OVERLAP = 0.25              # a parser record this much of whose height (or the mark's) a mark covers on a page is replaced
EXACT, NEAR = 0.9, 0.5              # compare: the IoU of an exact and of a near cut (the review gate passes at 0.9)
_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


# ---- the ledger --------------------------------------------------------------

def ledger_path(book: str, marks_dir: str | None = None) -> str:
    return os.path.join(marks_dir or MARKS_DIR, book + ".jsonl")


def read_marks(book: str, marks_dir: str | None = None) -> list[dict]:
    """The live marks of a book: the last line per mark_id, tombstones out, in page order."""
    path = ledger_path(book, marks_dir)
    if not os.path.exists(path):
        return []
    last: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = json.loads(line)
            if not m.get("mark_id"):
                continue
            last[m["mark_id"]] = m
    out = [m for m in last.values() if not m.get("deleted")]
    out.sort(key=sort_key)
    return out


def append_mark(mark: dict, marks_dir: str | None = None) -> str:
    """One line more; the mark is checked first."""
    problems = validate_mark(mark)
    if problems:
        raise ValueError("; ".join(problems))
    path = ledger_path(mark["book_key"], marks_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(mark, ensure_ascii=False) + "\n")
    return path


def tombstone(book: str, mark_id: str, by: str = "") -> dict:
    return {"mark_id": mark_id, "book_key": book, "deleted": True, "by": by, "at": now()}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sort_key(mark: dict) -> tuple:
    s = mark.get("start") or {}
    return (int(s.get("page") or 0), int(s.get("y") or 0))


def validate_mark(mark: dict) -> list[str]:
    """What is wrong with a mark, as plain sentences; nothing when it is sound."""
    out = []
    if not _ID.match(str(mark.get("mark_id") or "")):
        out.append("mark_id must be a short slug")
    if not mark.get("book_key"):
        out.append("book_key is required")
    if mark.get("deleted"):
        return out
    if mark.get("kind") not in KINDS:
        out.append("kind must be one of %s" % ", ".join(KINDS))
    for end in ("start", "end"):
        e = mark.get(end) or {}
        if not isinstance(e.get("page"), int) or e["page"] < 1 or not isinstance(e.get("y"), (int, float)) or e["y"] < 0:
            out.append("%s needs a page (1-based) and a y (pixels, >= 0)" % end)
    if not out:
        s, e = mark["start"], mark["end"]
        if (e["page"], e["y"]) <= (s["page"], s["y"]):
            out.append("the end line must come after the start line")
        sizes = mark.get("page_sizes") or {}
        for p in range(s["page"], e["page"] + 1):
            if str(p) not in sizes or len(sizes[str(p)]) != 2:
                out.append("page_sizes must give [w, h] for page %d" % p)
    sh = mark.get("shabad")
    if sh is not None and (not isinstance(sh, dict) or not isinstance(sh.get("shabad_id"), int)):
        out.append("shabad must be null or carry an integer shabad_id")
    return out


# ---- geometry ----------------------------------------------------------------

def pages_of(mark: dict) -> list[int]:
    return list(range(int(mark["start"]["page"]), int(mark["end"]["page"]) + 1))


def extent_of(mark: dict) -> dict[int, list[int]]:
    """{page: [0, y0, w, y1]} across the pages the mark spans: the first from its start line, the last to its end line."""
    sizes = mark.get("page_sizes") or {}
    s, e = mark["start"], mark["end"]
    out = {}
    for p in pages_of(mark):
        w, h = sizes.get(str(p)) or [0, 0]
        y0 = int(s["y"]) if p == s["page"] else 0
        y1 = int(e["y"]) if p == e["page"] else int(h)
        out[p] = [0, y0, int(w), y1]
    return out


def record_extent(rec: dict) -> dict[int, list[int]]:
    from lib.notation_review import extent_of as rec_extent
    return rec_extent(rec)


def overlaps(mark: dict, rec: dict, share: float = REPLACE_OVERLAP) -> bool:
    """Whether the mark covers `share` of the record's height (or the record the mark's) on some page they share."""
    a, b = extent_of(mark), record_extent(rec)
    return any(_overlap_1d(a[p], b[p]) >= share for p in set(a) & set(b))


# ---- the record ------------------------------------------------------------------

def mark_to_record(mark: dict, book: dict, style: dict, seq: int, book_dir: str, images_dir: str,
                   page_files: dict[int, str], con=None, scriptures=None, cut: bool = True) -> dict:
    """
    A notation record from a notation mark: the shabad the owner picked, read
    from the corpus; the raag and taal as typed, normalised; one block crop per
    page between the lines (cut=False leaves the images uncut, for tests).
    `book` is 29's book dict (key, author, author_key, title, title_en, part).
    """
    from lib.notation_resolve import corpus_facts
    ext = extent_of(mark)
    pages = pages_of(mark)
    nid = make_id(book["key"], pages[0], seq)
    sh = mark.get("shabad") or {}
    sid = sh.get("shabad_id")
    source = sh.get("source") or "G"
    facts = corpus_facts(con, sid, source, scriptures) if sid is not None else {}
    shabad = {"shabad_id": sid, "source": source if sid is not None else None,
              "confidence": 1.0 if sid is not None else 0.0, "method": "manual" if sid is not None else "none",
              "line_ids": [sh["line_id"]] if sh.get("line_id") is not None else [],
              "ang": facts.get("ang"), "writer": facts.get("writer"), "first_line": facts.get("first_line"),
              "printed": [sh["text"]] if sh.get("text") else [], "ref": None, "verified": sid is not None,
              "raag": facts.get("raag")}
    raag = _raag_of(mark.get("raag"))
    taal = _taal_of(mark.get("taal"))
    heading = None
    if raag or taal:
        heading = {"raw": " ".join(x for x in ((mark.get("raag") or {}).get("printed"), (mark.get("taal") or {}).get("printed")) if x),
                   "number": None, "raag": raag, "taal": taal, "reet_of": None}
    raag_shabad = raag_key_from_corpus(facts.get("raag")) if source == "G" else None
    used = (raag or {}).get("key")
    raag_differs = bool(raag_shabad and used and used != raag_shabad and (raag or {}).get("parent_key") != raag_shabad)
    flags = ["partial-grid"]                 # the grid is not read: the record is the owner's cut and shabad, the crop its reading
    if sid is None:
        flags.append("unresolved-shabad")
    if raag and raag.get("key") is None:
        flags.append("raag-unknown")
    if taal and taal.get("key") is None:
        flags.append("taal-unknown")
    if raag_differs:
        flags.append("raag-differs")
    images = _cut_images(nid, ext, book_dir, images_dir, page_files) if cut else []
    rec = {
        "schema_version": SCHEMA_VERSION, "notation_id": nid, "book_key": book["key"], "page": pages[0], "seq": seq,
        "pages": pages, "kind": "partial", "script": style.get("script", "gurmukhi"),
        "heading": heading, "shabad": shabad, "raag_shabad": raag_shabad, "raag_differs": raag_differs,
        "sections": [],
        "layout": {"sections": [], "notes": [mark["note"]] if mark.get("note") else [], "text": [],
                   "extent": {str(p): b for p, b in ext.items()}, "shabad_at_number": None},
        "images": images,
        "source": {"book_key": book["key"], "author": book.get("author"), "author_key": book.get("author_key"),
                   "title": book.get("title"), "title_en": book.get("title_en"), "part": book.get("part"),
                   "pages": pages, "engines": [], "row_engine": None,
                   "parser": {"name": "marks", "version": MARKS_VERSION}, "commit": book.get("commit") or "unknown",
                   "vocab_version": VOCAB_VERSION, "built": now(), "content_hash": "", "gold": None},
        "quality": {"cells": 0, "unknown": 0, "conf_mean": 0.0, "matra_check": "unmeasured", "marks": 0, "watermark": 0.0},
        "flags": sorted(set(flags)), "verified": True,
        "mark_id": mark["mark_id"],
    }
    rec["source"]["content_hash"] = content_hash(rec)
    return rec


def _raag_of(raag: dict | None) -> dict | None:
    if not raag or not (raag.get("printed") or raag.get("key")):
        return None
    key, parent, conf = raag.get("key"), raag.get("parent_key"), 1.0
    if not key and raag.get("printed"):
        got = normalise_raag(raag["printed"])
        key, parent, conf = got["key"], got["parent"], got["confidence"]
    return {"printed": raag.get("printed"), "key": key, "parent_key": parent, "confidence": conf}


def _taal_of(taal: dict | None) -> dict | None:
    if not taal or not (taal.get("printed") or taal.get("key")):
        return None
    got = normalise_taal(taal.get("printed") or taal.get("key") or "")
    key = taal.get("key") or got["key"]
    return {"printed": taal.get("printed"), "key": key, "matras": got["matras"], "laya": got["laya"],
            "confidence": 1.0 if taal.get("key") else got["confidence"]}


def _cut_images(nid: str, ext: dict[int, list[int]], book_dir: str, images_dir: str, page_files: dict[int, str]) -> list[dict]:
    import cv2
    from lib.notation import image_name
    from lib.ocr_grid import crop_bilevel, thumbnail
    out = []
    n = 0
    for p in sorted(ext):
        f = page_files.get(p)
        img = cv2.imread(os.path.join(book_dir, "pages", f), cv2.IMREAD_GRAYSCALE) if f else None
        if img is None:
            continue
        ph, pw = img.shape[:2]
        margin = int(pw * 0.03)
        y0, y1 = max(0, ext[p][1]), min(ph, ext[p][3])
        if y1 - y0 < 8:
            continue
        bbox = [margin, y0, pw - margin, y1]
        n += 1
        fname = image_name(nid, n)
        dst = os.path.join(images_dir, fname)
        os.makedirs(images_dir, exist_ok=True)
        w, h = crop_bilevel(img, bbox, CROP_PAD, dst)
        entry = {"n": n, "file": "images/" + fname, "role": "block", "page": p, "bbox": bbox,
                 "w": w, "h": h, "bytes": os.path.getsize(dst), "sha256": _sha256(dst), "thumb": None}
        if n == 1:
            tname = image_name(nid, n, thumb=True)
            thumbnail(img, [max(0, bbox[0] - CROP_PAD), max(0, bbox[1] - CROP_PAD), bbox[2] + CROP_PAD, bbox[3] + CROP_PAD],
                      os.path.join(images_dir, tname))
            entry["thumb"] = "images/" + tname
            entry["thumb_sha256"] = _sha256(os.path.join(images_dir, tname))
            entry["thumb_bytes"] = os.path.getsize(os.path.join(images_dir, tname))
        out.append(entry)
    return out


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def description_note(mark: dict, n: int, book: str, book_dir: str, images_dir: str, page_files: dict[int, str], cut: bool = True) -> dict:
    """A raag-description mark as a raags.jsonl note (32 builds it into raag_notes)."""
    ext = extent_of(mark)
    raag = _raag_of(mark.get("raag")) or {}
    nid = "%s:%04d:%d" % (book, mark["start"]["page"], MANUAL_SEQ + n)
    images = _cut_images(nid, ext, book_dir, images_dir, page_files) if cut else []
    for im in images:
        im["file"] = im["file"].replace(nid.replace(":", "-"), "%s-raag-%04d-m%d" % (book, mark["start"]["page"], n))
    return {"n": n, "book_key": book, "raag": {"printed": raag.get("printed"), "key": raag.get("key"),
                                               "parent_key": raag.get("parent_key"), "confidence": raag.get("confidence")},
            "heading": (mark.get("raag") or {}).get("printed") or "", "page": mark["start"]["page"], "pages": pages_of(mark),
            "text": mark.get("note") or "", "image": images[0] if images else None, "images": images, "mark_id": mark["mark_id"]}


# ---- marks over the parser's records -----------------------------------------------

def seq_for(mark: dict, marks: list[dict]) -> int:
    """book:PPPP:9nn -- the nth mark starting on that page."""
    page = mark["start"]["page"]
    same = [m for m in marks if m.get("kind") != "description" and m["start"]["page"] == page]
    same.sort(key=sort_key)
    return MANUAL_SEQ + 1 + same.index(mark)


def apply_marks(records: list[dict], marks: list[dict], mark_records: dict[str, dict]) -> dict:
    """
    The parser's records a notation or skip mark overlaps go; a notation
    mark's record (mark_records[mark_id]) takes their place. Returns
    {"records", "replaced": [ids], "added": [ids], "skipped": n}.
    """
    live = [m for m in marks if m.get("kind") in ("notation", "skip")]
    replaced = []
    kept = []
    for r in records:
        if (r.get("source") or {}).get("parser", {}).get("name") == "marks":
            continue                                                   # a previous run's mark records: made afresh below
        if any(overlaps(m, r) for m in live):
            replaced.append(r["notation_id"])
        else:
            kept.append(r)
    added = [mark_records[m["mark_id"]] for m in live if m["kind"] == "notation" and m["mark_id"] in mark_records]
    out = kept + added
    out.sort(key=_record_order)
    return {"records": out, "replaced": replaced, "added": [r["notation_id"] for r in added],
            "skipped": sum(1 for m in live if m["kind"] == "skip")}


def _record_order(rec: dict) -> tuple:
    ext = record_extent(rec)
    first = min(rec.get("pages") or [rec.get("page") or 0])
    y = ext.get(first, [0, 0, 0, 0])[1]
    return (first, y, rec.get("seq") or 0)


def compare(records: list[dict], marks: list[dict]) -> dict:
    """
    The parser's cut against the owner's: for every notation mark the record
    that overlaps it most (IoU), whether its shabad and raag agree; and the
    rates -- exact (IoU >= 0.9), near (>= 0.5), missed; shabad agreement over
    the marks that name one.
    """
    rows = []
    parser = [r for r in records if (r.get("source") or {}).get("parser", {}).get("name") != "marks"]
    exts = [(r, record_extent(r)) for r in parser]
    for m in marks:
        if m.get("kind") != "notation":
            continue
        me = extent_of(m)
        best, best_iou = None, 0.0
        for r, e in exts:
            v = iou(me, e)
            if v > best_iou:
                best, best_iou = r, v
        sid = (m.get("shabad") or {}).get("shabad_id")
        rkey = (_raag_of(m.get("raag")) or {}).get("key")
        row = {"mark_id": m["mark_id"], "pages": pages_of(m), "notation_id": best["notation_id"] if best else None,
               "iou": round(best_iou, 3), "cut": "exact" if best_iou >= EXACT else "near" if best_iou >= NEAR else "missed",
               "shabad": None, "raag": None}
        if best and sid is not None:
            row["shabad"] = ((best.get("shabad") or {}).get("shabad_id") == sid)
        if best and rkey:
            row["raag"] = (((best.get("heading") or {}).get("raag") or {}).get("key") == rkey)
        rows.append(row)
    n = len(rows)
    cuts = {k: sum(1 for r in rows if r["cut"] == k) for k in ("exact", "near", "missed")}
    sh = [r["shabad"] for r in rows if r["shabad"] is not None]
    rg = [r["raag"] for r in rows if r["raag"] is not None]
    return {"marks": n, "cuts": cuts,
            "cut_rate": round((cuts["exact"] + cuts["near"]) / n, 3) if n else None,
            "shabad_rate": round(sum(sh) / len(sh), 3) if sh else None, "shabad_judged": len(sh),
            "raag_rate": round(sum(rg) / len(rg), 3) if rg else None, "raag_judged": len(rg),
            "rows": rows}
