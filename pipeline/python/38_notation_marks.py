#!/usr/bin/env python3
"""
The owner's hand-drawn cuts of a book (tools/notation-marks.mjs writes them to
<data>/marks/<book>.jsonl; lib/notation_marks.py reads them), and what the
pipeline makes of them.

  38_notation_marks.py books                       every work of every manifest, with its PDF, pages, renders, marks (JSON)
  38_notation_marks.py render --book K --pages 1-20   render the pages (20_ocr_pages.py through the book's manifest)
  38_notation_marks.py vocab                       the raag and taal vocabulary, for the page's autocomplete (JSON)
  38_notation_marks.py records --book K            the marks as records: merged over the parser's notations.jsonl
                                                   where there is one, alone where the book was never read
  38_notation_marks.py compare --book K            the parser's cut against the owner's marks (the last run's, or now)
  38_notation_marks.py status                      marks per book

The y of a mark is in the pixel space of data/ocr/<book>/pages/NNNN.png, so a
page must be rendered before it is marked; `render` does that for a book the
OCR never read (a scan too coarse for it), and `records` renders whatever a
mark spans that is not there yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import notation  # noqa: E402
from lib.notation import merge_style, notation_slug  # noqa: E402
from lib.notation_marks import (MARKS_VERSION, apply_marks, compare, description_note, mark_to_record,  # noqa: E402
                                pages_of, read_marks, seq_for)
from lib.notation_vocab import RAAGS, TAALS, VERSION as VOCAB_VERSION  # noqa: E402
from lib.paths import CORPUS_DB, MARKS_DIR, NOTATIONS_DIR, OCR_DIR, SCRIPTURES_DB, SCRIPTURES_FULL_DB  # noqa: E402
from lib.writings_manifest import list_sources, load_manifest, parse_source  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# the library's manifests and PDFs: <data>/keertan/<author>/ (link_library.py in the data repository)
KEERTAN_DIR = os.environ.get("KEERTAN_DIR") or os.path.join(os.path.dirname(NOTATIONS_DIR), "keertan")


# ---- the books -----------------------------------------------------------------

def manifests(keertan_dir: str) -> list[tuple[str, dict]]:
    out = []
    for d in sorted(glob.glob(os.path.join(keertan_dir, "*"))):
        m = load_manifest(d)
        if m is not None:
            out.append((d, m))
    return out


def works(keertan_dir: str) -> list[dict]:
    """Every work of every manifest: its key, folder, PDF (when the library is linked), kind, author, style."""
    out = []
    for d, m in manifests(keertan_dir):
        present = {os.path.basename(p): p for p in list_sources(d, m)}
        for w in m.get("works", []):
            path = present.get(w["file"])
            meta = parse_source(path, m) if path else {}
            out.append({"book": w.get("book") or meta.get("book"), "src": d, "folder": os.path.basename(d),
                        "file": w["file"], "pdf": path, "title": w.get("title") or meta.get("title") or w.get("work"),
                        "author": m.get("author") or meta.get("author"), "kind": w.get("kind") or m.get("kind") or meta.get("kind"),
                        "style": merge_style(m.get("style"), w.get("style")) if (m.get("style") or w.get("style")) else None})
    return [w for w in out if w["book"]]


def source_of(book: str, keertan_dir: str) -> dict | None:
    return next((w for w in works(keertan_dir) if w["book"] == book), None)


def pdf_pages(pdf: str) -> int | None:
    try:
        import pymupdf
        with pymupdf.open(pdf) as doc:
            return doc.page_count
    except Exception:                                     # noqa: BLE001
        return None


def book_state(w: dict, ocr_dir: str, notations_dir: str, marks_dir: str) -> dict:
    book_dir = os.path.join(ocr_dir, w["book"])
    meta_path = os.path.join(book_dir, "pages.json")
    pages = None
    rendered = 0
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            pm = json.load(fh)
        pages = (pm.get("probe") or {}).get("pages")
        rendered = sum(1 for p in pm.get("pages", []) if os.path.exists(os.path.join(book_dir, "pages", p["file"])))
    if pages is None and w.get("pdf"):
        pages = pdf_pages(w["pdf"])
    records = 0
    np = os.path.join(notations_dir, w["book"], "notations.jsonl")
    if os.path.exists(np):
        with open(np, encoding="utf-8") as fh:
            records = max(0, sum(1 for line in fh if line.strip()) - 1)
    marks = read_marks(w["book"], marks_dir)
    return {**w, "pages": pages, "rendered": rendered, "records": records, "marks": len(marks),
            "marked_to": max((m["end"]["page"] for m in marks), default=None)}


# ---- rendering ---------------------------------------------------------------------

def render(book: str, pages: str, keertan_dir: str, ocr_dir: str, workers: int | None = None) -> dict:
    """The pages rendered through the book's manifest, so the render carries the book's kind and style."""
    w = source_of(book, keertan_dir)
    if not w or not w.get("pdf"):
        sys.exit("%s: no PDF in the library (link_library.py in the data repository names it)" % book)
    cmd = [sys.executable, os.path.join(HERE, "20_ocr_pages.py"), "--src", w["src"], "--book", book, "--out", ocr_dir, "--pages", pages]
    if workers:
        cmd += ["--workers", str(workers)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode:
        sys.stderr.write(got.stdout + got.stderr)
        sys.exit("render failed for %s pages %s" % (book, pages))
    return {"book": book, "pages": pages, "log": got.stdout.strip().splitlines()[-2:]}


def missing_pages(book: str, needed: list[int], ocr_dir: str) -> list[int]:
    book_dir = os.path.join(ocr_dir, book)
    meta_path = os.path.join(book_dir, "pages.json")
    files = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            files = {p["page"]: p["file"] for p in json.load(fh).get("pages", [])}
    return [p for p in needed if p not in files or not os.path.exists(os.path.join(book_dir, "pages", files[p]))]


def spec_of(pages: list[int]) -> str:
    """[3,4,5,9] -> '3-5,9'."""
    out = []
    for p in sorted(set(pages)):
        if out and out[-1][1] == p - 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)


# ---- the records -------------------------------------------------------------------

def connection(path: str | None) -> sqlite3.Connection | None:
    if path and os.path.exists(path):
        return sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    return None


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def build(book: str, keertan_dir: str, ocr_dir: str, notations_dir: str, marks_dir: str, no_images: bool = False) -> dict:
    """
    The marks of a book as records. Over the parser's notations.jsonl when
    there is one (the records a mark overlaps go, the mark's take their
    place; the rest stay), alone when the book was never read. images.json
    and raags.jsonl follow; marks.json keeps what was done and the compare.
    """
    marks = read_marks(book, marks_dir)
    if not marks:
        sys.exit("%s: no marks in %s" % (book, os.path.join(marks_dir, book + ".jsonl")))
    needed = sorted({p for m in marks for p in pages_of(m)})
    gone = missing_pages(book, needed, ocr_dir)
    if gone:
        print("  rendering %d page(s) the marks span: %s" % (len(gone), spec_of(gone)))
        render(book, spec_of(gone), keertan_dir, ocr_dir)
    book_dir = os.path.join(ocr_dir, book)
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        pages_meta = json.load(fh)
    style = merge_style(pages_meta.get("style"))
    page_files = {p["page"]: p["file"] for p in pages_meta["pages"]}
    author = pages_meta.get("author") or "Unknown"
    info = {"key": book, "author": author, "author_key": notation_slug(author),
            "title": pages_meta.get("title") or pages_meta.get("work") or book, "title_en": None,
            "part": pages_meta.get("part"), "commit": code_commit()}
    out_dir = os.path.join(notations_dir, book)
    images_dir = os.path.join(out_dir, "images")
    con = connection(CORPUS_DB)
    scriptures = connection(SCRIPTURES_FULL_DB) or connection(SCRIPTURES_DB)
    mark_records = {}
    for m in marks:
        if m.get("kind") == "notation":
            mark_records[m["mark_id"]] = mark_to_record(m, info, style, seq_for(m, marks), book_dir, images_dir, page_files,
                                                        con, scriptures, cut=not no_images)
    notes = []
    for n, m in enumerate([m for m in marks if m.get("kind") == "description"], 1):
        notes.append(description_note(m, n, book, book_dir, images_dir, page_files, cut=not no_images))

    path = os.path.join(out_dir, "notations.jsonl")
    old_meta, old_records = notation.read_jsonl(path) if os.path.exists(path) else ({}, [])
    parser_records = [r for r in old_records if (r.get("source") or {}).get("parser", {}).get("name") != "marks"]
    cmp = compare(parser_records, marks) if parser_records else None
    applied = apply_marks(old_records, marks, mark_records)
    records = applied["records"]

    old_images = []
    ip = os.path.join(out_dir, "images.json")
    if os.path.exists(ip):
        with open(ip, encoding="utf-8") as fh:
            old_images = json.load(fh).get("files") or []
    keep = {r["notation_id"] for r in records} - set(applied["added"])
    files = [im for im in old_images if im.get("notation_id") in keep]
    for r in records:
        if r["notation_id"] in applied["added"]:
            files += [{**im, "notation_id": r["notation_id"]} for im in r["images"]]

    rp = os.path.join(out_dir, "raags.jsonl")
    old_notes = []
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            old_notes = [json.loads(l) for l in fh if l.strip()]
        old_notes = [x for x in old_notes if "_meta" not in x and not x.get("mark_id")]
    all_notes = old_notes + notes
    with open(rp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"book": book, "raag_notes": len(all_notes)}}, ensure_ascii=False) + "\n")
        for x in all_notes:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    meta = notation.meta_for({"book": book, "author": author, "title": info["title"], "part": info["part"],
                              "language": pages_meta.get("language"), "style": style},
                             pages=old_meta.get("pages") or (pages_meta.get("probe") or {}).get("pages") or len(page_files),
                             pages_notation=old_meta.get("pages_notation"), pages_prose=old_meta.get("pages_prose"),
                             notations=len(records), kinds=kinds,
                             resolved=sum(1 for r in records if r["shabad"]["shabad_id"] is not None),
                             marks={"version": MARKS_VERSION, "count": len(marks), "applied": built},
                             built=built)
    os.makedirs(out_dir, exist_ok=True)
    notation.write_jsonl(path, meta, records)
    with open(ip, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"book": book, "dpi": pages_meta.get("dpi"), "built": built, "files": files}, fh, ensure_ascii=False, indent=1)
    summary = {"book": book, "built": built, "marks": len(marks),
               "kinds": {k: sum(1 for m in marks if m.get("kind") == k) for k in ("notation", "description", "skip")},
               "with_shabad": sum(1 for m in marks if (m.get("shabad") or {}).get("shabad_id") is not None),
               "replaced": applied["replaced"], "added": applied["added"], "skipped": applied["skipped"],
               "parser_records": len(parser_records), "records": len(records), "compare": cmp}
    with open(os.path.join(out_dir, "marks.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def print_compare(cmp: dict | None, book: str) -> None:
    if not cmp:
        print("%s: nothing to compare (the book has no parser records)" % book)
        return
    c = cmp["cuts"]
    print("%s: %d mark(s) against the parser -- cut exact %d, near %d, missed %d (%s); shabad agrees %s of %d; raag agrees %s of %d"
          % (book, cmp["marks"], c["exact"], c["near"], c["missed"],
             "%.0f%%" % (100 * cmp["cut_rate"]) if cmp["cut_rate"] is not None else "-",
             int(round((cmp["shabad_rate"] or 0) * cmp["shabad_judged"])), cmp["shabad_judged"],
             int(round((cmp["raag_rate"] or 0) * cmp["raag_judged"])), cmp["raag_judged"]))
    for r in cmp["rows"]:
        pages = "%d-%d" % (r["pages"][0], r["pages"][-1]) if len(r["pages"]) > 1 else str(r["pages"][0])
        print("  %-10s p.%-8s %-6s iou %.2f  %-32s shabad %s  raag %s"
              % (r["mark_id"], pages, r["cut"], r["iou"], r["notation_id"] or "-",
                 {True: "same", False: "DIFFERS", None: "-"}[r["shabad"]], {True: "same", False: "DIFFERS", None: "-"}[r["raag"]]))


# ---- main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keertan-dir", default=KEERTAN_DIR)
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory")
    ap.add_argument("--notations-dir", default=NOTATIONS_DIR)
    ap.add_argument("--marks-dir", default=MARKS_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("books")
    r = sub.add_parser("render"); r.add_argument("--book", required=True); r.add_argument("--pages", required=True); r.add_argument("--workers", type=int)
    sub.add_parser("vocab")
    b = sub.add_parser("records"); b.add_argument("--book", required=True); b.add_argument("--no-images", action="store_true")
    c = sub.add_parser("compare"); c.add_argument("--book", required=True); c.add_argument("--now", action="store_true", help="compare against notations.jsonl as it is, not the last records run")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "books":
        rows = [book_state(w, args.out, args.notations_dir, args.marks_dir) for w in works(args.keertan_dir)]
        print(json.dumps({"keertan_dir": args.keertan_dir, "books": rows}, ensure_ascii=False))
    elif args.cmd == "render":
        got = render(args.book, args.pages, args.keertan_dir, args.out, args.workers)
        print(json.dumps(got, ensure_ascii=False))
    elif args.cmd == "vocab":
        print(json.dumps({"version": VOCAB_VERSION,
                          "raags": [{"key": k, "en": v.get("en"), "pa": v.get("pa"), "parent": v.get("parent"),
                                     "aliases": v.get("aliases") or {}} for k, v in RAAGS.items()],
                          "taals": [{"key": k, "en": v.get("en"), "pa": v.get("pa"), "matras": v.get("matras"),
                                     "aliases": v.get("aliases") or {}} for k, v in TAALS.items()]}, ensure_ascii=False))
    elif args.cmd == "records":
        s = build(args.book, args.keertan_dir, args.out, args.notations_dir, args.marks_dir, no_images=args.no_images)
        print("%s: %d mark(s) (%d notation, %d description, %d skip; %d with a shabad) -> %d record(s): %d of the parser's replaced, %d added"
              % (s["book"], s["marks"], s["kinds"]["notation"], s["kinds"]["description"], s["kinds"]["skip"], s["with_shabad"],
                 s["records"], len(s["replaced"]), len(s["added"])))
        if s["compare"]:
            print_compare(s["compare"], s["book"])
    elif args.cmd == "compare":
        out_dir = os.path.join(args.notations_dir, args.book)
        mp = os.path.join(out_dir, "marks.json")
        if not args.now and os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                print_compare(json.load(fh).get("compare"), args.book)
        else:
            path = os.path.join(out_dir, "notations.jsonl")
            _, records = notation.read_jsonl(path) if os.path.exists(path) else ({}, [])
            print_compare(compare(records, read_marks(args.book, args.marks_dir)) if records else None, args.book)
    elif args.cmd == "status":
        for f in sorted(glob.glob(os.path.join(args.marks_dir, "*.jsonl"))):
            book = os.path.basename(f)[:-6]
            marks = read_marks(book, args.marks_dir)
            kinds = {k: sum(1 for m in marks if m.get("kind") == k) for k in ("notation", "description", "skip")}
            with_sh = sum(1 for m in marks if (m.get("shabad") or {}).get("shabad_id") is not None)
            mp = os.path.join(args.notations_dir, book, "marks.json")
            applied = None
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as fh:
                    applied = json.load(fh).get("built")
            print("%-50s %4d mark(s): %d notation (%d with a shabad), %d description, %d skip; pages %s; records built %s"
                  % (book, len(marks), kinds["notation"], with_sh, kinds["description"], kinds["skip"],
                     ("%d-%d" % (marks[0]["start"]["page"], max(m["end"]["page"] for m in marks))) if marks else "-", applied or "never"))


if __name__ == "__main__":
    main()
