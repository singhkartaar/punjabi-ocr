#!/usr/bin/env python3
"""
A scorecard per book from a sample of its pages: how well the pipeline reads
a book it has not seen, before anyone spends hours on the whole of it.

  29_bench_books.py --src /path/to/books                     every OCR work in the folder's manifest
  29_bench_books.py --src A --src B --book teeka-1           several folders, one book
  29_bench_books.py --src A --pages 1-6,40-45,200            these pages instead of the sample
  29_bench_books.py --src A --gt 20                          also sample 20 ground-truth lines per book
  29_bench_books.py --src A --score-only                     the scorecard from what is already on disk

For each work the steps 27_ingest_book.py would run up to the merge -- render
(20), the language's engines (21), the evaluation where there is ground truth
(24), the merge with the coverage pass (22) -- run on a sample of pages: the
first SAMPLE_HEAD pages and SAMPLE_SPREAD more spread evenly over the rest,
unless --pages says which. Then the merged pages and the paragraph reader
(lib/writings_ocr, the same reader 12 uses) are read back into one row per
book, printed as Markdown and written to data/raw/bench-<stamp>.json:

  reading     pages run, the reader (ocr | pdf-text | legacy-font), seconds a page,
              how the columns were found (rule | ink | words | none), lines by kind,
              matched Gurbani lines, lines recovered by the coverage pass and ink
              left uncovered, the three commonest reasons a line was routed
  structure   pages read as paired | columns | single, paragraphs paired and
              explaining a verse, the ang range the headers give against the
              manifest's `angs`, where each page's ang came from
  accuracy    where gt/lines.jsonl exists: word accuracy (all, prose, Gurbani)
              and the match precision; where gt/page-counts.jsonl exists: line
              recall and excess
  keys        the manifest keys the work set, so a row with a problem says what
              was already tried

Nothing here is new to the pipeline: the steps are 27's, the numbers are
24's and the reader's. A book that reads badly gets a manifest key (`layout`,
`header_pattern`, `angs`, `coverage`, `bleed`, `scripture`, `kind`), never a
branch in the code; a book that needs something no key covers is a decision
to write down with its number (docs/ocr-runbook.md, "A new book").
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib.ocr_engines import read_page
from lib.paths import CORPUS_DB, OCR_DIR, ROOT
from lib.writings_manifest import list_sources, load_manifest, parse_source

SAMPLE_HEAD = 12        # the first pages: the front matter and the first body pages, where the layout is set
SAMPLE_SPREAD = 8       # more pages spread evenly over the rest: the book may change shape after the preface
BENCH_STEPS = ("pages", "ocr", "eval", "merge")
# the manifest keys that shape how a book is read; a row lists the ones set
KNOBS = ("kind", "layout", "translate", "angs", "header_pattern", "coverage", "correct_agreed", "bleed", "scripture",
         "reader")
ROUTES_SHOWN = 3


def sample_pages(total: int, head: int = SAMPLE_HEAD, spread: int = SAMPLE_SPREAD) -> str:
    """'1-12,77,142,...': the first `head` pages and `spread` more spread evenly over the rest."""
    if total <= head + spread:
        return "1-%d" % total
    rest = total - head
    picks = sorted({head + max(1, round((i + 0.5) * rest / spread)) for i in range(spread)})
    return ",".join(["1-%d" % head] + [str(p) for p in picks if p <= total])


def works_in(src: str, only: str | None) -> list[tuple[str, dict]]:
    """(pdf path, meta) for the folder's works, or the one named."""
    manifest = load_manifest(src)
    if manifest is None:
        sys.exit("%s has no manifest.json; see docs/ocr-runbook.md for the shape" % src)
    pairs = [(p, parse_source(p, manifest)) for p in list_sources(src, manifest)]
    if only:
        pairs = [(p, m) for p, m in pairs if m["book"] == only]
    return pairs


def keys_set(src: str, meta: dict) -> list[str]:
    """The KNOBS the manifest sets for this work, at the top or on its entry."""
    manifest = load_manifest(src) or {}
    entry = next((w for w in manifest.get("works", []) if w.get("book") == meta["book"]
                  or w.get("file") == os.path.basename(meta.get("path", ""))), {})
    return [k for k in KNOBS if k in entry or k in manifest]


def run_steps(src: str, meta: dict, pages: str, ocr_dir: str, coverage: bool, engines: list[str] | None,
              dry_run: bool) -> dict[str, float]:
    """27's plan for one book, up to the merge, on these pages; seconds per step."""
    drv = import_module("27_ingest_book")
    steps = drv.plan(src, [meta], engines=engines, ocr_dir=ocr_dir, pages=pages, coverage=coverage)
    secs: dict[str, float] = defaultdict(float)
    for step, argv in steps:
        if step not in BENCH_STEPS:
            continue
        if isinstance(argv, str):
            print("  [%s] %s" % (step, argv))
            continue
        print("  [%s] python %s" % (step, drv.render(argv)))
        if dry_run:
            continue
        t0 = time.time()
        code = subprocess.call([sys.executable] + argv, cwd=HERE)
        secs[step] += time.time() - t0
        if code != 0:
            sys.exit("step %s failed (exit %d) for %s" % (step, code, meta["book"]))
    return dict(secs)


def scorecard(book_dir: str, meta: dict, corpus: str | None = None, merged: str = "merged") -> dict:
    """One book's row, from its merged pages, the paragraph reader and its ground truth."""
    ev = import_module("24_ocr_eval")
    from lib.writings_ocr import read_ocr_book
    row: dict = {"book": meta["book"], "reader": meta.get("reader") or "ocr", "language": meta.get("language", "en"),
                 "kind": meta.get("kind", "essay")}
    pages_path = os.path.join(book_dir, "pages.json")
    if not os.path.exists(pages_path):
        row["error"] = "no pages.json: nothing rendered"
        return row
    with open(pages_path, encoding="utf-8") as fh:
        pages_meta = json.load(fh)
    row["probe"] = (pages_meta.get("probe") or {}).get("shape")
    files = sorted(glob.glob(os.path.join(book_dir, merged, "*.jsonl")))
    row["pages"] = len(files)
    if not files:
        row["error"] = "no merged pages"
        return row
    kinds: Counter = Counter()
    routes: Counter = Counter()
    columns: Counter = Counter()
    matched = recovered = split = adopted = 0
    for path in files:
        pmeta, lines = read_page(path)
        columns[pmeta.get("columns_source") or ("words" if pmeta.get("columns") else "none")] += 1
        for ln in lines:
            if ln.get("dropped") or ln.get("zone") not in ("body", "footnote"):
                continue
            kinds[ln.get("kind") or "-"] += 1
            matched += bool(ln.get("matches"))
            recovered += bool(ln.get("recovered"))
            split += bool(ln.get("split"))
            adopted += bool(ln.get("adopted_from"))
            if ln.get("route"):
                routes[ln["route"]] += 1
    row["columns"] = dict(columns)
    row["lines"] = {k: kinds[k] for k in ("gurbani", "gurbani-unmatched", "heading", "commentary")} | \
        {k: v for k, v in kinds.items() if k not in ("gurbani", "gurbani-unmatched", "heading", "commentary")}
    row["matched"] = matched
    row["recovered"] = recovered
    row["split"] = split
    row["adopted"] = adopted
    row["routes"] = routes.most_common(ROUTES_SHOWN)
    cov = ev.coverage_summary(book_dir, merged)
    if cov:
        row["coverage"] = cov

    # the manifest's keys, as 12 passes them: a pages.json rendered before the
    # manifest gained `angs` would otherwise leave the reader without them
    book = read_ocr_book(book_dir, layout=meta.get("layout") or "auto", kind=meta.get("kind") or "essay",
                         angs=meta.get("angs"))
    layouts: Counter = Counter(p.get("layout") or "single" for p in book["pages"])
    paras = [par for p in book["pages"] for par in p["paragraphs"]]
    row["layout"] = dict(layouts)
    # where each page's ang came from, after the reader's smoothing (a merge
    # from before smoothing has none of its own)
    row["ang_source"] = dict(Counter(p.get("ang_source") or "-" for p in book["pages"]))
    row["paragraphs"] = {"n": len(paras), "quotes": sum(1 for p in paras if p["style"] == "quote"),
                         "paired": sum(1 for p in paras if p.get("pair")),
                         "explains": sum(1 for p in paras if p.get("explains")),
                         "footnotes": sum(1 for p in paras if p["style"] == "footnote")}
    froms = [p["ang_from"] for p in book["pages"] if p.get("ang_from") and p.get("ang_source") in ("header", "smoothed")]
    tos = [p.get("ang_to") or p["ang_from"] for p in book["pages"]
           if p.get("ang_from") and p.get("ang_source") in ("header", "smoothed")]
    row["angs"] = {"found": [min(froms), max(tos)] if froms else None, "manifest": meta.get("angs")}

    gt_path = os.path.join(book_dir, "gt", "lines.jsonl")
    gt = [g for g in ev.read_jsonl(gt_path) if g.get("verified")] if os.path.exists(gt_path) else []
    if gt:
        index = ev.load_index(corpus or CORPUS_DB) if row["language"] == "pa" else None
        r = ev.evaluate(book_dir, merged, gt, index, row["language"], meta.get("scripture", "G"))
        acc = {"lines": r["lines"], "word_acc": r["word_acc"], "cer": r["cer"],
               "prose": (r.get("by_kind", {}).get("commentary") or {}).get("word_acc"),
               "gurbani": (r.get("by_kind", {}).get("gurbani") or {}).get("word_acc")}
        if r.get("gurbani_match"):
            acc["match_precision"] = r["gurbani_match"]["precision"]
        if r.get("by_provenance", {}).get("recovered"):
            acc["recovered_word_acc"] = r["by_provenance"]["recovered"]["word_acc"]
        row["accuracy"] = acc
    counts = ev.read_jsonl(os.path.join(book_dir, "gt", "page-counts.jsonl"))
    lc = ev.line_counts(book_dir, merged, counts) if counts else None
    if lc:
        row["line_counts"] = lc
    return row


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.3f" % v
    if isinstance(v, dict):
        return " ".join("%s %s" % (k, _fmt(x)) for k, x in v.items()) or "-"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) if not isinstance(x, (list, tuple)) else ":".join(str(y) for y in x) for x in v) or "-"
    return str(v)


def markdown(rows: list[dict]) -> str:
    """The rows as two Markdown tables: how each book read, and what it is made of."""
    out = ["| book | reader | pages | s/page | columns | lines g/gu/h/c | matched | recovered | uncovered | routes | word acc (prose, gurbani) | line recall |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            out.append("| %s | %s | - | - | %s | | | | | | | |" % (r["book"], r.get("reader", "-"), r["error"]))
            continue
        ln = r.get("lines", {})
        cov = r.get("coverage") or {}
        acc = r.get("accuracy") or {}
        lc = r.get("line_counts") or {}
        out.append("| %s | %s | %d | %s | %s | %d/%d/%d/%d | %d | %d | %s | %s | %s | %s |" % (
            r["book"], r["reader"], r["pages"], _fmt(r.get("sec_per_page")), _fmt(r.get("columns")),
            ln.get("gurbani", 0), ln.get("gurbani-unmatched", 0), ln.get("heading", 0), ln.get("commentary", 0),
            r.get("matched", 0), r.get("recovered", 0),
            _fmt(cov.get("uncovered_after")) if cov else "off",
            _fmt(r.get("routes")),
            ("%s (%s, %s)" % (_fmt(acc.get("word_acc")), _fmt(acc.get("prose")), _fmt(acc.get("gurbani")))) if acc else "no gt",
            ("%s excess %s" % (_fmt(lc.get("recall")), _fmt(lc.get("excess")))) if lc else "no counts"))
    out += ["", "| book | kind | layout p/c/s | paragraphs | quotes | paired | explains | angs found | angs manifest | ang source | keys set |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            continue
        lay = r.get("layout", {})
        pg = r.get("paragraphs", {})
        angs = r.get("angs", {})
        out.append("| %s | %s | %d/%d/%d | %d | %d | %d | %d | %s | %s | %s | %s |" % (
            r["book"], r.get("kind", "essay"), lay.get("paired", 0), lay.get("columns", 0), lay.get("single", 0),
            pg.get("n", 0), pg.get("quotes", 0), pg.get("paired", 0), pg.get("explains", 0),
            _fmt(angs.get("found")), _fmt(angs.get("manifest")), _fmt(r.get("ang_source")),
            ", ".join(r.get("keys", [])) or "-"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", action="append", required=True, help="folder with manifest.json and PDFs; repeatable")
    ap.add_argument("--book", help="one work's book key")
    ap.add_argument("--pages", help="'1-12,77' instead of the sample (%d first + %d spread)" % (SAMPLE_HEAD, SAMPLE_SPREAD))
    ap.add_argument("--engines", help="comma list replacing the language default")
    ap.add_argument("--coverage", dest="coverage", action="store_true", default=True,
                    help="merge with the coverage pass (default for a bench: the point is to see what is missed)")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--gt", type=int, default=0, help="sample this many ground-truth lines per book after the merge")
    ap.add_argument("--score-only", action="store_true", help="run nothing; score what is on disk")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--json", help="default: data/raw/bench-<stamp>.json")
    args = ap.parse_args()

    rows = []
    for src in args.src:
        src = os.path.abspath(src)
        for path, meta in works_in(src, args.book):
            meta = {**meta, "path": path}
            book_dir = os.path.join(args.out, meta["book"])
            print("%s (%s, %s, %s)" % (meta["book"], meta.get("language", "en"), meta.get("reader") or "ocr",
                                       meta.get("kind", "essay")))
            if (meta.get("reader") or "ocr") != "ocr":
                rows.append({"book": meta["book"], "reader": meta["reader"], "kind": meta.get("kind", "essay"),
                             "error": "not a scanned book: 12_ingest_writings.py reads its text layer directly",
                             "keys": keys_set(src, meta)})
                continue
            secs: dict[str, float] = {}
            if not args.score_only:
                pages = args.pages
                if not pages:
                    from lib.ocr_probe import probe_pdf
                    pages = sample_pages(probe_pdf(path)["pages"])
                print("  pages %s" % pages)
                secs = run_steps(src, meta, pages, args.out, args.coverage,
                                 [e.strip() for e in args.engines.split(",")] if args.engines else None, args.dry_run)
                if args.gt and not args.dry_run:
                    argv = [os.path.join(HERE, "23_ocr_gt.py"), "--book", meta["book"], "--sample", str(args.gt),
                            "--out", args.out]
                    print("  [gt] python %s" % " ".join(os.path.relpath(a, HERE) if a.startswith(HERE) else a for a in argv))
                    subprocess.call([sys.executable] + argv, cwd=HERE)
                    print("  review %s, then 23_ocr_gt.py --book %s --promote and run this again with --score-only"
                          % (os.path.join(book_dir, "gt", "review.html"), meta["book"]))
            if args.dry_run:
                continue
            row = scorecard(book_dir, meta, args.corpus)
            row["keys"] = keys_set(src, meta)
            if secs and row.get("pages"):
                row["seconds"] = {k: round(v, 1) for k, v in secs.items()}
                row["sec_per_page"] = round(sum(secs.values()) / row["pages"], 2)
            rows.append(row)
    if args.dry_run:
        return
    print()
    print(markdown(rows))
    out = args.json or os.path.join(ROOT, "data", "raw", "bench-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"src": args.src, "rows": rows, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh,
                  ensure_ascii=False, indent=1)
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
