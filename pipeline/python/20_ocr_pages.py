"""
Scanned PDFs to page images: the first step of OCR ingestion.

  20_ocr_pages.py --src ./books                      every OCR work in the manifest
  20_ocr_pages.py --src ./books --book santhya-vol-1 --pages 1-20
  20_ocr_pages.py --pdf book.pdf --book my-book --bleed

Writes data/ocr/<book>/pages/NNNN.png (300 dpi grayscale, deskewed) and
data/ocr/<book>/pages.json: the probe (what kind of PDF this is), the options
used, and one record per page with its size, the deskew angle applied, the
border cut, and the scanner's own text stamps with their boxes.

Resumable: a page whose PNG exists is skipped unless --force. Parallel over
page ranges; each worker opens the PDF itself.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_pages import DPI, parse_pages, render_pages
from lib.ocr_probe import probe_pdf
from lib.paths import OCR_DIR
from lib.writings_manifest import list_sources, load_manifest, parse_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def chunks(seq: list, n: int) -> list[list]:
    n = max(1, n)
    size = max(1, (len(seq) + n - 1) // n)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def run_book(pdf: str, book: str, meta: dict, args) -> dict:
    book_dir = os.path.join(args.out, book)
    pages_dir = os.path.join(book_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    probe = probe_pdf(pdf)
    if probe["shape"] != "image" and meta.get("reader") != "ocr":
        print("  %s: %s PDF (%s); not an OCR source, skipped" % (book, probe["shape"], ", ".join(probe["fonts"][:3])))
        return {"book": book, "skipped": probe["shape"]}
    bleed = bool(args.bleed or meta.get("bleed"))
    crop = not args.no_crop
    deskew = not args.no_deskew
    indices = parse_pages(args.pages, probe["pages"])
    print("  %s: %d pages (%s, %s, ~%s dpi) -> %d to render%s"
          % (book, probe["pages"], "1-bit" if probe["bilevel"] else "gray/colour",
             "tiled" if probe["tiled"] else "whole", probe["dpi"], len(indices),
             " [bleed]" if bleed else ""))
    t0 = time.time()
    records: list[dict] = []
    if args.workers <= 1 or len(indices) < 4:
        records = render_pages(pdf, indices, pages_dir, args.dpi, crop, deskew, bleed, args.force)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(render_pages, pdf, part, pages_dir, args.dpi, crop, deskew, bleed, args.force)
                    for part in chunks(indices, args.workers * 2)]
            for f in cf.as_completed(futs):
                records.extend(f.result())
    records.sort(key=lambda r: r["page"])
    done = [r for r in records if not r.get("skipped")]
    secs = time.time() - t0
    angles = [abs(r.get("angle", 0.0)) for r in done]
    print("    rendered %d (%d already there) in %.0fs; deskew mean %.2f deg, max %.2f"
          % (len(done), len(records) - len(done), secs,
             sum(angles) / max(len(angles), 1), max(angles) if angles else 0.0))

    # merge into pages.json so partial runs accumulate
    meta_path = os.path.join(book_dir, "pages.json")
    existing: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    by_page = {r["page"]: r for r in existing.get("pages", [])}
    for r in records:
        if r.get("skipped") and r["page"] in by_page:
            continue
        if r.get("skipped"):
            r = dict(r)
            path = os.path.join(pages_dir, r["file"])
            try:
                from PIL import Image
                with Image.open(path) as im:
                    r["w"], r["h"] = im.width, im.height
            except Exception:                            # noqa: BLE001
                pass
        by_page[r["page"]] = r
    out = {"pdf": pdf, "book": book, "dpi": args.dpi, "language": meta.get("language", "en"),
           "author": meta.get("author"), "work": meta.get("work"), "part": meta.get("part"),
           "title": meta.get("title") or meta.get("work_title"),
           "scripture": meta.get("scripture", "G"), "quote_policy": meta.get("quote_policy"),
           "kind": meta.get("kind", "essay"), "layout": meta.get("layout", "auto"), "angs": meta.get("angs"),
           "header_pattern": meta.get("header_pattern"), "coverage": meta.get("coverage"), "style": meta.get("style"),
           "probe": probe, "options": {"crop": crop, "deskew": deskew, "bleed": bleed},
           "pages": [by_page[k] for k in sorted(by_page)]}
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return {"book": book, "rendered": len(done), "pages": probe["pages"], "seconds": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder with manifest.json and the PDFs")
    ap.add_argument("--pdf", help="one PDF instead of a manifest")
    ap.add_argument("--book", help="book key (required with --pdf; a filter with --src)")
    ap.add_argument("--out", default=OCR_DIR)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--pages", help="'1-20,35' (1-based); default all")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-crop", action="store_true", help="do not cut a dark scanner border")
    ap.add_argument("--bleed", action="store_true", help="suppress reverse-side show-through")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs: list[tuple[str, str, dict]] = []
    if args.pdf:
        if not args.book:
            sys.exit("--pdf needs --book")
        jobs.append((args.pdf, args.book, {"reader": "ocr"}))
    elif args.src:
        manifest = load_manifest(args.src)
        if manifest is None:
            sys.exit("no manifest.json in %s (see lib/writings_manifest.py)" % args.src)
        for path in list_sources(args.src, manifest):
            meta = parse_source(path, manifest)
            if args.book and meta["book"] != args.book:
                continue
            if meta.get("reader") not in (None, "ocr"):
                continue
            jobs.append((path, meta["book"], meta))
    else:
        sys.exit("give --src or --pdf")
    if not jobs:
        sys.exit("nothing to do")
    print("%d book(s), %d workers, %d dpi" % (len(jobs), args.workers, args.dpi))
    results = [run_book(pdf, book, meta, args) for pdf, book, meta in jobs]
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
