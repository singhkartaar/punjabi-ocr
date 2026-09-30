"""
Run one OCR engine over a book's page images.

  21_ocr_run.py --book santhya-vol-1 --engine tesseract --lang pa --pages 1-20
  21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --budget-usd 2 --dry-run
  21_ocr_run.py --book ten-masters --engine surya --lang en

Writes data/ocr/<book>/ocr/<engine>/NNNN.jsonl, one file per page, the first
line a _meta record (engine, model, lang, page size, seconds). Resumable: a page
whose file exists is skipped unless --force.

Paid engines (vision, gemini) run through lib/ocr_route.Budget: the monthly cap
is --budget-usd, every request is written to data/ocr/costs.jsonl, and
--dry-run prints the page count and the cost and sends nothing. --max-pages is
a second, per-run ceiling.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_engines import ENGINES, VISION_USD_PER_PAGE, image_size, load_engine, timed, write_page
from lib.ocr_pages import parse_pages
from lib.ocr_route import Budget
from lib.paths import OCR_COSTS, OCR_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ENGINE = None


def _init(name: str, opts: dict):
    """Worker-process initialiser: one engine per process, one Tesseract thread each."""
    global _ENGINE
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    _ENGINE = load_engine(name, **opts)


def _work(job: tuple) -> tuple:
    path, lang = job
    lines, secs = timed(_ENGINE.recognise, path, lang)
    return path, lines, secs


def load_pages(book_dir: str) -> dict:
    p = os.path.join(book_dir, "pages.json")
    if not os.path.exists(p):
        sys.exit("no %s; run 20_ocr_pages.py first" % p)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--lang", choices=["pa", "en", "hi"], help="default: the book's language")
    ap.add_argument("--pages", help="'1-20,35'; default every rendered page")
    ap.add_argument("--gt-pages", action="store_true",
                    help="only the pages in gt/lines.jsonl: a cheap benchmark of a paid engine")
    ap.add_argument("--routed", action="store_true",
                    help="only the pages 22_ocr_merge.py listed in merged/route.json (the arbiter's share)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="tesseract only; GPU engines run one at a time")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-pages", type=int, default=0, help="per-run ceiling (0 = none)")
    ap.add_argument("--budget-usd", type=float, default=5.0, help="monthly cap for paid engines")
    ap.add_argument("--dry-run", action="store_true", help="paid engines: print the cost, send nothing")
    ap.add_argument("--http", help="dotsocr: OpenAI-compatible endpoint instead of in-process")
    ap.add_argument("--model-path", help="dotsocr/indicocr: local weights or HF id")
    ap.add_argument("--psm", type=int, default=3, help="tesseract page segmentation mode")
    ap.add_argument("--tess-lang", help="tesseract: traineddata to use instead of the default for --lang, "
                    "e.g. pan or script/Gurmukhi; output goes to ocr/tesseract-<name>/")
    ap.add_argument("--out", default=OCR_DIR)
    args = ap.parse_args()

    book_dir = os.path.join(args.out, args.book)
    meta = load_pages(book_dir)
    lang = args.lang or meta.get("language") or "en"
    pages_dir = os.path.join(book_dir, "pages")
    engine_key = args.engine
    if args.engine == "tesseract" and args.tess_lang:
        engine_key = "tesseract-" + args.tess_lang.replace("script/", "").replace("+", "-").lower()
    if args.engine == "tesseract" and args.psm != 3:
        engine_key += "-psm%d" % args.psm                 # a notation book runs pan twice: psm 3 and 4 vote
    out_dir = os.path.join(book_dir, "ocr", engine_key)
    wanted = set(n + 1 for n in parse_pages(args.pages, max((p["page"] for p in meta["pages"]), default=0)))
    if args.gt_pages:
        # a benchmark run: only the pages the ground truth covers, so a paid
        # engine measured by 24_ocr_eval.py costs cents, not the book
        gt_path = os.path.join(book_dir, "gt", "lines.jsonl")
        if not os.path.exists(gt_path):
            sys.exit("--gt-pages needs %s (23_ocr_gt.py --sample, then --promote)" % gt_path)
        with open(gt_path, encoding="utf-8") as fh:
            gt_pages = {json.loads(line)["page"] for line in fh if line.strip()}
        wanted &= gt_pages
        print("ground-truth pages: %d" % len(wanted))
    if args.routed:
        route_path = os.path.join(book_dir, "merged", "route.json")
        if not os.path.exists(route_path):
            sys.exit("--routed needs %s (22_ocr_merge.py)" % route_path)
        with open(route_path, encoding="utf-8") as fh:
            routed = {p["page"] for p in json.load(fh).get("pages", [])}
        wanted &= routed
        print("routed pages: %d" % len(wanted))
    todo = []
    for rec in meta["pages"]:
        if rec["page"] not in wanted:
            continue
        png = os.path.join(pages_dir, rec["file"])
        dst = os.path.join(out_dir, "%04d.jsonl" % rec["page"])
        if not os.path.exists(png):
            continue
        if os.path.exists(dst) and not args.force:
            continue
        todo.append((rec["page"], png, dst))
    if args.max_pages:
        todo = todo[:args.max_pages]
    paid = ENGINES[args.engine].paid
    est = round(len(todo) * VISION_USD_PER_PAGE, 4) if args.engine == "vision" else 0.0
    print("%s / %s / %s: %d page(s) to do%s" % (args.book, args.engine, lang, len(todo),
                                                (", est. $%.4f" % est) if paid else ""))
    if not todo:
        return
    if args.dry_run:
        if paid:
            b = Budget(OCR_COSTS, args.budget_usd)
            print("dry run: spent this month $%.4f, cap $%.2f, this run would add $%.4f -> %s"
                  % (b.spent_this_month(), args.budget_usd, est, "OK" if b.allows(est) else "REFUSED"))
        else:
            print("dry run: nothing sent")
        return

    opts: dict = {}
    if args.engine == "tesseract":
        opts["psm"] = args.psm
        if args.tess_lang:
            opts["tess_lang"] = args.tess_lang
    if args.engine == "dotsocr":
        if args.http:
            opts["http"] = args.http
        if args.model_path:
            opts["model_path"] = args.model_path
    if args.engine == "indicocr" and args.model_path:
        opts["repo"] = args.model_path
    if args.engine == "pdftext":
        if not any(p.get("text_layer") for p in meta["pages"]):
            sys.exit("%s has no text layer to read (pages.json carries none)" % args.book)
        opts["pages"] = meta
    if paid:
        opts["budget"] = Budget(OCR_COSTS, args.budget_usd)
        opts["book"] = args.book

    t0 = time.time()
    done = 0
    total_lines = 0

    def finish(page: int, png: str, dst: str, lines: list, secs: float, engine_desc: str):
        nonlocal done, total_lines
        w, h = image_size(png)
        write_page(dst, {"engine": engine_key, "model": engine_desc, "lang": lang, "page": page,
                         "page_w": w, "page_h": h, "dpi": meta.get("dpi"), "seconds": secs}, lines)
        done += 1
        total_lines += len(lines)
        if done % 10 == 0 or done == len(todo):
            print("  %d/%d  (%.1fs/page)" % (done, len(todo), (time.time() - t0) / done))

    if args.engine == "tesseract" and args.workers > 1 and len(todo) > 1:
        desc = load_engine("tesseract", **opts).model_desc()
        by_png = {png: (page, dst) for page, png, dst in todo}
        with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                    initargs=(args.engine, opts)) as pool:
            for png, lines, secs in pool.map(_work, [(png, lang) for _, png, _ in todo]):
                page, dst = by_png[png]
                finish(page, png, dst, lines, secs, desc)
    else:
        engine = load_engine(args.engine, **opts)
        desc = engine.model_desc()
        if args.engine == "vision":
            batch = 16
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                results, secs = timed(engine.recognise_many, [png for _, png, _ in chunk], lang)
                for (page, png, dst), lines in zip(chunk, results):
                    finish(page, png, dst, lines, round(secs / len(chunk), 2), desc)
        else:
            for page, png, dst in todo:
                lines, secs = timed(engine.recognise, png, lang)
                finish(page, png, dst, lines, secs, desc)
    secs = time.time() - t0
    print("done: %d pages, %d lines, %.0fs (%.2fs/page) -> %s" % (done, total_lines, secs, secs / max(done, 1), out_dir))


if __name__ == "__main__":
    main()
