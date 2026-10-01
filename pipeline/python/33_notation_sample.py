"""
A sample across the library: the notations at two places in every book, on one review page.

  33_notation_sample.py --library ~/Documents/Keertan                       # every book, middle and end
  33_notation_sample.py --library ~/Documents/Keertan --places random --books 20 --per-book 2 --seed 1
  33_notation_sample.py --library ~/Documents/Keertan --skip-ocr            # the page only, from what was parsed

The per-book review (30_notation_gt.py --mid) proves a book before it runs
whole; this asks a different question -- does the reader hold up across
the shelf? -- by reading a short window at the middle of every book and
one near its end (a book that keeps its notations late, or has none, shows
itself there), running the OCR route on each window (pages, three
Tesseract passes, the merge, 29_notation_parse.py) and putting what came
out side by side: the notation as printed, the original page, the crops
the parser cut, and the grid it read. Every book gets a manifest under the
keertan folder (kind: notation, the defaults for style) if it has none, so
a book that reads well here is one command from its own review. The page
is rewritten after every book, so a long run can be read as it goes.

Writes data/ocr/_sample/review.html, sample.json (what was picked and how
each book fared) and candidates.jsonl (the same judgement records the
per-book page uses, one per shown notation).
"""
from __future__ import annotations
import argparse
import datetime as dt
import html as html_mod
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.notation import read_jsonl
from lib.paths import NOTATIONS_DIR, OCR_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
KEERTAN_DIR = os.environ.get("KEERTAN_DIR") or os.path.join(os.path.dirname(NOTATIONS_DIR.rstrip("/\\")), "keertan")


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "book"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pdf_pages(path: str) -> int:
    try:
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


def library_books(library: str) -> list[dict]:
    """Every readable PDF under the library: {path, author, title, book, pages}."""
    out = []
    for root, _dirs, files in os.walk(library):
        for f in sorted(files):
            if not f.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(root, library)
            author = rel.split(os.sep)[0] if rel not in (".", "") else "Misc"
            n = pdf_pages(path)
            if n < 20:
                continue
            stem = os.path.splitext(f)[0]
            out.append({"path": path, "author": author, "author_slug": slug(author), "title": stem.replace("_", " ").replace("-", " ").strip(),
                        "book": slug(stem), "pages": n})
    return out


def folder_with(keertan_dir: str, filename: str) -> str | None:
    """A folder already declaring this PDF (a pilot's, with its style tuned): reuse it."""
    for name in sorted(os.listdir(keertan_dir)) if os.path.isdir(keertan_dir) else []:
        mpath = os.path.join(keertan_dir, name, "manifest.json")
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            continue
        if any(w.get("file") == filename for w in manifest.get("works", [])):
            return os.path.join(keertan_dir, name)
    return None


def book_key_in(folder: str, filename: str) -> str | None:
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    for w in manifest.get("works", []):
        if w.get("file") == filename:
            return w.get("book") or slug(os.path.splitext(filename)[0])
    return None


def ensure_manifest(folder: str, author: str, works: list[dict]) -> None:
    """A manifest in the keertan folder naming every PDF of the author as a notation book, with symlinks to the PDFs."""
    os.makedirs(folder, exist_ok=True)
    mpath = os.path.join(folder, "manifest.json")
    manifest = {"author": author, "language": "pa", "reader": "ocr", "licence": "copyright", "kind": "notation",
                "style": {"table": "bars", "shabad_position": "before"}, "works": []}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest.setdefault("kind", "notation")
    have = {w.get("file") for w in manifest.get("works", [])}
    for w in works:
        name = os.path.basename(w["path"])
        link = os.path.join(folder, name)
        if not os.path.exists(link):
            try:
                os.symlink(w["path"], link)
            except OSError:
                import shutil
                shutil.copyfile(w["path"], link)
        if name not in have:
            manifest["works"].append({"file": name, "work": w["book"], "title": w["title"], "book": w["book"]})
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)


PLACES = {"middle": 0.45, "end": 0.82, "start": 0.15}
LOOKAHEAD = 2               # 27_ingest_book.NOTATION_LOOKAHEAD: pages read past each window


def windows_for(n_pages: int, k: int, width: int, rng: random.Random, places: list[str] | None = None) -> list[tuple[int, int]]:
    """
    `width` pages at each place: the named places (middle at 45% of the
    book, end at 82% -- past the last shabads of a book that puts its
    notations late, before its index) or, with no places, k random ones
    between 12% and 88%. A place never runs past the last page.
    """
    if places:
        starts = sorted({max(1, min(n_pages - width + 1, int(round(PLACES[p] * n_pages)))) for p in places})
    else:
        lo, hi = max(1, int(0.12 * n_pages)), max(1, int(0.88 * n_pages) - width)
        starts = sorted(set(rng.randint(lo, max(lo, hi)) for _ in range(k)))
    return [(a, min(n_pages, a + width - 1)) for a in starts]


def run_book(folder: str, book: str, pages: str, engines: str | None, log) -> int:
    cmd = [sys.executable, os.path.join(HERE, "27_ingest_book.py"), "--src", folder, "--book", book, "--pages", pages, "--to", "notation"]
    if engines:
        cmd += ["--engines", engines]
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def page_files(book: str) -> dict[int, str]:
    p = os.path.join(OCR_DIR, book, "pages.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {r["page"]: r["file"] for r in json.load(fh).get("pages", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="the folder of keertan books (one subfolder an author)")
    ap.add_argument("--books", type=int, default=0, help="how many books (0: every readable one)")
    ap.add_argument("--per-book", type=int, default=2, help="random places a book (with --places random)")
    ap.add_argument("--places", default="middle,end", help="'middle,end' (the default), 'start,middle,end', or 'random'")
    ap.add_argument("--show", type=int, default=6, help="notations shown a book at most")
    ap.add_argument("--window", type=int, default=4, help="pages read at each place")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engines", help="comma list for 21_ocr_run.py (default: the notation set)")
    ap.add_argument("--keertan-dir", default=KEERTAN_DIR, help="where the manifests and links live")
    ap.add_argument("--out", default=os.path.join(OCR_DIR, "_sample"))
    ap.add_argument("--skip-ocr", action="store_true", help="only write the page from what is already parsed")
    ap.add_argument("--only", help="comma list of book keys to restrict to")
    ap.add_argument("--judge", action="store_true", help="the per-field right/wrong controls instead of one comment box a notation")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    books = library_books(os.path.expanduser(args.library))
    if not books:
        sys.exit("no readable PDFs under %s" % args.library)
    by_author: dict[str, list[dict]] = {}
    for b in books:
        by_author.setdefault(b["author"], []).append(b)
    picked = list(books)
    rng.shuffle(picked)
    if args.only:
        keep = set(args.only.split(","))
        picked = [b for b in picked if b["book"] in keep]
    # one book an author first, so the sample covers the shelf, then the rest
    seen_authors: set[str] = set()
    first, rest = [], []
    for b in picked:
        (rest if b["author"] in seen_authors else first).append(b)
        seen_authors.add(b["author"])
    picked = (first + rest)[: args.books] if args.books else (first + rest)
    places = None if args.places == "random" else [p.strip() for p in args.places.split(",") if p.strip()]
    if places and any(p not in PLACES for p in places):
        sys.exit("--places: one of %s, or random" % ", ".join(PLACES))

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "run.log"), "a", encoding="utf-8")
    log.write("\n==== sample %s seed %d\n" % (dt.datetime.now().isoformat(timespec="seconds"), args.seed))
    plan = []
    for b in picked:
        existing = folder_with(args.keertan_dir, os.path.basename(b["path"]))
        if existing:
            folder = existing
            b["book"] = book_key_in(folder, os.path.basename(b["path"])) or b["book"]
        else:
            folder = os.path.join(args.keertan_dir, b["author_slug"])
            ensure_manifest(folder, b["author"], by_author[b["author"]])
        wins = windows_for(b["pages"], args.per_book, args.window, rng, places)
        plan.append({**b, "folder": folder, "windows": wins, "pages_arg": ",".join("%d-%d" % w for w in wins),
                     "read": ",".join("%d-%d" % (a, c + LOOKAHEAD) for a, c in wins)})
    print("%d book(s), %s, %d pages a place" % (len(plan), (", ".join(places) if places else "%d random place(s)" % args.per_book), args.window))

    results = []
    for k, b in enumerate(plan):
        t0 = time.time()
        status = "skipped"
        if not args.skip_ocr:
            code = run_book(b["folder"], b["book"], b["pages_arg"], args.engines, log)
            status = "ok" if code == 0 else "failed (%d)" % code
        recs = []
        path = os.path.join(NOTATIONS_DIR, b["book"], "notations.jsonl")
        if os.path.exists(path):
            _, all_recs = read_jsonl(path)
            wanted = {p for a, c in b["windows"] for p in range(a, c + 1)}
            recs = [r for r in all_recs if r["pages"][0] in wanted]       # the notations that begin in a window
        # every notation the windows hold, the resolved ones first, up to --show
        recs.sort(key=lambda r: (r["shabad"].get("shabad_id") is None, not r.get("sections"), r["pages"][0]))
        chosen = recs[: args.show]
        results.append({**b, "status": status, "found": len(recs), "shown": [r["notation_id"] for r in chosen],
                        "seconds": round(time.time() - t0, 1), "records": chosen})
        print("  [%2d/%d] %-40s %-10s %d notation(s) in %s, %d shown, %.0fs"
              % (k + 1, len(plan), b["book"][:40], status, len(recs), b["pages_arg"], len(chosen), time.time() - t0))
        log.flush()
        write_page(args, results)          # the page grows as the run goes, so it can be watched
    log.close()
    print("-> %s" % os.path.join(args.out, "review.html"))
    print("   to tick and comment with the verdicts saved: 34_notation_review.py serve --sample %s" % os.path.basename(args.out.rstrip("/")))


def write_page(args, results: list[dict]) -> None:
    gt = load_script("30_notation_gt.py")
    cands = []
    parts = []
    rows_html = []
    for b in results:
        recs = b["records"]
        cells = sum(r["quality"]["cells"] for r in recs)
        unknown = sum(r["quality"]["unknown"] for r in recs)
        resolved = sum(1 for r in recs if r["shabad"].get("shabad_id") is not None)
        with_grid = sum(1 for r in recs if r.get("sections"))
        multi = sum(1 for r in recs if len(r["pages"]) > 1)
        verdict = ("no shabad found" if not recs else
                   "shabads" if resolved == len(recs) else
                   "partly" if resolved else "unnamed")
        status = b["status"] if b["status"] not in ("ok", "skipped") else ""
        rows_html.append("<tr><td><a href='#b-%s'>%s</a></td><td>%s</td><td>%d</td><td>%s</td><td>%d</td><td>%d / %d</td><td>%d</td><td>%d / %d</td><td>%s</td><td class='v-%s'>%s%s</td></tr>"
                         % (html_mod.escape(b["book"]), html_mod.escape(b["title"]), html_mod.escape(b["author"]), b["pages"], html_mod.escape(b["pages_arg"]),
                            b["found"], resolved, len(recs), multi, with_grid, len(recs),
                            ("%d%%" % round(100.0 * unknown / cells)) if cells else "-", verdict.split()[0], verdict,
                            (" · run " + html_mod.escape(status)) if status else ""))
    parts.append("<table class='sum'><thead><tr><th>book</th><th>author</th><th>pages</th><th>read at</th><th>found</th>"
                 "<th>shabad named</th><th>over pages</th><th>grid read</th><th>unread cells</th><th>verdict</th></tr></thead><tbody>%s</tbody></table>"
                 "<p class='legend'>The verdict is mechanical and about the cut, not the grid: <b>shabads</b> = every notation found in the windows is linked to a shabad "
                 "the corpus knows; <b>partly</b> = some are; <b>unnamed</b> = notations were cut but none linked (a shabad the corpus lacks, or its text unread); "
                 "<b>no shabad found</b> = the windows held no shabad (prose, exercises, a picture), or the book is not a notation book. "
                 "<b>over pages</b> counts the notations that run over a page turn. The grid columns are the parked machine reading. "
                 "Judge each notation below against its page; the original page is the authority.</p>" % "".join(rows_html))
    for b in results:
        parts.append('<h2 class="book" id="b-%s">%s <small>%s · %d pages · read at %s · %s · %d found</small></h2>'
                     % (html_mod.escape(b["book"]), html_mod.escape(b["title"]), html_mod.escape(b["author"]), b["pages"], html_mod.escape(b["pages_arg"]),
                        html_mod.escape(b["status"]), b["found"]))
        if not b["records"]:
            parts.append('<p class="none">nothing parsed here%s</p>' % (" -- see run.log" if b["status"].startswith("failed") else ""))
            continue
        files = page_files(b["book"])
        img_base = os.path.relpath(os.path.join(NOTATIONS_DIR, b["book"]), args.out).replace("\\", "/")
        pages_base = os.path.relpath(os.path.join(OCR_DIR, b["book"], "pages"), args.out).replace("\\", "/")
        shown_pages = {p for r in b["records"] for p in r["pages"]}
        parts.append(gt.raag_notes_html(gt.load_raag_notes(os.path.join(NOTATIONS_DIR, b["book"])), img_base, shown_pages))
        from lib.notation_review import assign_keys
        assign_keys(b["records"])
        for rec in b["records"]:
            cand = gt.candidate(rec)
            cand["book"] = b["book"]
            cands.append(cand)
            parts.append(gt.card(rec, cand, img_base, {}, pages_base, files, comments=not args.judge))
    data = json.dumps({"book": os.path.basename(args.out.rstrip("/")) or "_sample", "reviewer": os.environ.get("USER") or "",
                       "mode": "judge" if args.judge else "comments", "candidates": cands}, ensure_ascii=False).replace("</", "<\\/")
    extra_css = """
.book{margin:26px 16px 6px;font-size:18px}.book small{color:#666;font-weight:normal;font-size:13px}
.none{margin:0 16px;color:#a33}
.orig{margin:0 16px;display:flex;gap:12px;flex-wrap:wrap}
.orig figure{margin:0}.orig figcaption{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.04em}
.orig img{width:260px;height:auto;border:1px solid #ccc;background:#fff;cursor:zoom-in}
.orig img.zoom{width:auto;max-width:100%;cursor:zoom-out}
table.sum{margin:14px 16px;border-collapse:collapse;font-size:13px;background:#fff}
table.sum th,table.sum td{border:1px solid #ddd;padding:4px 8px;text-align:left}
table.sum th{background:#f0efe9}
td.v-shabads{background:#e3f4e6}td.v-partly{background:#fff4d6}td.v-unnamed{background:#fde2e2}td.v-no{background:#eee}
.legend{margin:0 16px 8px;color:#555;font-size:13px}
"""
    extra_js = """
document.addEventListener('click', e => { if (e.target.tagName === 'IMG' && e.target.closest('.orig')) e.target.classList.toggle('zoom'); });
"""
    n_books = len(results)
    n_shown = sum(len(b["records"]) for b in results)
    doc = ("<!doctype html><html lang=en><meta charset=utf-8><title>notation sample · %d books</title>"
           "<meta name=viewport content='width=device-width,initial-scale=1'>"
           "<style>%s%s%s</style><body>"
           "<header><h1>A sample across the shelf <small>%d books · %d notations · %s · each notation as printed, its page, the finer cuts</small></h1>"
           "<span id=count class=count></span><button id=download class=primary>download %s</button><button id=reset>reset</button></header>"
           "%s<script id=cands type=application/json>%s</script><script>%s%s</script></html>"
           % (n_books, gt.PAGE_CSS, gt.NOTATION_CSS, extra_css, n_books, n_shown,
              ("%d pages at the %s of every book" % (args.window, " and the ".join(args.places.split(","))) if args.places != "random"
               else "%d random places a book, seed %d" % (args.per_book, args.seed)),
              "candidates.jsonl" if args.judge else "comments.jsonl",
              "".join(parts), data, gt.PAGE_JS, extra_js))
    with open(os.path.join(args.out, "review.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    with open(os.path.join(args.out, "candidates.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "sample.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump([{k: v for k, v in b.items() if k != "records"} for b in results], fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
