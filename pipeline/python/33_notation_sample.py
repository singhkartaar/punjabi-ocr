"""
A sample across the library: a few notations from each of many books, on one review page.

  33_notation_sample.py --library ~/Documents/Keertan --books 20 --per-book 2 --seed 1
  33_notation_sample.py --library ~/Documents/Keertan --books 20 --per-book 2 --seed 1 --skip-ocr   # page only

The per-book review (30_notation_gt.py --mid) proves a book before it runs
whole; this asks a different question -- does the reader hold up across
the shelf? -- by picking books at random, a couple of random places in
each, running the OCR route on a short window there (pages, three
Tesseract passes, the merge, 29_notation_parse.py) and putting what came
out side by side: the original page, the crops the parser cut, and the
grid it read, rendered in Gurmukhi and in English. Every book gets a
manifest under the keertan folder (kind: notation, the defaults for style)
if it has none, so a book that reads well here is one command from its
own review.

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


def windows_for(n_pages: int, k: int, width: int, rng: random.Random) -> list[tuple[int, int]]:
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
    ap.add_argument("--books", type=int, default=20)
    ap.add_argument("--per-book", type=int, default=2)
    ap.add_argument("--window", type=int, default=3, help="pages read at each random place")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engines", help="comma list for 21_ocr_run.py (default: the notation set)")
    ap.add_argument("--keertan-dir", default=KEERTAN_DIR, help="where the manifests and links live")
    ap.add_argument("--out", default=os.path.join(OCR_DIR, "_sample"))
    ap.add_argument("--skip-ocr", action="store_true", help="only write the page from what is already parsed")
    ap.add_argument("--only", help="comma list of book keys to restrict to")
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
    picked = (first + rest)[: args.books]

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
        wins = windows_for(b["pages"], args.per_book, args.window, rng)
        plan.append({**b, "folder": folder, "windows": wins, "pages_arg": ",".join("%d-%d" % w for w in wins)})
    print("%d book(s), %d place(s) each, %d pages a place" % (len(plan), args.per_book, args.window))

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
            recs = [r for r in all_recs if any(p in wanted for p in r["pages"])]
        rng.shuffle(recs)
        recs.sort(key=lambda r: (r["shabad"].get("shabad_id") is None, not r.get("sections")))
        chosen = recs[: args.per_book]
        results.append({**b, "status": status, "found": len(recs), "shown": [r["notation_id"] for r in chosen],
                        "seconds": round(time.time() - t0, 1), "records": chosen})
        print("  [%2d/%d] %-40s %-10s %d notation(s) in %s, %d shown, %.0fs"
              % (k + 1, len(plan), b["book"][:40], status, len(recs), b["pages_arg"], len(chosen), time.time() - t0))
        log.flush()
        write_page(args, results)          # the page grows as the run goes, so it can be watched
    log.close()
    print("-> %s" % os.path.join(args.out, "review.html"))


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
        verdict = ("no notation found" if not recs else
                   "reads" if resolved and with_grid and cells and unknown / cells <= 0.15 else
                   "partly" if with_grid or resolved else "not a notation page")
        rows_html.append("<tr><td><a href='#b-%s'>%s</a></td><td>%s</td><td>%d</td><td>%s</td><td>%d</td><td>%d / %d</td><td>%d / %d</td><td>%s</td><td class='v-%s'>%s</td></tr>"
                         % (html_mod.escape(b["book"]), html_mod.escape(b["title"]), html_mod.escape(b["author"]), b["pages"], html_mod.escape(b["pages_arg"]),
                            b["found"], resolved, len(recs), with_grid, len(recs),
                            ("%d%%" % round(100.0 * unknown / cells)) if cells else "-", verdict.split()[0], verdict))
    parts.append("<table class='sum'><thead><tr><th>book</th><th>author</th><th>pages</th><th>read at</th><th>found</th>"
                 "<th>shabad named</th><th>grid read</th><th>unread cells</th><th>verdict</th></tr></thead><tbody>%s</tbody></table>"
                 "<p class='legend'>The verdict is mechanical: <b>reads</b> = a shabad named and a grid read with at most 15%% of its cells unread; "
                 "<b>partly</b> = one of the two; <b>not a notation page</b> = the window fell on prose, a picture or a table of another kind. "
                 "Judge each notation below against its page; the original page is the authority.</p>" % "".join(rows_html))
    for b in results:
        parts.append('<h2 class="book" id="b-%s">%s <small>%s · %d pages · read at %s · %s · %d found</small></h2>'
                     % (html_mod.escape(b["book"]), html_mod.escape(b["title"]), html_mod.escape(b["author"]), b["pages"], html_mod.escape(b["pages_arg"]),
                        html_mod.escape(b["status"]), b["found"]))
        if not b["records"]:
            parts.append('<p class="none">nothing parsed here%s</p>' % (" -- see run.log" if b["status"].startswith("failed") else ""))
            continue
        files = page_files(b["book"])
        for rec in b["records"]:
            cand = gt.candidate(rec)
            cand["book"] = b["book"]
            cands.append(cand)
            img_base = os.path.relpath(os.path.join(NOTATIONS_DIR, b["book"]), args.out).replace("\\", "/")
            pages_html = "".join('<figure><figcaption>original page %d</figcaption><img src="%s" loading="lazy"></figure>'
                                 % (p, html_mod.escape(os.path.relpath(os.path.join(OCR_DIR, b["book"], "pages", files.get(p, "")), args.out).replace("\\", "/")))
                                 for p in rec["pages"] if files.get(p))
            parts.append('<div class="orig">%s</div>' % pages_html)
            parts.append(gt.card(rec, cand, img_base, {}))
    data = json.dumps({"book": "_sample", "reviewer": os.environ.get("USER") or "", "candidates": cands}, ensure_ascii=False).replace("</", "<\\/")
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
td.v-reads{background:#e3f4e6}td.v-partly{background:#fff4d6}td.v-not{background:#fde2e2}td.v-no{background:#eee}
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
           "<header><h1>A sample across the shelf <small>%d books · %d notations · seed %d · the original page, the crops, and the grid as read</small></h1>"
           "<span id=count class=count></span><button id=download class=primary>download candidates.jsonl</button><button id=reset>reset</button></header>"
           "%s<script id=cands type=application/json>%s</script><script>%s%s</script></html>"
           % (n_books, gt.PAGE_CSS, gt.NOTATION_CSS, extra_css, n_books, n_shown, args.seed, "".join(parts), data, gt.PAGE_JS, extra_js))
    with open(os.path.join(args.out, "review.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    with open(os.path.join(args.out, "candidates.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for c in cands:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "sample.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump([{k: v for k, v in b.items() if k != "records"} for b in results], fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
