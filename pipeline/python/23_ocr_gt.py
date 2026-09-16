"""
Ground truth for a book: a stratified sample of lines, and the crops to check them against.

  23_ocr_gt.py --book santhya-vol-1 --sample 30 --seed 0 --draft-engine tesseract
  23_ocr_gt.py --book santhya-vol-1 --check
  23_ocr_gt.py --book santhya-vol-1 --promote

--sample writes data/ocr/<book>/gt/candidates.jsonl, one record per chosen
line with the engine's reading as a draft, a crop PNG under gt/crops/, and
gt/review.html that shows every crop beside its draft. A person (or the
assistant, reading the crops) edits "text" in candidates.jsonl and sets
"verified": true. Where the draft matched a corpus line the draft IS the corpus
text ("source": "corpus") and only the match has to be confirmed.

The sample is stratified so the measurement says something about each kind of
line -- Gurbani, commentary, footnotes, headings -- and each third of the page,
rather than thirty lines of easy body text. Quotas are minimums; the rest is
random under the seed, so a re-run with the same seed gives the same sample.

--check validates the edits (empty text, wrong script, characters normalise()
would change, so the typist sees what the comparison will fold).
--promote appends verified candidates to gt/lines.jsonl, replacing an earlier
record with the same id.
"""
from __future__ import annotations
import argparse
import glob
import html
import json
import os
import random
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_engines import read_page
from lib.ocr_match import CorpusIndex, match_text
from lib.ocr_text import GURMUKHI, LATIN, normalise, script_of, words
from lib.ocr_zones import classify_zones, footnote_rule_y, header_of, page_columns
from lib.paths import CORPUS_DB, OCR_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUOTAS = {"gurbani": 8, "commentary": 12, "footnote": 3, "heading": 2}
CROP_PAD = 10
# Only the double danda marks verse: the single danda (।) is the full stop of
# modern Punjabi prose and appears in most commentary lines.
VERSE_MARK = "॥‖"                  # ॥ ‖


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    out.append(json.loads(raw))
    return out


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def kind_of(line: dict, lang: str, match: dict | None) -> str:
    if match:
        return "gurbani"
    zone = line.get("zone")
    if zone == "footnote":
        return "footnote"
    text = line.get("text", "")
    n = len(words(text))
    if lang == "pa" and any(c in text for c in VERSE_MARK):
        return "gurbani"
    if n <= 4 and (lang != "pa" or GURMUKHI.search(text)):
        return "heading"
    return "commentary"


def page_lines(book_dir: str, engine: str, page_rec: dict, lang: str, index: CorpusIndex | None):
    """The draft engine's body lines on one page, classified, with hints and columns."""
    import cv2
    path = os.path.join(book_dir, "ocr", engine, "%04d.jsonl" % page_rec["page"])
    if not os.path.exists(path):
        return None
    meta, lines = read_page(path)
    img = cv2.imread(os.path.join(book_dir, "pages", page_rec["file"]), cv2.IMREAD_GRAYSCALE)
    rule = footnote_rule_y(img) if img is not None else None
    lines = classify_zones(lines, meta["page_w"], meta["page_h"], page_rec.get("stamps"), rule)
    hints = header_of(lines)
    wordboxes = [w for ln in lines for w in ln.get("words", [])] or lines
    cols = page_columns(wordboxes, meta["page_w"], img, lines)
    layout = "two-column" if cols else "single"
    out = []
    for ln in lines:
        if ln["zone"] not in ("body", "footnote") or not ln.get("text", "").strip():
            continue
        match = None
        if index is not None and lang == "pa" and GURMUKHI.search(ln["text"]):
            match = match_text(ln["text"], index, hints, source=page_rec.get("scripture", "G"))
        out.append({"page": page_rec["page"], "n": ln["n"], "bbox": ln["bbox"],
                    "kind": kind_of(ln, lang, match), "layout": layout, "zone": ln["zone"],
                    "draft": ln["text"], "match": match, "conf": ln.get("conf")})
    return out


def stratified(pool: list[dict], n: int, seed: int, page_h_of) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen, used = [], set()

    def third(rec):
        h = page_h_of(rec["page"]) or 1
        return min(2, int(3 * ((rec["bbox"][1] + rec["bbox"][3]) / 2.0) / h))

    def take(pred, k):
        count = 0
        thirds: Counter = Counter()
        for rec in pool:
            if count >= k:
                break
            key = (rec["page"], rec["n"])
            if key in used or not pred(rec):
                continue
            # spread over page thirds: do not take a fourth from one third while another has none
            t = third(rec)
            if thirds[t] >= 3 and min(thirds[x] for x in range(3)) == 0 and k >= 6:
                continue
            used.add(key)
            chosen.append(rec)
            thirds[t] += 1
            count += 1

    for kind, quota in QUOTAS.items():
        take(lambda r, kind=kind: r["kind"] == kind, quota)
    take(lambda r: True, max(0, n - len(chosen)))
    chosen.sort(key=lambda r: (r["page"], r["n"]))
    return chosen[:max(n, len(chosen))] if len(chosen) <= n else chosen[:n]


def crop(book_dir: str, page_rec: dict, bbox: list, dst: str):
    import cv2
    img = cv2.imread(os.path.join(book_dir, "pages", page_rec["file"]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - CROP_PAD), max(0, y0 - CROP_PAD)
    x1, y1 = min(w, x1 + CROP_PAD), min(h, y1 + CROP_PAD)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, img[y0:y1, x0:x1])


def review_html(rows: list[dict], path: str):
    parts = ["<!doctype html><meta charset='utf-8'><title>OCR ground truth</title>",
             "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}td{border:1px solid #ccc;"
             "padding:6px;vertical-align:top}img{max-width:900px}.pa{font-size:22px}</style>",
             "<p>Edit <code>candidates.jsonl</code>: fix <b>text</b>, set <b>verified</b> to true. "
             "A record with source=corpus shows the corpus line; confirm the crop shows that line.</p><table>"]
    for r in rows:
        cls = "pa" if GURMUKHI.search(r["text"]) else ""
        parts.append("<tr><td><code>%s</code><br>%s<br>%s<br>%s</td><td><img src='%s'><br><span class='%s'>%s</span></td></tr>"
                     % (html.escape(r["id"]), r["kind"], r["layout"], r["source"],
                        html.escape(r["crop"].replace("\\", "/")), cls, html.escape(r["text"])))
    parts.append("</table>")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts))


def do_sample(args, book_dir: str, meta: dict):
    lang = meta.get("language", "en")
    index = None
    corpus = args.corpus
    if lang == "pa" and corpus and os.path.exists(corpus) and os.path.getsize(corpus) > 0:
        con = sqlite3.connect(corpus)
        index = CorpusIndex.from_sqlite(con)
        con.close()
        print("corpus: %d lines for matching" % len(index))
    pages = meta["pages"]
    if args.pages:
        from lib.ocr_pages import parse_pages
        keep = set(i + 1 for i in parse_pages(args.pages, max(p["page"] for p in pages)))
        pages = [p for p in pages if p["page"] in keep]
    pool: list[dict] = []
    heights: dict = {}
    for rec in pages:
        rec = dict(rec, scripture=meta.get("scripture", "G"))
        got = page_lines(book_dir, args.draft_engine, rec, lang, index)
        if got is None:
            continue
        heights[rec["page"]] = rec.get("h")
        pool.extend(got)
    if not pool:
        sys.exit("no %s output under %s; run 21_ocr_run.py first" % (args.draft_engine, book_dir))
    kinds = Counter(r["kind"] for r in pool)
    print("pool: %d lines on %d pages: %s" % (len(pool), len(heights), dict(kinds)))
    chosen = stratified(pool, args.sample, args.seed, heights.get)
    by_page = {p["page"]: p for p in pages}
    rows = []
    for rec in chosen:
        rid = "%s:%d:%d" % (args.book, rec["page"], rec["n"])
        crop_rel = os.path.join("crops", "%04d-%03d.png" % (rec["page"], rec["n"]))
        crop(book_dir, by_page[rec["page"]], rec["bbox"], os.path.join(book_dir, "gt", crop_rel))
        m = rec["match"]
        rows.append({"id": rid, "page": rec["page"], "n": rec["n"], "bbox": rec["bbox"], "kind": rec["kind"],
                     "layout": rec["layout"], "text": m["text"] if m else rec["draft"],
                     "line_id": m["line_id"] if m else None, "match_score": m["score"] if m else None,
                     "source": "corpus" if m else "human", "verified": False,
                     "draft_engine": args.draft_engine, "draft": rec["draft"], "crop": crop_rel, "note": ""})
    gt_dir = os.path.join(book_dir, "gt")
    write_jsonl(os.path.join(gt_dir, "candidates.jsonl"), rows)
    review_html(rows, os.path.join(gt_dir, "review.html"))
    print("sampled %d: %s -> %s" % (len(rows), dict(Counter(r["kind"] for r in rows)),
                                    os.path.join(gt_dir, "candidates.jsonl")))


def do_check(book_dir: str, meta: dict) -> int:
    rows = read_jsonl(os.path.join(book_dir, "gt", "candidates.jsonl"))
    lang = meta.get("language", "en")
    problems = 0
    for r in rows:
        if not r.get("verified"):
            continue
        t = r.get("text", "")
        if not t.strip():
            print("EMPTY   %s" % r["id"])
            problems += 1
            continue
        s = script_of(t)
        # A Punjabi book carries English pages (the Vayakaran opens with an
        # English introduction) and an English one quotes Gurmukhi, so a
        # script mismatch is shown, not refused; kind "gurbani" must be Gurmukhi.
        if r["kind"] == "gurbani" and not GURMUKHI.search(t):
            print("SCRIPT  %s: a gurbani line must be Gurmukhi: %s" % (r["id"], t[:40]))
            problems += 1
        elif lang == "pa" and not GURMUKHI.search(t):
            print("script  %s: %s in a Punjabi book (kept): %s" % (r["id"], s, t[:40]))
        elif lang == "en" and not LATIN.search(t):
            print("script  %s: %s in an English book (kept): %s" % (r["id"], s, t[:40]))
        if normalise(t) != t:
            print("FOLDED  %s: normalise() changes this line (fine, just so you know)" % r["id"])
    verified = sum(1 for r in rows if r.get("verified"))
    print("%d candidates, %d verified, %d problems" % (len(rows), verified, problems))
    return problems


def do_promote(book_dir: str):
    cands = read_jsonl(os.path.join(book_dir, "gt", "candidates.jsonl"))
    path = os.path.join(book_dir, "gt", "lines.jsonl")
    lines = {r["id"]: r for r in read_jsonl(path)}
    n = 0
    for r in cands:
        if r.get("verified"):
            keep = {k: r[k] for k in ("id", "page", "n", "bbox", "kind", "layout", "text", "line_id",
                                       "source", "verified", "draft_engine", "note") if k in r}
            lines[r["id"]] = keep
            n += 1
    write_jsonl(path, [lines[k] for k in sorted(lines, key=lambda i: (int(i.split(":")[1]), int(i.split(":")[2])))])
    print("promoted %d; %d ground-truth lines in %s" % (n, len(lines), path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--sample", type=int, default=0, help="draw this many candidates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-engine", default="tesseract")
    ap.add_argument("--pages", help="restrict the sample to these pages, '1-20'")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--out", default=OCR_DIR)
    args = ap.parse_args()
    book_dir = os.path.join(args.out, args.book)
    meta = load_json(os.path.join(book_dir, "pages.json"))
    if args.sample:
        do_sample(args, book_dir, meta)
    if args.check:
        if do_check(book_dir, meta):
            sys.exit(1)
    if args.promote:
        if do_check(book_dir, meta):
            sys.exit("fix the problems above before promoting")
        do_promote(book_dir)
    if not (args.sample or args.check or args.promote):
        ap.print_help()


if __name__ == "__main__":
    main()
