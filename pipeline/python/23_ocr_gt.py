"""
Ground truth for a book: a stratified sample of lines, and the crops to check them against.

  23_ocr_gt.py --book santhya-vol-1 --sample 30 --seed 0 --draft-engine tesseract
  23_ocr_gt.py --book santhya-vol-1 --coverage-sample 30 --merged-name merged-cov
  23_ocr_gt.py --book santhya-vol-1 --count-pages 10
  23_ocr_gt.py --book santhya-vol-1 --check
  23_ocr_gt.py --book santhya-vol-1 --promote

--sample writes data/ocr/<book>/gt/candidates.jsonl, one record per chosen
line with the engine's reading as a draft, a crop PNG under gt/crops/, and
gt/review.html that shows every crop beside its draft. A person (or the
assistant, reading the crops) edits "text" in candidates.jsonl and sets
"verified": true. Where the draft matched a corpus line the draft IS the corpus
text ("source": "corpus") and only the match has to be confirmed.

That sample is drawn from lines the engine FOUND, so it can never hold a
line the engine skipped, and a merge that misses lines scores as well as one
that does not. Two samples see what --sample cannot:

--coverage-sample draws from what the merge's coverage pass did with the ink
its lines left uncovered (22_ocr_merge.py --coverage): lines it recovered,
with their reading as the draft, and regions it refused, with an empty draft.
The reviewer types what the crop says, or sets "not_text": true where it is a
rule, an ornament or noise. 24_ocr_eval.py then scores recovered lines apart
and counts a recovery on a not-text crop as false.

--count-pages draws whole pages, half of them two-column, and writes a
half-size PNG of each; the reviewer counts the printed body and footnote
lines, per column on a two-column page, into gt/page-counts.candidates.jsonl.
That count is the one measure of lines missed altogether, by any engine.

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
from lib.ocr_zones import classify_zones, find_vertical_rule, footnote_rule_y, header_of, page_columns
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
    rule_x = find_vertical_rule(img, meta["page_w"]) if img is not None else None
    cols = page_columns(wordboxes, meta["page_w"], img, lines, rule_x)     # for the layout label only: no split here
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
        origin = ("<br>coverage: %s%s" % (r["status"], (" (%s)" % r["why"]) if r.get("why") else "")
                  if r.get("origin") == "coverage" else "")
        parts.append("<tr><td><code>%s</code><br>%s<br>%s<br>%s%s</td><td><img src='%s'><br><span class='%s'>%s</span></td></tr>"
                     % (html.escape(r["id"]), r["kind"], r["layout"], r["source"], origin,
                        html.escape(r["crop"].replace("\\", "/")), cls, html.escape(r["text"])))
    parts.append("</table>")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts))


def draft_engine(book_dir: str, wanted: str) -> str:
    """
    The engine whose reading is the draft: the one asked for when it has
    output, else the first Tesseract variant that has (a Punjabi book has
    tesseract-pan and tesseract-gurmukhi, never plain tesseract, and
    29_bench_books.py --gt asked for plain tesseract and drew nothing).
    """
    ocr = os.path.join(book_dir, "ocr")
    if os.path.isdir(os.path.join(ocr, wanted)) or not os.path.isdir(ocr):
        return wanted
    have = sorted(d for d in os.listdir(ocr) if d.startswith("tesseract"))
    pick = next((d for d in ("tesseract-pan", "tesseract") if d in have), have[0] if have else wanted)
    if pick != wanted:
        print("no %s output; drafting from %s" % (wanted, pick))
    return pick


def do_sample(args, book_dir: str, meta: dict):
    args.draft_engine = draft_engine(book_dir, args.draft_engine)
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


def do_coverage_sample(args, book_dir: str, meta: dict):
    """
    Candidates from the coverage pass of one merge: recovered lines (their
    reading as the draft) and refused regions (no draft), stratified by
    status and column, each with a crop.
    """
    merged = os.path.join(book_dir, args.merged_name)
    if not os.path.isdir(merged):
        sys.exit("no %s; run 22_ocr_merge.py --coverage%s first"
                 % (merged, "" if args.merged_name == "merged" else " --out-name " + args.merged_name))
    by_page = {p["page"]: p for p in meta["pages"]}
    pool: list[dict] = []
    for path in sorted(glob.glob(os.path.join(merged, "*.jsonl"))):
        pmeta, lines = read_page(path)
        cov = pmeta.get("coverage") or {}
        if not cov.get("on"):
            continue
        page = pmeta["page"]
        for k, region in enumerate(cov.get("regions") or []):
            if region["status"] == "recovered":
                read = [ln for ln in lines if (ln.get("recovered") or {}).get("region") == k]
                draft = " ".join(ln["text"] for ln in sorted(read, key=lambda l: l["bbox"][0]))
                kind = next((ln.get("kind") for ln in read if ln.get("kind")), "commentary")
            else:
                draft, kind = "", "commentary"
            pool.append({"page": page, "bbox": region["bbox"], "col": region.get("col", 0), "status": region["status"],
                         "why": region.get("why"), "draft": draft, "kind": kind if kind != "gurbani-unmatched" else "gurbani",
                         "layout": "two-column" if pmeta.get("columns") else "single"})
    if not pool:
        sys.exit("no coverage regions in %s: was the merge run with --coverage?" % merged)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen: list[dict] = []
    # half recovered, half refused, each half spread over the columns, so the
    # sample says how good the recovered lines are AND what the refusals hide
    for status in ("recovered", "rejected"):
        rows = [r for r in pool if r["status"] == status]
        want = args.coverage_sample // 2 if status == "recovered" else args.coverage_sample - len(chosen)
        cols = sorted({r["col"] for r in rows})
        taken: list[dict] = []
        while rows and len(taken) < want:
            for col in cols:
                nxt = next((r for r in rows if r["col"] == col), None)
                if nxt is not None:
                    rows.remove(nxt)
                    taken.append(nxt)
                if len(taken) >= want:
                    break
            if not any(r["col"] in cols for r in rows):
                break
        chosen.extend(taken)
    chosen.sort(key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))
    out = []
    for rec in chosen:
        x0, y0 = rec["bbox"][0], rec["bbox"][1]
        rid = "%s:%d:r%d-%d" % (args.book, rec["page"], y0, x0)
        crop_rel = os.path.join("crops", "%04d-r%d-%d.png" % (rec["page"], y0, x0))
        crop(book_dir, by_page[rec["page"]], rec["bbox"], os.path.join(book_dir, "gt", crop_rel))
        out.append({"id": rid, "page": rec["page"], "bbox": rec["bbox"], "kind": rec["kind"], "layout": rec["layout"],
                    "text": rec["draft"], "line_id": None, "source": "human", "verified": False,
                    "origin": "coverage", "status": rec["status"], "why": rec["why"], "not_text": False,
                    "draft_engine": args.merged_name, "draft": rec["draft"], "crop": crop_rel, "note": ""})
    gt_dir = os.path.join(book_dir, "gt")
    path = os.path.join(gt_dir, "candidates.jsonl")
    existing = [r for r in read_jsonl(path) if r.get("origin") != "coverage"]
    write_jsonl(path, existing + out)
    review_html(existing + out, os.path.join(gt_dir, "review.html"))
    print("sampled %d coverage regions (%s) -> %s" % (len(out), dict(Counter(r["status"] for r in out)), path))


def do_count_pages(args, book_dir: str, meta: dict):
    """Whole pages for a line count: half two-column where the merge saw any, with a half-size PNG each."""
    import cv2
    merged = os.path.join(book_dir, args.merged_name)
    layout: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(merged, "*.jsonl"))):
        pmeta, lines = read_page(path)
        body = [ln for ln in lines if ln.get("zone") in ("body", "footnote")
                and (ln.get("text", "").strip() or ln.get("merged_into"))]
        cols = pmeta.get("columns") or []
        by_col = [sum(1 for ln in body if ln.get("col") == k) for k in range(len(cols))] if cols else None
        layout[pmeta["page"]] = {"columns": len(cols), "body": len(body), "by_col": by_col,
                                 "regions": len((pmeta.get("coverage") or {}).get("regions") or [])}
    if not layout:
        sys.exit("no %s; run 22_ocr_merge.py first" % merged)
    rng = random.Random(args.seed)
    two = [p for p, d in layout.items() if d["columns"]]
    one = [p for p, d in layout.items() if not d["columns"]]
    rng.shuffle(two)
    rng.shuffle(one)
    half = min(len(two), args.count_pages // 2)
    pages = sorted(two[:half] + one[:args.count_pages - half])
    by_page = {p["page"]: p for p in meta["pages"]}
    rows = []
    for page in pages:
        img = cv2.imread(os.path.join(book_dir, "pages", by_page[page]["file"]), cv2.IMREAD_GRAYSCALE)
        rel = os.path.join("crops", "page-%04d.png" % page)
        os.makedirs(os.path.join(book_dir, "gt", "crops"), exist_ok=True)
        cv2.imwrite(os.path.join(book_dir, "gt", rel), cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        d = layout[page]
        rows.append({"page": page, "layout": "two-column" if d["columns"] else "single", "columns": d["columns"],
                     "draft": {"body": d["body"], "by_col": d["by_col"]}, "body_lines": None, "by_col": None,
                     "verified": False, "image": rel, "note": ""})
    path = os.path.join(book_dir, "gt", "page-counts.candidates.jsonl")
    write_jsonl(path, rows)
    print("%d pages to count (%d two-column) -> %s; fill body_lines (and by_col on a two-column page), set verified"
          % (len(rows), half, path))


def do_check(book_dir: str, meta: dict) -> int:
    rows = read_jsonl(os.path.join(book_dir, "gt", "candidates.jsonl"))
    lang = meta.get("language", "en")
    problems = 0
    for r in rows:
        if not r.get("verified"):
            continue
        t = r.get("text", "")
        if r.get("not_text"):
            # a crop the reviewer saw no text in: an empty text is the point
            if t.strip():
                print("NOTTEXT %s: not_text is set but text is not empty" % r["id"])
                problems += 1
            continue
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
                                       "source", "verified", "draft_engine", "note",
                                       "origin", "status", "why", "not_text") if k in r}
            lines[r["id"]] = keep
            n += 1
    # by place on the page, not by the id: a coverage row has no line number
    write_jsonl(path, sorted(lines.values(), key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0])))
    print("promoted %d; %d ground-truth lines in %s" % (n, len(lines), path))
    counts_path = os.path.join(book_dir, "gt", "page-counts.candidates.jsonl")
    counted = [r for r in read_jsonl(counts_path) if r.get("verified") and r.get("body_lines") is not None]
    if counted:
        dst = os.path.join(book_dir, "gt", "page-counts.jsonl")
        have = {r["page"]: r for r in read_jsonl(dst)}
        for r in counted:
            have[r["page"]] = {k: r[k] for k in ("page", "layout", "columns", "body_lines", "by_col", "verified", "note") if k in r}
        write_jsonl(dst, [have[p] for p in sorted(have)])
        print("promoted %d page counts; %d in %s" % (len(counted), len(have), dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--sample", type=int, default=0, help="draw this many candidates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-engine", default="tesseract")
    ap.add_argument("--pages", help="restrict the sample to these pages, '1-20'")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--coverage-sample", type=int, default=0,
                    help="draw this many regions of a merge's coverage pass (recovered lines and refusals)")
    ap.add_argument("--count-pages", type=int, default=0, help="draw this many whole pages for a line count")
    ap.add_argument("--merged-name", default="merged", help="which merge the coverage sample or page count reads")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--out", default=OCR_DIR)
    args = ap.parse_args()
    book_dir = os.path.join(args.out, args.book)
    meta = load_json(os.path.join(book_dir, "pages.json"))
    if args.sample:
        do_sample(args, book_dir, meta)
    if args.coverage_sample:
        do_coverage_sample(args, book_dir, meta)
    if args.count_pages:
        do_count_pages(args, book_dir, meta)
    if args.check:
        if do_check(book_dir, meta):
            sys.exit(1)
    if args.promote:
        if do_check(book_dir, meta):
            sys.exit("fix the problems above before promoting")
        do_promote(book_dir)
    if not (args.sample or args.coverage_sample or args.count_pages or args.check or args.promote):
        ap.print_help()


if __name__ == "__main__":
    main()
