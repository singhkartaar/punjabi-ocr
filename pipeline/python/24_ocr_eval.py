"""
How well each engine reads, measured against the ground truth.

  24_ocr_eval.py --book santhya-vol-1                     every engine with output, vs gt/lines.jsonl
  24_ocr_eval.py --bakeoff --books santhya-vol-1,ten-masters --engines tesseract,surya,indicocr
  24_ocr_eval.py --bakeoff --books santhya-vol-1 --engines all --run    (runs missing engines on the GT pages)

Per ground-truth line the engine's reading is the union of its lines that
overlap the truth box (an engine that split a line in two is not punished for
the split), and the score is the grapheme error rate after normalise() on both
sides -- a dropped matra is one error, a spurious nukta is one error, and the
NFC treatment of ਸ਼ is the same on both sides. Word accuracy is 1 - WER on
punctuation-stripped tokens; it is the number the >= 90% bar is applied to.

Gurbani lines are judged differently: their text is the corpus's by
construction once matched, so the question is whether the engine's reading
FINDS the right line_id (match precision / recall), not how many graphemes it
got right. Both are reported.

The bake-off writes data/raw/ocr-bakeoff.json with a "chosen" block: every
engine that clears the bar per language, or an empty list and a note when none
does. 22_ocr_merge.py reads that block for its defaults.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_engines import ENGINES, VISION_USD_PER_PAGE, read_page
from lib.ocr_match import CorpusIndex, match_text, match_text_all
from lib.ocr_text import GURMUKHI, cer, graphemes, normalise, wer, words
from lib.ocr_zones import classify_zones, header_of
from lib.paths import CORPUS_DB, OCR_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = os.path.join(ROOT, "data", "raw")
BAR_WORD_ACC = 0.90
V_OVERLAP = 0.5           # share of the truth box's height an engine line must cover
H_OVERLAP = 0.3


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    out.append(json.loads(raw))
    return out


def overlap(a, b):
    """(vertical share of a covered by b, horizontal share of a covered by b)."""
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return iy / max(1, a[3] - a[1]), ix / max(1, a[2] - a[0])


def window_of(ref: str, hyp: str) -> str:
    """
    The part of a paragraph-level reading that corresponds to one truth line.

    Surya 2 and the VLM engines return a paragraph as one block. Scoring the
    whole block against one line would count every other line of the
    paragraph as an insertion, so the best-aligned window of the block is
    taken instead -- which is what the merge does with the pivot's lines.
    """
    from rapidfuzz import fuzz
    r, h = graphemes(normalise(ref)), graphemes(normalise(hyp))
    # a merged verse line carries the WHOLE corpus line where the book printed
    # half of it, and an engine line may run into the next; both are longer
    # than the truth by more than a word or two, and both are judged on the
    # window that best matches the truth
    if len(h) <= 1.2 * len(r) + 4 or len(r) < 4:
        return hyp
    from lib.ocr_match import _Codebook
    code = _Codebook()
    rq, hq = code.encode(r), code.encode(h)
    al = fuzz.partial_ratio_alignment(rq, hq)
    if al is None:
        return hyp
    return "".join(h[al.dest_start:al.dest_end])


def reading(lines: list[dict], bbox: list) -> str:
    """What the engine read inside this truth box: overlapping lines, left to right."""
    by_n = {ln.get("n"): ln for ln in lines}
    hits = []
    for ln in lines:
        v, h = overlap(bbox, ln["bbox"])
        # a block (paragraph) box covers the truth line rather than the reverse
        v_block = overlap(ln["bbox"], bbox)[0] if (ln["bbox"][3] - ln["bbox"][1]) > 1.8 * (bbox[3] - bbox[1]) else 0.0
        if (v >= V_OVERLAP or v_block > 0.0 and v >= 0.9) and h >= H_OVERLAP:
            # a merged continuation line's verse was consolidated on its anchor
            # line (22_ocr_merge.py stream matching): read the anchor, and let
            # window_of() cut the part this truth line covers
            anchor = by_n.get(ln.get("merged_into")) if ln.get("merged_into") else None
            hits.append({**ln, "text": ((anchor or {}).get("text") or "") + " " + (ln.get("text") or "")}
                        if anchor else ln)
    if not hits:
        return ""
    # a line box wider than the truth (a two-column page read as one line) is
    # cut to the words inside the truth box when the engine gave words
    parts = []
    for ln in sorted(hits, key=lambda l: l["bbox"][0]):
        ws = ln.get("words") or []
        inside = [w for w in ws if overlap(w["bbox"], bbox)[1] >= 0.5]
        if ws and inside and len(inside) < len(ws):
            parts.append(" ".join(w["text"] for w in inside))
        else:
            parts.append(ln["text"])
    return " ".join(parts)


def evaluate(book_dir: str, engine: str, gt: list[dict], index: CorpusIndex | None, lang: str,
             scripture: str = "G") -> dict:
    by_page: dict = defaultdict(list)
    for g in gt:
        by_page[g["page"]].append(g)
    tot_edits = tot_ref = 0
    per_kind: dict = defaultdict(lambda: {"n": 0, "edits": 0, "ref": 0, "wer_sum": 0.0, "found": 0})
    line_scores = []
    secs = []
    found = 0
    m_right = m_wrong = m_none = m_total = m_false = 0
    for page, rows in sorted(by_page.items()):
        path = os.path.join(engine_dir(book_dir, engine), "%04d.jsonl" % page)
        if not os.path.exists(path):
            continue
        meta, lines = read_page(path)
        lines = [ln for ln in lines if not ln.get("dropped")]
        if meta.get("seconds"):
            secs.append(float(meta["seconds"]))
        hints = header_of(classify_zones(lines, meta["page_w"], meta["page_h"]))
        for g in rows:
            ref = g["text"]
            hyp = window_of(ref, reading(lines, g["bbox"]))
            c, w = cer(ref, hyp), wer(ref, hyp)
            n_ref = len(graphemes(normalise(ref)))
            edits = round(c * n_ref)
            tot_edits += edits
            tot_ref += n_ref
            k = per_kind[g["kind"]]
            k["n"] += 1
            k["edits"] += edits
            k["ref"] += n_ref
            k["wer_sum"] += w
            if hyp:
                found += 1
                k["found"] += 1
            line_scores.append({"id": g["id"], "kind": g["kind"], "cer": round(c, 3), "wer": round(w, 3),
                                "ref": ref, "hyp": hyp})
            if g["kind"] == "gurbani" and index is not None and GURMUKHI.search(ref):
                # The reference is what the TRUTH text matches: a clean reading
                # of a corpus line finds it; a Bhai Gurdas line finds nothing and
                # an engine that "matches" it anyway is wrong.
                expected = {m["line_id"] for m in match_text_all(ref, index, hints, source=scripture)}
                if g.get("line_id") is not None:
                    expected.add(g["line_id"])
                got = match_text_all(hyp, index, hints, source=scripture) if hyp else []
                got_ids = {m["line_id"] for m in got}
                if not expected:
                    if got_ids:
                        m_false += 1
                    continue
                m_total += 1
                if not got_ids:
                    m_none += 1
                elif got_ids & expected:
                    m_right += 1
                else:
                    m_wrong += 1
    n = len(line_scores)
    wer_mean = sum(s["wer"] for s in line_scores) / max(n, 1)
    # the merge's Gurbani lines are the corpus's text by construction and are
    # judged by match precision; its reading accuracy is the commentary's
    prose = [s for s in line_scores if s["kind"] != "gurbani"] or line_scores
    wer_prose = sum(s["wer"] for s in prose) / max(len(prose), 1)
    out = {
        "engine": engine, "lines": n, "lines_found": found,
        "cer": round(tot_edits / max(tot_ref, 1), 4),
        "wer": round(wer_mean, 4), "word_acc": round(max(0.0, 1 - wer_mean), 4),
        "word_acc_prose": round(max(0.0, 1 - wer_prose), 4),
        "by_kind": {k: {"n": v["n"], "cer": round(v["edits"] / max(v["ref"], 1), 4),
                        "word_acc": round(max(0.0, 1 - v["wer_sum"] / max(v["n"], 1)), 4),
                        "found": v["found"]} for k, v in per_kind.items()},
        "sec_per_page": round(sum(secs) / max(len(secs), 1), 2) if secs else None,
        "usd_per_page": VISION_USD_PER_PAGE if engine == "vision" else 0.0,
        "lines_detail": line_scores,
    }
    if m_total or m_false:
        out["gurbani_match"] = {"total": m_total, "right": m_right, "wrong": m_wrong, "unmatched": m_none,
                                "false_positive": m_false,
                                "precision": round(m_right / max(m_right + m_wrong + m_false, 1), 3),
                                "recall": round(m_right / max(m_total, 1), 3)}
    return out


def load_index(corpus: str) -> CorpusIndex | None:
    if corpus and os.path.exists(corpus) and os.path.getsize(corpus) > 0:
        con = sqlite3.connect(corpus)
        try:
            return CorpusIndex.from_sqlite(con)
        finally:
            con.close()
    return None


def engines_with_output(book_dir: str) -> list[str]:
    """Every engine directory, plus "merged" (22_ocr_merge.py's output) when it exists."""
    out = sorted(os.path.basename(d) for d in glob.glob(os.path.join(book_dir, "ocr", "*")) if os.path.isdir(d))
    if os.path.isdir(os.path.join(book_dir, "merged")):
        out.append("merged")
    return out


def engine_dir(book_dir: str, engine: str) -> str:
    return os.path.join(book_dir, "merged") if engine == "merged" else os.path.join(book_dir, "ocr", engine)


def table(rows: list[dict], lang: str) -> str:
    head = "%-10s %-4s %7s %7s %8s %9s %9s %8s %8s %9s" % (
        "engine", "lang", "CER", "WER", "word-acc", "CER-gurb", "CER-comm", "found", "sec/pg", "usd/pg")
    out = [head, "-" * len(head)]
    for r in rows:
        bk = r["by_kind"]
        out.append("%-10s %-4s %7.3f %7.3f %8.3f %9s %9s %8s %8s %9.4f" % (
            r["engine"], lang, r["cer"], r["wer"], r["word_acc"],
            ("%.3f" % bk["gurbani"]["cer"]) if "gurbani" in bk else "-",
            ("%.3f" % bk["commentary"]["cer"]) if "commentary" in bk else "-",
            "%d/%d" % (r["lines_found"], r["lines"]),
            ("%.1f" % r["sec_per_page"]) if r["sec_per_page"] is not None else "-", r["usd_per_page"]))
        if r.get("gurbani_match"):
            gm = r["gurbani_match"]
            out.append("%-10s      gurbani match: %d right, %d wrong, %d unmatched, %d false (precision %.2f, recall %.2f)"
                       % ("", gm["right"], gm["wrong"], gm["unmatched"], gm.get("false_positive", 0),
                          gm["precision"], gm["recall"]))
    return "\n".join(out)


def eval_book(book: str, engines: list[str] | None, corpus: str, out_dir: str, verbose: bool) -> dict:
    book_dir = os.path.join(out_dir, book)
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    lang = meta.get("language", "en")
    gt = [g for g in read_jsonl(os.path.join(book_dir, "gt", "lines.jsonl")) if g.get("verified")]
    if not gt:
        sys.exit("no verified ground truth in %s/gt/lines.jsonl (23_ocr_gt.py)" % book_dir)
    index = load_index(corpus) if lang == "pa" else None
    engines = engines or engines_with_output(book_dir)
    results = []
    for e in engines:
        if not os.path.isdir(engine_dir(book_dir, e)):
            continue
        results.append(evaluate(book_dir, e, gt, index, lang, meta.get("scripture", "G")))
    report = {"book": book, "language": lang, "gt_lines": len(gt), "gt_pages": sorted({g["page"] for g in gt}),
              "engines": [{k: v for k, v in r.items() if k != "lines_detail"} for r in results],
              "routing": routing_report(book_dir, gt)}
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "ocr-eval-%s.json" % book), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({**report, "lines": {r["engine"]: r["lines_detail"] for r in results}}, fh, ensure_ascii=False, indent=1)
    print("\n%s (%s): %d ground-truth lines on %d pages" % (book, lang, len(gt), len(report["gt_pages"])))
    print(table(results, lang))
    if report.get("routing"):
        rr = report["routing"]
        print("routing: %.0f%% of lines routed; precision %.2f, recall %.2f (of %d bad lines) %s"
              % (100 * rr["routed_share"], rr["precision"], rr["recall"], rr["routed_bad"] + rr["kept_bad"],
                 json.dumps(rr["by_reason"], ensure_ascii=False)))
    if verbose:
        for r in results:
            worst = sorted(r["lines_detail"], key=lambda s: -s["cer"])[:5]
            print("\n  %s, worst lines:" % r["engine"])
            for s in worst:
                print("    %s cer=%.2f\n      ref: %s\n      hyp: %s" % (s["id"], s["cer"], s["ref"], s["hyp"]))
    return report


def run_missing(book: str, engine: str, pages: list[int], out_dir: str, allow_paid: bool) -> bool:
    if ENGINES[engine].paid and not allow_paid:
        print("  %s: paid engine, not run (use --allow-paid)" % engine)
        return False
    spec = ",".join(str(p) for p in pages)
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "21_ocr_run.py"),
           "--book", book, "--engine", engine, "--pages", spec, "--out", out_dir]
    print("  running: %s" % " ".join(cmd[1:]))
    return subprocess.call(cmd) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book")
    ap.add_argument("--bakeoff", action="store_true")
    ap.add_argument("--books", help="comma-separated, for --bakeoff")
    ap.add_argument("--engines", help="comma-separated or 'all' (every engine with output)")
    ap.add_argument("--run", action="store_true", help="bakeoff: run engines that have no output on the GT pages")
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--out", default=OCR_DIR)
    ap.add_argument("-v", "--verbose", action="store_true", help="print the worst lines per engine")
    args = ap.parse_args()
    engines = None
    if args.engines and args.engines != "all":
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    if args.bakeoff:
        books = [b.strip() for b in (args.books or args.book or "").split(",") if b.strip()]
        if not books:
            sys.exit("--bakeoff needs --books")
        reports = []
        for book in books:
            book_dir = os.path.join(args.out, book)
            gt = [g for g in read_jsonl(os.path.join(book_dir, "gt", "lines.jsonl")) if g.get("verified")]
            pages = sorted({g["page"] for g in gt})
            if args.run and engines:
                for e in engines:
                    missing = [p for p in pages if not os.path.exists(os.path.join(book_dir, "ocr", e, "%04d.jsonl" % p))]
                    if missing:
                        run_missing(book, e, missing, args.out, args.allow_paid)
            reports.append(eval_book(book, engines, args.corpus, args.out, args.verbose))
        chosen: dict = {}
        for rep in reports:
            lang = rep["language"]
            for r in rep["engines"]:
                acc = r["word_acc_prose"] if r["engine"] == "merged" else r["word_acc"]
                ok = acc >= BAR_WORD_ACC
                chosen.setdefault(lang, {}).setdefault(r["engine"], []).append(
                    {"book": rep["book"], "word_acc": acc, "cer": r["cer"], "clears_bar": ok})
        summary = {lang: sorted(e for e, rows in d.items() if all(x["clears_bar"] for x in rows))
                   for lang, d in chosen.items()}
        out = {"bar_word_acc": BAR_WORD_ACC, "books": reports, "per_engine": chosen, "chosen": summary,
               "note": {lang: ("no engine clears the bar alone; the ensemble (22_ocr_merge.py) must, or M1 waits"
                               if not v else "") for lang, v in summary.items()}}
        os.makedirs(RAW, exist_ok=True)
        with open(os.path.join(RAW, "ocr-bakeoff.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("\nclears the %.0f%% word-accuracy bar: %s" % (BAR_WORD_ACC * 100, json.dumps(summary)))
        print("-> %s" % os.path.join(RAW, "ocr-bakeoff.json"))
        return
    if not args.book:
        sys.exit("give --book, or --bakeoff --books a,b")
    eval_book(args.book, engines, args.corpus, args.out, args.verbose)



def routing_report(book_dir: str, gt: list[dict], bad_cer: float = 0.05) -> dict | None:
    """
    Does the merge's routing flag the lines that are actually wrong?

    For every truth line, the merged line under it: routed or not, and CER
    above bad_cer or not. The four cells give precision (routed lines that
    needed it) and recall (bad lines that were routed), plus the share of
    lines routed -- the number the thresholds in lib/ocr_route.py are set by.
    """
    merged_dir = os.path.join(book_dir, "merged")
    if not os.path.isdir(merged_dir):
        return None
    cells = {"routed_bad": 0, "routed_ok": 0, "kept_bad": 0, "kept_ok": 0}
    reasons: dict = defaultdict(lambda: [0, 0])
    for g in gt:
        path = os.path.join(merged_dir, "%04d.jsonl" % g["page"])
        if not os.path.exists(path):
            continue
        _, lines = read_page(path)
        hits = [ln for ln in lines if not ln.get("dropped") and overlap(g["bbox"], ln["bbox"])[0] >= V_OVERLAP
                and overlap(g["bbox"], ln["bbox"])[1] >= H_OVERLAP]
        if not hits:
            continue
        hyp = " ".join(ln["text"] for ln in sorted(hits, key=lambda l: l["bbox"][0]))
        bad = cer(g["text"], window_of(g["text"], hyp)) > bad_cer
        if g["kind"] == "gurbani" and any(ln.get("kind") == "gurbani" for ln in hits):
            bad = False                                    # corpus text differs from the print by design
        routed = [ln.get("route") for ln in hits if ln.get("route")]
        key = ("routed" if routed else "kept") + ("_bad" if bad else "_ok")
        cells[key] += 1
        for r in routed:
            reasons[r][0 if bad else 1] += 1
    n = sum(cells.values())
    routed = cells["routed_bad"] + cells["routed_ok"]
    bad = cells["routed_bad"] + cells["kept_bad"]
    return {"lines": n, "bad_cer": bad_cer, **cells,
            "routed_share": round(routed / max(n, 1), 3),
            "precision": round(cells["routed_bad"] / max(routed, 1), 3),
            "recall": round(cells["routed_bad"] / max(bad, 1), 3),
            "by_reason": {k: {"bad": v[0], "ok": v[1]} for k, v in reasons.items()}}


if __name__ == "__main__":
    main()
