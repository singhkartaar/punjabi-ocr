"""
How well each engine reads, measured against the ground truth.

  24_ocr_eval.py --book santhya-vol-1                     every engine with output, vs gt/lines.jsonl
  24_ocr_eval.py --book santhya-vol-1 --merged merged,merged-cov --eval-out data/raw/ocr-eval-santhya-vol-1-cov.json
  24_ocr_eval.py --bakeoff --books santhya-vol-1,ten-masters --engines tesseract,surya,indicocr
  24_ocr_eval.py --bakeoff --books santhya-vol-1 --engines all --run    (runs missing engines on the GT pages)

Two merges side by side: 22_ocr_merge.py --out-name writes a second directory
(merged-<name>), every merged* directory is an "engine" here, and --eval-out
keeps the comparison out of data/raw/ocr-eval-<book>.json, which the merge
reads for its vote weights. A merged line that the coverage pass recovered
(`recovered` on the record) is scored apart from the lines the page layout
found, and a truth row marked `not_text` (an ornament, a rule, noise the
reviewer refused) counts a false recovery when a recovered line sits on it.

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
import re
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


_DANDA_SPACE = re.compile(r"\s*([।॥])\s*")


def canon(text: str) -> str:
    """
    The scored form of a line: a danda's spacing is the typesetter's, not the
    reader's. The book prints "ਵਾਵਣਹਾਰੇ॥" and the corpus text the merge puts
    in its place prints "ਵਾਵਣਹਾਰੇ ॥"; scoring the space as an edit counted a
    right line as 8% wrong.
    """
    return _DANDA_SPACE.sub(r" \1 ", text or "").replace("  ", " ").strip()


def hits_of(lines: list[dict], bbox: list) -> list[dict]:
    """The engine's lines that overlap this truth box, an anchor's verse folded into its continuation."""
    by_n = {ln.get("n"): ln for ln in lines}
    hits = []
    for ln in lines:
        v, h = overlap(bbox, ln["bbox"])
        # a block (paragraph) box covers the truth line rather than the reverse
        v_block = overlap(ln["bbox"], bbox)[0] if (ln["bbox"][3] - ln["bbox"][1]) > 1.8 * (bbox[3] - bbox[1]) else 0.0
        # a truth row drawn across both columns of a page (an older merge read
        # it as one line) is covered by two half lines, each mostly inside it
        h_in = overlap(ln["bbox"], bbox)[1]
        if (v >= V_OVERLAP or v_block > 0.0 and v >= 0.9) and (h >= H_OVERLAP or h_in >= 0.8):
            # a merged continuation line's verse was consolidated on its anchor
            # line (22_ocr_merge.py stream matching): read the anchor, and let
            # window_of() cut the part this truth line covers
            anchor = by_n.get(ln.get("merged_into")) if ln.get("merged_into") else None
            hits.append({**ln, "text": ((anchor or {}).get("text") or "") + " " + (ln.get("text") or "")}
                        if anchor else ln)
    return hits


def reading(lines: list[dict], bbox: list, hits: list[dict] | None = None) -> str:
    """What the engine read inside this truth box: overlapping lines, left to right."""
    hits = hits_of(lines, bbox) if hits is None else hits
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


def _truth_words(text: str) -> set[str]:
    out = set()
    for w in words(canon(text)):
        out.add(w)
        out.update(p for p in w.split("-") if p)
    return out


def audit_corrections(fixes: dict, ref: str, hits: list[dict], gid: str) -> None:
    """
    The merge's word corrections on the lines under one ground-truth row
    (22 records them as [was, now, why]), judged by the truth: right when the
    truth has the correction and not the reading, wrong when the reverse (the
    corrector changed a word the author wrote), unsure when it has neither.
    """
    truth = _truth_words(ref)
    for h in hits:
        for was, now, why in h.get("corrections") or []:
            fixes["made"] += 1
            was_in, now_in = canon(was) in truth, canon(now) in truth
            if now_in and not was_in:
                fixes["right"] += 1
            elif was_in and not now_in:
                fixes["wrong"] += 1
                fixes["wrong_detail"].append({"id": gid, "was": was, "now": now, "why": why})
            else:
                fixes["unsure"] += 1


def evaluate(book_dir: str, engine: str, gt: list[dict], index: CorpusIndex | None, lang: str,
             scripture: str = "G") -> dict:
    by_page: dict = defaultdict(list)
    for g in gt:
        by_page[g["page"]].append(g)
    tot_edits = tot_ref = 0
    fresh = lambda: {"n": 0, "edits": 0, "ref": 0, "wer_sum": 0.0, "found": 0}
    per_kind: dict = defaultdict(fresh)
    # the lines the page layout found against the ones the merge's coverage
    # pass recovered from uncovered ink: read by a different path (a crop, one
    # line at a time), so they are scored apart before anyone trusts them
    per_prov: dict = {"page": fresh(), "recovered": fresh()}
    not_text_n = false_recoveries = 0
    line_scores = []
    secs = []
    found = 0
    m_right = m_wrong = m_none = m_total = m_false = 0
    fixes = {"made": 0, "right": 0, "wrong": 0, "unsure": 0, "wrong_detail": []}
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
            hits = hits_of(lines, g["bbox"])
            recovered = any(h.get("recovered") for h in hits)
            if g.get("not_text"):
                # the reviewer looked at this crop and saw no text: a line the
                # coverage pass put here is a false recovery, and nothing is scored
                not_text_n += 1
                false_recoveries += int(recovered)
                continue
            ref = g["text"]
            audit_corrections(fixes, ref, hits, g["id"])
            hyp = window_of(ref, reading(lines, g["bbox"], hits))
            c, w = cer(canon(ref), canon(hyp)), wer(canon(ref), canon(hyp))
            n_ref = len(graphemes(normalise(ref)))
            edits = round(c * n_ref)
            # A row drawn from the coverage pass (23 --coverage-sample) is a
            # line the page layout MISSED, chosen for that: it scores the
            # recovered lines apart and says nothing about how an engine
            # reads the lines it finds. In the headline it would count every
            # such line against the engines that skipped it -- 22 of 52 rows
            # took the pivot from 94.5% to 53% -- so it stays out of it.
            sampled = g.get("origin") != "coverage"
            buckets = ([per_kind[g["kind"]], per_prov["page"]] if sampled and not recovered
                       else [per_prov["recovered"]] if recovered else [])
            if sampled:
                tot_edits += edits
                tot_ref += n_ref
                if hyp:
                    found += 1
                line_scores.append({"id": g["id"], "kind": g["kind"], "cer": round(c, 3), "wer": round(w, 3),
                                    "ref": ref, "hyp": hyp, **({"recovered": True} if recovered else {})})
            for k in buckets:
                k["n"] += 1
                k["edits"] += edits
                k["ref"] += n_ref
                k["wer_sum"] += w
                if hyp:
                    k["found"] += 1
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
        "by_provenance": {k: {"n": v["n"], "cer": round(v["edits"] / max(v["ref"], 1), 4),
                              "word_acc": round(max(0.0, 1 - v["wer_sum"] / max(v["n"], 1)), 4),
                              "found": v["found"]} for k, v in per_prov.items() if v["n"]},
        "sec_per_page": round(sum(secs) / max(len(secs), 1), 2) if secs else None,
        "usd_per_page": VISION_USD_PER_PAGE if engine == "vision" else 0.0,
        "lines_detail": line_scores,
    }
    if fixes["made"]:
        out["corrections"] = fixes
    if not_text_n:
        out["not_text"] = {"n": not_text_n, "false_recoveries": false_recoveries}
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


def merged_dirs(book_dir: str) -> list[str]:
    """Every merge output: "merged" and any "merged-<name>" 22_ocr_merge.py --out-name wrote."""
    if not os.path.isdir(book_dir):
        return []
    return sorted(d for d in os.listdir(book_dir) if d.startswith("merged") and os.path.isdir(os.path.join(book_dir, d)))


def engines_with_output(book_dir: str, merged: list[str] | None = None) -> list[str]:
    """Every engine directory, plus the merges (all of them, or the ones named)."""
    out = sorted(os.path.basename(d) for d in glob.glob(os.path.join(book_dir, "ocr", "*")) if os.path.isdir(d))
    return out + [m for m in merged_dirs(book_dir) if merged is None or m in merged]


def engine_dir(book_dir: str, engine: str) -> str:
    return os.path.join(book_dir, engine) if engine.startswith("merged") else os.path.join(book_dir, "ocr", engine)


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
        rec = (r.get("by_provenance") or {}).get("recovered")
        if rec:
            nt = r.get("not_text") or {}
            out.append("%-10s      recovered lines: %d, word-acc %.3f, CER %.3f, found %d/%d%s"
                       % ("", rec["n"], rec["word_acc"], rec["cer"], rec["found"], rec["n"],
                          ("; false recoveries %d of %d not-text crops" % (nt["false_recoveries"], nt["n"])) if nt else ""))
        fx = r.get("corrections")
        if fx:
            out.append("%-10s      corrections on ground-truth lines: %d made, %d right, %d wrong, %d unsure"
                       % ("", fx["made"], fx["right"], fx["wrong"], fx["unsure"]))
        lc = r.get("line_counts")
        if lc:
            two = lc.get("two_column")
            out.append("%-10s      lines on %d counted pages: %d of %d, recall %.3f, excess %.3f%s"
                       % ("", lc["pages"], lc["found"], lc["true"], lc["recall"], lc["excess"],
                          ("; two-column: recall %.3f, excess %.3f on %d pages" % (two["recall"], two["excess"], two["pages"]))
                          if two else ""))
        cs = r.get("coverage")
        if cs:
            out.append("%-10s      coverage: %d regions on %d of %d pages, %d recovered, %d still uncovered; refused %s"
                       % ("", cs["regions"], cs["pages_with_residual"], cs["pages"], cs["recovered"], cs["uncovered_after"],
                          json.dumps(cs["rejected"])))
    return "\n".join(out)


def eval_book(book: str, engines: list[str] | None, corpus: str, out_dir: str, verbose: bool,
              merged: list[str] | None = None, eval_out: str | None = None) -> dict:
    book_dir = os.path.join(out_dir, book)
    with open(os.path.join(book_dir, "pages.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    lang = meta.get("language", "en")
    gt = [g for g in read_jsonl(os.path.join(book_dir, "gt", "lines.jsonl")) if g.get("verified")]
    if not gt:
        sys.exit("no verified ground truth in %s/gt/lines.jsonl (23_ocr_gt.py)" % book_dir)
    index = load_index(corpus) if lang == "pa" else None
    engines = engines or engines_with_output(book_dir, merged)
    counts = read_jsonl(os.path.join(book_dir, "gt", "page-counts.jsonl"))
    results = []
    for e in engines:
        if not os.path.isdir(engine_dir(book_dir, e)):
            continue
        r = evaluate(book_dir, e, gt, index, lang, meta.get("scripture", "G"))
        lc, cs = line_counts(book_dir, e, counts), coverage_summary(book_dir, e)
        if lc:
            r["line_counts"] = lc
        if cs:
            r["coverage"] = cs
        results.append(r)
    # the routing report reads one merge: the first named, else the plain one
    report = {"book": book, "language": lang, "gt_lines": len(gt), "gt_pages": sorted({g["page"] for g in gt}),
              "engines": [{k: v for k, v in r.items() if k != "lines_detail"} for r in results],
              "routing": routing_report(book_dir, gt, merged_name=(merged or ["merged"])[0])}
    default_out = os.path.join(RAW, "ocr-eval-%s.json" % book)
    eval_path = os.path.abspath(eval_out) if eval_out else default_out
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({**report, "lines": {r["engine"]: r["lines_detail"] for r in results}}, fh, ensure_ascii=False, indent=1)
    print("\n%s (%s): %d ground-truth lines on %d pages" % (book, lang, len(gt), len(report["gt_pages"])))
    print(table(results, lang))
    if eval_path != os.path.abspath(default_out):
        print("-> %s (a comparison; 22_ocr_merge.py takes its vote weights from %s only)"
              % (eval_path, os.path.relpath(default_out, ROOT)))
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
    ap.add_argument("--merged", help="comma-separated merge directories to score (merged, merged-cov, ...); "
                                     "default every merged* directory")
    ap.add_argument("--eval-out", help="write the report here instead of data/raw/ocr-eval-<book>.json, "
                                       "which 22_ocr_merge.py reads for its vote weights")
    ap.add_argument("--corpus", default=CORPUS_DB)
    ap.add_argument("--out", default=OCR_DIR)
    ap.add_argument("-v", "--verbose", action="store_true", help="print the worst lines per engine")
    args = ap.parse_args()
    engines = None
    if args.engines and args.engines != "all":
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    merged = [m.strip() for m in args.merged.split(",") if m.strip()] if args.merged else None

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
            reports.append(eval_book(book, engines, args.corpus, args.out, args.verbose, merged, args.eval_out))
        chosen: dict = {}
        for rep in reports:
            lang = rep["language"]
            for r in rep["engines"]:
                acc = r["word_acc_prose"] if r["engine"].startswith("merged") else r["word_acc"]
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
    eval_book(args.book, engines, args.corpus, args.out, args.verbose, merged, args.eval_out)


def line_counts(book_dir: str, engine: str, counts: list[dict]) -> dict | None:
    """
    How many of the printed lines an engine (or a merge) has, against the
    reviewer's count of them (23_ocr_gt.py --count-pages): recall is
    sum(min(found, true)) / sum(true), so a page cut into too many lines
    cannot pay for a page with lines missing, and excess is what it found
    over the count. Per column too, where the reviewer counted per column.
    """
    counts = [c for c in counts if c.get("verified") and c.get("body_lines") is not None]
    if not counts:
        return None
    tot: dict = {"pages": 0, "true": 0, "found": 0, "capped": 0, "excess": 0}
    two: dict = {"pages": 0, "true": 0, "found": 0, "capped": 0, "excess": 0}
    for c in counts:
        path = os.path.join(engine_dir(book_dir, engine), "%04d.jsonl" % c["page"])
        if not os.path.exists(path):
            continue
        meta, lines = read_page(path)
        if not engine.startswith("merged"):
            lines = classify_zones(lines, meta["page_w"], meta["page_h"])
        # a wrapped verse row the merge folded into its anchor (merged_into) is
        # a printed line still, whatever its own text now says
        body = [ln for ln in lines if not ln.get("dropped") and ln.get("zone") in ("body", "footnote")
                and (ln.get("text", "").strip() or ln.get("merged_into"))]
        by_col = c.get("by_col")
        if by_col and engine.startswith("merged") and any(ln.get("col") is not None for ln in body):
            pairs = [(sum(1 for ln in body if ln.get("col") == k), true) for k, true in enumerate(by_col)]
        else:
            pairs = [(len(body), c["body_lines"])]
        for bucket in ((tot, two) if c.get("columns") else (tot,)):
            bucket["pages"] += 1
            for found, true in pairs:
                bucket["true"] += true
                bucket["found"] += found
                bucket["capped"] += min(found, true)
                bucket["excess"] += max(0, found - true)
    if not tot["pages"]:
        return None

    def finish(b: dict) -> dict:
        return {"pages": b["pages"], "true": b["true"], "found": b["found"],
                "recall": round(b["capped"] / max(b["true"], 1), 3), "excess": round(b["excess"] / max(b["true"], 1), 3)}
    out = finish(tot)
    if two["pages"]:
        out["two_column"] = finish(two)
    return out


def coverage_summary(book_dir: str, engine: str) -> dict | None:
    """What a merge's coverage pass did over its pages (_meta.coverage), summed."""
    if not engine.startswith("merged"):
        return None
    out: dict = {"pages": 0, "pages_with_residual": 0, "regions": 0, "recovered": 0, "uncovered_after": 0,
                 "rejected": defaultdict(int)}
    for path in glob.glob(os.path.join(engine_dir(book_dir, engine), "*.jsonl")):
        meta, _ = read_page(path)
        cov = meta.get("coverage") or {}
        if not cov.get("on"):
            continue
        out["pages"] += 1
        out["pages_with_residual"] += bool(cov.get("regions"))
        out["regions"] += len(cov.get("regions") or [])
        out["recovered"] += cov.get("recovered", 0)
        out["uncovered_after"] += cov.get("uncovered_after", 0)
        for why, n in (cov.get("rejected") or {}).items():
            out["rejected"][why] += n
    if not out["pages"]:
        return None
    out["rejected"] = dict(out["rejected"])
    return out


def routing_report(book_dir: str, gt: list[dict], bad_cer: float = 0.05, merged_name: str = "merged") -> dict | None:
    """
    Does the merge's routing flag the lines that are actually wrong?

    For every truth line, the merged line under it: routed or not, and CER
    above bad_cer or not. The four cells give precision (routed lines that
    needed it) and recall (bad lines that were routed), plus the share of
    lines routed -- the number the thresholds in lib/ocr_route.py are set by.
    """
    merged_dir = os.path.join(book_dir, merged_name)
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
