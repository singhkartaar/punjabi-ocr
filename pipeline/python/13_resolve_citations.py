"""
Resolve the Gurbani each passage quotes to a shabad in the corpus.

  13_resolve_citations.py
  13_resolve_citations.py --work simran --print 20
  13_resolve_citations.py --src data/rama --restyle

Reads data/writings/*.jsonl, writes data/writings/citations.jsonl (one record
per resolved quotation) and the numbers into data/raw/writings-citations.json.
`--src` names another corpus that quotes the same way -- In Search of the True
Guru prints the English of the verse and "(SGGS p. 920)" after it -- and its
report goes to data/raw/<corpus>-citations.json.

`--restyle` marks a paragraph that is nothing but a resolved quotation as a
`quote`, the way 14-resolve-legacy.js does for a verse in the legacy face, so
that 14_embed_writings.py keeps the verse out of the unit's text and hangs the
citation on the prose that introduced it. Off by default: Bau Ji's corpus was
built and shipped without it, and does not get to shift.

The measurement that matters is HELD-OUT ANG SELF-CONSISTENCY: resolve each
quotation again with the ang window taken away, and use the cited ang purely as
the judge. That is a real precision estimate on a real labelled set, obtained
for free, and it is the number to quote. Anything else here is a yield.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.citations import (ACCEPT, MARGIN, MIN_OVERLAP, MIN_TOKENS, corpus_by_ang, detect,
                           find_ang, precomputed, resolve_lexical, tokens, wholly)
from lib.paths import CORPUS_DB, ROOT

WRITINGS = os.path.join(ROOT, "data", "writings")
OUT = os.path.join(WRITINGS, "citations.jsonl")
REPORT = os.path.join(ROOT, "data", "raw", "writings-citations.json")


def load_works(src: str, work: str | None) -> list[dict]:
    """Each work's file, its `_meta` line and its paragraph records, in reading order."""
    out = []
    for path in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        name = os.path.basename(path)
        # not the works: the citations, the units files (one per language),
        # and a work's English translation (26_translate_writings.py)
        if name == "citations.jsonl" or name.startswith("units") or name.endswith(".en.jsonl"):
            continue
        if work and name != work + ".jsonl":
            continue
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        out.append({"path": path, "meta": rows[0], "records": [r for r in rows[1:]]})
    return out


def kept_citations(path: str, replaced: set[str]) -> list[dict]:
    """The citations already in `path` of every work not in `replaced`, in file order."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                continue
            work = rec.get("work") or rec.get("unit_id", "").split(":")[0]
            if work not in replaced:
                out.append(rec)
    return out


def restyle(works: list[dict], resolved: list[dict]) -> int:
    """
    Paragraphs that are nothing but a resolved quotation, marked as quotes.

    Only a paragraph that opens with a quotation mark and closes with one after
    the citation is taken, and only where the quotation resolved: an unresolved
    verse restyled would vanish from the passage with no shabad card to stand
    in for it, and a paragraph that is half prose would lose the prose.
    """
    by_id = {}
    for w in works:
        for rec in w["records"]:
            by_id[rec["unit_id"]] = rec
    changed = 0
    for r in resolved:
        rec = by_id.get(r["unit_id"])
        if rec is None or rec["style"] == "quote":
            continue
        first = by_id.get(r.get("opened_by") or "")
        if first is not None:
            # the two halves of one quotation: the first opens it, this closes it
            _, tail = find_ang(rec["text"])
            if not re.search(r"[\"”]\s*[.)]?\s*$", tail.strip()):
                continue
            for half in (first, rec):
                half["style"] = "quote"
                changed += 1
        elif wholly(rec["text"]):
            rec["style"] = "quote"
            changed += 1
    for w in works:
        with open(w["path"], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(w["meta"], ensure_ascii=False) + "\n")
            for rec in w["records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed


class FreeIndex:
    """
    Every English line of the Granth, searchable without an ang.

    Used ONLY as the judge in the held-out measurement -- never to resolve a
    citation, because without the ang window a confident-looking top hit on a
    paraphrase is a guess. An inverted index makes it cheap enough to judge
    every ang-bearing quotation rather than a sample of them, which matters:
    judging a few hundred selects for unusually distinctive verses and flatters
    the result.
    """

    def __init__(self, by_ang: dict):
        self.shabad, self.ang, self.size = [], [], []
        self.postings: dict[str, list[int]] = {}
        for ang, rows in by_ang.items():
            for shabad_id, _line_id, tok in rows:
                i = len(self.shabad)
                self.shabad.append(shabad_id)
                self.ang.append(ang)
                self.size.append(len(tok))
                for t in tok:
                    self.postings.setdefault(t, []).append(i)

    def best(self, text: str):
        """The best-scoring shabad anywhere, or None where nothing stands clear."""
        q = tokens(text)
        if len(q) < MIN_TOKENS:
            return None
        hits: dict[int, int] = {}
        for t in q:
            for i in self.postings.get(t, ()):
                hits[i] = hits.get(i, 0) + 1
        if not hits:
            return None
        best: dict[int, float] = {}
        for i, n in hits.items():
            if n < MIN_OVERLAP:
                continue
            score = n / float(self.size[i])
            sid = self.shabad[i]
            if score > best.get(sid, 0.0):
                best[sid] = score
        if not best:
            return None
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        top = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if top < ACCEPT or top - second < MARGIN:
            return None
        return ranked[0][0]


def precomputed_citation(p: dict) -> dict:
    """A quotation the OCR merge (or 35_link_scriptures.py) already placed, as a citations.jsonl row.

    The line range and the scripture travel with it: a Dasam Bani line id read
    as a Guru Granth Sahib one is a wrong verse, silently.
    """
    return {**{k: p[k] for k in ("unit_id", "work", "part", "page", "para_no", "ang", "how")},
            "span": p["text"][:400], "shabad_id": p["shabad_id"], "line_id": p["line_id"],
            "line_ids": p["line_ids"], "line_from": p["line_from"], "line_to": p["line_to"],
            "source": p["source"], "score": p["score"], "method": p["method"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None, help="one work only")
    ap.add_argument("--print", dest="show", type=int, default=0, help="print N resolved citations")
    ap.add_argument("--holdout", type=int, default=0, help="limit the precision measurement to N spans (0 = all)")
    ap.add_argument("--src", default=WRITINGS, help="directory of <work>.jsonl (default data/writings)")
    ap.add_argument("--restyle", action="store_true",
                    help="mark a paragraph that is nothing but a resolved quotation as a quote")
    args = ap.parse_args()
    src = os.path.abspath(args.src)
    out_path = os.path.join(src, "citations.jsonl")
    report_path = (REPORT if src == os.path.abspath(WRITINGS)
                   else os.path.join(ROOT, "data", "raw", os.path.basename(src) + "-citations.json"))

    works = load_works(src, args.work)
    records = [r for w in works for r in w["records"]]
    # OCR'd books arrive with their Gurbani already matched in Gurmukhi
    # (lib/citations.precomputed); the English resolver runs on the rest
    pre = precomputed(records)
    spans = detect([r for r in records if not r.get("line_ids")])
    print("%d paragraphs, %d quotations detected (%d with an ang, %d continued across a page), "
          "%d resolved by the OCR merge"
          % (len(records), len(spans), sum(1 for s in spans if s["ang"]),
             sum(1 for s in spans if s.get("opened_by")), len(pre)))

    con = sqlite3.connect(CORPUS_DB)
    by_ang = corpus_by_ang(con)
    con.close()
    if by_ang:
        print("corpus lines indexed by ang: %d" % sum(len(v) for v in by_ang.values()))
    else:
        print("no English translations in %s: only the merge's own Gurmukhi matches are kept" % CORPUS_DB)

    resolved, ambiguous, unresolved = [], 0, 0
    for p in pre:
        resolved.append(precomputed_citation(p))
    for span in spans:
        hit = resolve_lexical(span, by_ang)
        if hit is None:
            unresolved += 1
            continue
        if hit.get("ambiguous"):
            ambiguous += 1
            continue
        resolved.append({**{k: span[k] for k in ("unit_id", "work", "part", "page", "para_no", "ang", "how")},
                         **({"opened_by": span["opened_by"]} if span.get("opened_by") else {}),
                         "span": span["text"][:400], **hit})

    restyled = restyle(works, resolved) if args.restyle else 0
    # --work replaces that work's citations and keeps every other work's: a
    # scanned book that joins a roster's corpus (Baru Sahib) shares the folder's
    # citations.jsonl with what the legacy-font and transliteration resolvers
    # wrote for the roster's works, which this script never reads again
    others = kept_citations(out_path, {args.work}) if args.work else []
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"accept": ACCEPT, "margin": MARGIN,
                                       "translators": ["ssk", "bdb", "ms"]}}, ensure_ascii=False) + "\n")
        for r in others + resolved:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if others:
        print("kept %d citations of the folder's other works" % len(others))

    # Held-out ang self-consistency. Resolve again with the window removed and
    # let the cited ang judge: agreement is precision on a free labelled set.
    free = FreeIndex(by_ang)
    checkable = [s for s in spans if s["ang"]]
    if args.holdout:
        checkable = checkable[: args.holdout]
    agree = disagree = silent = 0
    misses = []
    for span in checkable:
        guess = free.best(span["text"])
        if guess is None:
            silent += 1
            continue
        window = resolve_lexical(span, by_ang)
        if window and window.get("shabad_id") == guess:
            agree += 1
        elif window and window.get("shabad_id"):
            disagree += 1
            if len(misses) < 12:
                misses.append({"ang": span["ang"], "window": window["shabad_id"], "free": guess,
                               "score": window["score"], "span": span["text"][:90]})
        else:
            silent += 1
    judged = agree + disagree
    precision = 100.0 * agree / judged if judged else 0.0

    with_ang = sum(1 for s in spans if s["ang"])
    report = {
        "paragraphs": len(records), "quotations": len(spans), "with_ang": with_ang,
        "resolved": len(resolved), "ambiguous": ambiguous, "unresolved": unresolved,
        "yield_of_ang_bearing_pct": round(100.0 * len(resolved) / max(with_ang, 1), 1),
        "distinct_shabads": len(set(r["shabad_id"] for r in resolved)),
        "precomputed": len(pre),
        "by_detection": dict(Counter([s["how"] for s in spans] + [p["how"] for p in pre])),
        "continued_across_a_page": sum(1 for s in spans if s.get("opened_by")),
        "restyled": restyled,
        "holdout": {"sample": len(checkable), "judged": judged, "agree": agree,
                    "disagree": disagree, "no_free_hit": silent,
                    "precision_pct": round(precision, 1), "disagreements": misses},
        "accept": ACCEPT, "margin": MARGIN,
    }
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\nresolved %d of %d ang-bearing quotations (%.1f%%) to %d distinct shabads"
          % (len(resolved), with_ang, report["yield_of_ang_bearing_pct"], report["distinct_shabads"]))
    print("  ambiguous %d | unresolved %d | restyled as quotes %d" % (ambiguous, unresolved, restyled))
    print("  held-out precision: %d of %d judged agree (%.1f%%), %d had no free hit"
          % (agree, judged, precision, silent))
    for r in resolved[: args.show]:
        print("   ang %-5d shabad %-5d score %.2f  %s" % (r["ang"], r["shabad_id"], r["score"], r["span"][:70]))
    print("  -> %s  and  %s" % (out_path, report_path))


if __name__ == "__main__":
    main()
