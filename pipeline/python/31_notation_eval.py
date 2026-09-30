"""
A notation book's parse against its gold: field accuracy, false links, the bars.

  31_notation_eval.py --book gurmat-sangeet-sagar-1
  31_notation_eval.py --book gurmat-sangeet-sagar-1 --strict     # exit 1 under the bar

Reads data/notations/<book>/notations.jsonl and data/ocr/<book>/gt/
notation-gold.jsonl (30_notation_gt.py --promote). For every gold record
with a status other than skip, each judged field of the current record is
compared with the truth the reviewer settled: the shabad, the raag used,
the taal, the laya, the section structure, and -- where the reviewer
corrected cells -- the grid's swaras and bol. A false link is a record that
names a shabad the reviewer says is not the one. The table goes to the
terminal and data/raw/notation-eval-<book>.json; 32_build_notations_db.py
reads `passed` from there and refuses a book under the bar unless told not to.

With fewer than DECISIVE reviewed notations the percentages are not the
gate -- every fault is -- so the report says so; the bars decide once a
book has that many.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.notation import read_jsonl
from lib.paths import NOTATIONS_DIR, OCR_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# docs/notations.md, "acceptance bars": accuracy a book must reach on its reviewed sample
BARS = {"shabad": 0.98, "false_link": 0.01, "raag_used": 0.95, "taal": 0.97, "laya": 0.90, "structure": 0.95,
        "swara": 0.95, "swara_letter": 0.98, "bol": 0.95, "beats": 0.95, "unknown": 0.03}
DECISIVE = 30
FIELDS = ("shabad", "raag_used", "taal", "laya", "structure")


def value_of(rec: dict, field: str):
    h = rec.get("heading") or {}
    if field == "shabad":
        return (rec.get("shabad") or {}).get("shabad_id")
    if field == "raag_used":
        return (h.get("raag") or {}).get("key")
    if field == "taal":
        return (h.get("taal") or {}).get("key")
    if field == "laya":
        return (h.get("taal") or {}).get("laya")
    if field == "structure":
        secs = rec.get("sections") or (rec.get("layout") or {}).get("sections") or []
        return ",".join(str(s.get("kind") or "?") for s in secs)
    return None


def truth_of(gold: dict, field: str):
    j = gold.get(field) or {}
    if "truth" in j:
        return j["truth"]
    return j.get("value") if j.get("ok") else (j.get("correct") or None)


def same(a, b) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    return str(a) == str(b)


def cell_metrics(rec: dict, gold: dict) -> dict:
    """
    Swara and bol accuracy over the lines the reviewer checked: every beat
    of a checked line counts, a correction the gold lists is a miss.
    """
    sections = rec.get("sections") or []
    checked = gold.get("cells_checked", "all")
    lines: list[tuple[int, int, dict]] = []
    for s, sec in enumerate(sections):
        for l, line in enumerate(sec.get("lines") or []):
            if checked == "all" or (isinstance(checked, list) and any(c.get("s") == s and c.get("l") == l for c in checked)):
                lines.append((s, l, line))
    beats = sum(len(line.get("beats") or []) for _, _, line in lines)
    wrong = {"notes": 0, "letter": 0, "bol": 0, "beats": 0}
    for c in gold.get("cells") or []:
        f = c.get("field")
        if f in ("notes", "ext", "rest"):
            wrong["notes"] += 1
            # a letter-only miss: the corrected swara differs in its letter, not only a mark
            try:
                s, l, b = int(c["s"]), int(c["l"]), int(c["b"])
                beat = sections[s]["lines"][l]["beats"][b]
                had = "".join(n.get("s", "?") for n in (beat.get("notes") or [])) if beat.get("notes") else ("-" if beat.get("ext") else "*")
                want = str(c.get("value") or "")
                if "".join(ch for ch in want.upper() if ch.isalpha()) != had.upper():
                    wrong["letter"] += 1
            except (KeyError, IndexError, TypeError, ValueError):
                wrong["letter"] += 1
        elif f == "bol":
            wrong["bol"] += 1
        elif f in ("beats", "m", "div"):
            wrong["beats"] += 1
    unknown = sum(1 for _, _, line in lines for b in (line.get("beats") or []) if b.get("notes") is None and not b.get("ext") and not b.get("rest"))
    return {"beats": beats, "wrong_swara": wrong["notes"], "wrong_letter": wrong["letter"], "wrong_bol": wrong["bol"],
            "wrong_beats": wrong["beats"], "unknown": unknown, "lines": len(lines)}


def evaluate(records: list[dict], gold: list[dict]) -> dict:
    by_id = {r["notation_id"]: r for r in records}
    fields = {f: {"n": 0, "correct": 0, "misses": []} for f in FIELDS}
    false_links, linked = [], 0
    cells = {"beats": 0, "wrong_swara": 0, "wrong_letter": 0, "wrong_bol": 0, "wrong_beats": 0, "unknown": 0, "lines": 0, "n": 0}
    missing = []
    n = 0
    for g in gold:
        if g.get("status") == "skip" or not g.get("verified", True):
            continue
        rec = by_id.get(g["notation_id"])
        if rec is None:
            missing.append(g["notation_id"])
            continue
        n += 1
        for f in FIELDS:
            t = truth_of(g, f)
            v = value_of(rec, f)
            fields[f]["n"] += 1
            if same(v, t):
                fields[f]["correct"] += 1
            else:
                fields[f]["misses"].append({"notation_id": g["notation_id"], "got": v, "truth": t})
        if value_of(rec, "shabad") is not None:
            linked += 1
            if not same(value_of(rec, "shabad"), truth_of(g, "shabad")):
                false_links.append(g["notation_id"])
        if rec.get("sections") and (g.get("cells") is not None):
            m = cell_metrics(rec, g)
            for k in ("beats", "wrong_swara", "wrong_letter", "wrong_bol", "wrong_beats", "unknown", "lines"):
                cells[k] += m[k]
            cells["n"] += 1
    out: dict = {"n": n, "missing": missing, "fields": {}, "false_link": {"linked": linked, "n": len(false_links),
                                                                          "rate": (len(false_links) / linked) if linked else 0.0,
                                                                          "ids": false_links}}
    for f, d in fields.items():
        out["fields"][f] = {"n": d["n"], "correct": d["correct"], "acc": (d["correct"] / d["n"]) if d["n"] else None,
                            "misses": d["misses"]}
    b = cells["beats"]
    out["cells"] = {**cells,
                    "swara": (1 - cells["wrong_swara"] / b) if b else None,
                    "swara_letter": (1 - cells["wrong_letter"] / b) if b else None,
                    "bol": (1 - cells["wrong_bol"] / b) if b else None,
                    "beats_ok": (1 - cells["wrong_beats"] / b) if b else None,
                    "unknown_rate": (cells["unknown"] / b) if b else None}
    return out


def judge(result: dict) -> tuple[bool, list[str]]:
    """Every field on or above its bar (fields with no measurement do not fail)."""
    fails = []
    for f in FIELDS:
        acc = result["fields"][f]["acc"]
        if acc is not None and acc < BARS[f]:
            fails.append("%s %.3f < %.2f" % (f, acc, BARS[f]))
    if result["false_link"]["linked"] and result["false_link"]["rate"] > BARS["false_link"]:
        fails.append("false links %.3f > %.2f" % (result["false_link"]["rate"], BARS["false_link"]))
    c = result["cells"]
    for k, bar in (("swara", BARS["swara"]), ("swara_letter", BARS["swara_letter"]), ("bol", BARS["bol"]), ("beats_ok", BARS["beats"])):
        if c.get(k) is not None and c[k] < bar:
            fails.append("%s %.3f < %.2f" % (k, c[k], bar))
    if c.get("unknown_rate") is not None and c["unknown_rate"] > BARS["unknown"]:
        fails.append("unknown cells %.3f > %.2f" % (c["unknown_rate"], BARS["unknown"]))
    return not fails, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--gold", help="a gold file other than data/ocr/<book>/gt/notation-gold.jsonl")
    ap.add_argument("--out", help="where to write the report (default data/raw/notation-eval-<book>.json)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the book is under a bar")
    args = ap.parse_args()

    path = os.path.join(NOTATIONS_DIR, args.book, "notations.jsonl")
    gold_path = args.gold or os.path.join(OCR_DIR, args.book, "gt", "notation-gold.jsonl")
    if not os.path.exists(path):
        sys.exit("no %s; run 29_notation_parse.py first" % path)
    if not os.path.exists(gold_path):
        sys.exit("no %s; review with 30_notation_gt.py and promote" % gold_path)
    _, records = read_jsonl(path)
    with open(gold_path, encoding="utf-8") as fh:
        gold = [json.loads(l) for l in fh if l.strip()]
    result = evaluate(records, gold)
    passed, fails = judge(result)
    decisive = result["n"] >= DECISIVE
    report = {"book": args.book, "gold": os.path.relpath(gold_path, ROOT).replace("\\", "/"), "bars": BARS,
              "decisive": decisive, "decisive_at": DECISIVE, "passed": passed and not result["missing"],
              "fails": fails, **result}
    out = args.out or os.path.join(ROOT, "data", "raw", "notation-eval-%s.json" % args.book)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print("%s: %d reviewed notation(s)%s" % (args.book, result["n"], "" if decisive else
                                             " (under %d: every fault counts, the percentages are not yet the gate)" % DECISIVE))
    print("  %-14s %6s %8s %6s" % ("field", "n", "accuracy", "bar"))
    for f in FIELDS:
        d = result["fields"][f]
        print("  %-14s %6d %8s %6.2f%s" % (f, d["n"], ("%.3f" % d["acc"]) if d["acc"] is not None else "-", BARS[f],
                                          "  FAIL" if d["acc"] is not None and d["acc"] < BARS[f] else ""))
        for m in d["misses"][:6]:
            print("      %s: got %r, truth %r" % (m["notation_id"], m["got"], m["truth"]))
    fl = result["false_link"]
    print("  %-14s %6d %8.3f %6.2f%s" % ("false links", fl["n"], fl["rate"], BARS["false_link"],
                                        "  FAIL" if fl["linked"] and fl["rate"] > BARS["false_link"] else ""))
    c = result["cells"]
    if c["beats"]:
        print("  cells: %d beats on %d lines of %d grid(s): swara %.3f, letter %.3f, bol %.3f, beats %.3f, unknown %.3f"
              % (c["beats"], c["lines"], c["n"], c["swara"], c["swara_letter"], c["bol"], c["beats_ok"], c["unknown_rate"]))
    else:
        print("  cells: no parsed grid reviewed yet (the grid reader is M2)")
    if result["missing"]:
        print("  gold records with no current parse: %s" % ", ".join(result["missing"]))
    print("  -> %s (%s)" % ("PASS" if report["passed"] else "FAIL: " + "; ".join(fails or ["gold without a parse"]), out))
    if args.strict and not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
