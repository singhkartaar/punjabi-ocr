"""
A commentary's text, line by line: what a teeka view shows beside each line.

  30_line_commentary.py --src data/santhya --work santhya --translator pa-santhya
  30_line_commentary.py ... --manifest data/books/santhya/manifest.json   # each volume's own angs only
  30_line_commentary.py --src data/santhya --work santhya --print 5

A commentary OCR'd from its printed pages (kind `commentary`, a paired
layout) arrives from 12_ingest_writings.py with each paragraph's `explains`:
the line or lines of the Granth it was printed beside. The Santhya prints a
line and then Bhai Vir Singh's explanation of it, line after line, so 17,874
of its 19,171 links name one line. That is the shape of a teeka -- one text
per line, like the Faridkot Teeka -- and this step writes it so:

  <src>/lines-<translator>.jsonl    {"line_id", "text", "pages": ["2:143", ...]}

which pipeline/node/src/06-build-translations-db.js imports into
translations.sqlite under that translator id, for server.js to offer as a
view (?tr=vs).

Every body paragraph that explains a line is joined to that line's text in
reading order; one that explains two or more lines goes on the last of them,
where the explanation ends. What is not prose is left out: a paragraph with
fewer than two Gurmukhi words, or mostly digits and marks (the page's stray
numerals and smudges the OCR read, "2੭", "4."), and footnotes, which belong
to a word, not the line. The text is the OCR's, at the volume's measured
accuracy; the report says so and the view says where it came from.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict


GURMUKHI = re.compile(r"[਀-੿]")
WORD = re.compile(r"[ਅ-ਹਖ਼-ਫ਼]{2,}")       # two letters or more: a word, not a mark
MIN_WORDS = 2
MIN_GURMUKHI_SHARE = 0.5


def is_prose(text: str) -> bool:
    """A paragraph of Punjabi prose, not a stray numeral or smudge the OCR read."""
    ink = [c for c in text if not c.isspace()]
    if not ink:
        return False
    share = sum(1 for c in ink if GURMUKHI.match(c)) / len(ink)
    return len(WORD.findall(text)) >= MIN_WORDS and share >= MIN_GURMUKHI_SHARE


def line_texts(records: list[dict], angs: dict[int, list[int]] | None = None) -> tuple[dict[int, dict], dict[str, int]]:
    """
    {line_id: {"text", "pages"}} from the paragraph records, in reading order,
    and how many paragraphs were taken or left out, by reason.

    `angs` {part: [from, to]}, when given, keeps a volume to the lines of its own
    angs. Off for the Santhya (the owner's call, 2026-10-01): a link outside
    them is a verse the commentary quotes and explains in passing (vol. 1's
    Japji commentary on ang 929's ਓਅੰਕਾਰਿ ਬੇਦ ਨਿਰਮਏ), 36 paragraphs, and what
    he says there is worth having beside that line too.
    """
    lines: dict[int, dict] = defaultdict(lambda: {"parts": [], "pages": []})
    seen = defaultdict(int)
    for rec in records:
        ex = [e for e in rec.get("explains") or [] if e.get("line_to") or e.get("line_from")]
        if not ex:
            continue
        if rec.get("style") not in ("body",):
            seen["not body (%s)" % rec.get("style")] += 1
            continue
        if (ex[0].get("source") or "G") != "G":
            seen["another scripture"] += 1
            continue
        span = (angs or {}).get(rec.get("part") or 0)
        if span and ex[-1].get("ang") and not (span[0] - 1 <= ex[-1]["ang"] <= span[-1] + 1):
            seen["outside the volume's angs"] += 1
            continue
        text = " ".join(rec["text"].split())
        if not is_prose(text):
            seen["not prose"] += 1
            continue
        line = ex[-1].get("line_to") or ex[-1]["line_from"]
        lines[line]["parts"].append(text)
        page = "%s:%s" % (rec.get("part") or 0, rec["page"])
        if page not in lines[line]["pages"]:
            lines[line]["pages"].append(page)
        seen["taken"] += 1
    out = {line: {"text": " ".join(v["parts"]), "pages": v["pages"]} for line, v in lines.items()}
    return out, dict(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the corpus folder with <work>.jsonl (data/santhya)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--translator", default=None, help="the translator id it is imported as (default pa-<work>)")
    ap.add_argument("--label", default=None, help="what a reader is shown it is (default: the work's title)")
    ap.add_argument("--manifest", help="the books' manifest.json: each part's `angs` bounds the lines it explains")
    ap.add_argument("--print", dest="show", type=int, default=0)
    args = ap.parse_args()
    src = os.path.abspath(args.src)
    translator = args.translator or "pa-%s" % args.work
    with open(os.path.join(src, args.work + ".jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    meta, records = rows[0]["_meta"], rows[1:]
    angs = None
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as fh:
            angs = {w.get("part") or 0: w["angs"] for w in json.load(fh)["works"] if w.get("angs")}
    lines, seen = line_texts(records, angs)
    out = os.path.join(src, "lines-%s.jsonl" % translator)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": {"translator": translator, "lang": meta.get("language", "pa"),
                                       "kind": "teeka", "label": args.label or meta.get("title"),
                                       "author": meta.get("author"), "work": args.work,
                                       "origin": "OCR of the printed volumes (%s)" % ", ".join(meta.get("files", [])),
                                       "lines": len(lines), "paragraphs": seen}}, ensure_ascii=False) + "\n")
        for line in sorted(lines):
            fh.write(json.dumps({"line_id": line, **lines[line]}, ensure_ascii=False) + "\n")
    words = sum(len(v["text"].split()) for v in lines.values())
    print("%s: %d lines with commentary, %d words; paragraphs %s -> %s"
          % (args.work, len(lines), words, json.dumps(seen, ensure_ascii=False), out))
    for line in sorted(lines)[:args.show]:
        print("  %d [%s] %s" % (line, ",".join(lines[line]["pages"]), lines[line]["text"][:200]))


if __name__ == "__main__":
    main()
