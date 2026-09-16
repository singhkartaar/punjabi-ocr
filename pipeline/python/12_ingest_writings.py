"""
Ingest an author's essays: PDFs in, one JSONL of paragraphs per work out.

  12_ingest_writings.py --src /path/to/essays --author "Bau Ji"
  12_ingest_writings.py --src ... --limit 5 --workers 4

Writes data/writings/<work>.jsonl (one record per source paragraph, in reading
order), data/writings/works.json (the roster a reader is shown), and
data/raw/writings-report.json (the numbers the W-1 gate is judged on).

A record is one SOURCE paragraph, deliberately not one retrieval unit, so that
re-chunking for the index never means re-reading a PDF. Everything later steps
need -- the work, the part, the page, the essay marker, whether the paragraph is
a quoted verse -- is decided once, here.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import functools
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paths import CORPUS_DB, OCR_DIR, ROOT
from lib.writings_manifest import list_sources, load_manifest, parse_source
from lib.writings_pdf import read_pdf

OUT_DIR = os.path.join(ROOT, "data", "writings")
REPORT = os.path.join(ROOT, "data", "raw", "writings-report.json")
GLUE = re.compile(r"[a-z][A-Z]|[a-z][0-9]|[0-9][A-Za-z]")
# a sentence ends on Latin punctuation or, in Punjabi, a danda
ENDS_SENTENCE = re.compile("[.!?:;\u2013\u2014\u201d\u2019\")\u0964\u0965]\\s*$")
ANG_TAIL = re.compile(r"(?:^|[^0-9])([0-9]{2,4})\s*$")
QUOTE_KEYS = ("line_ids", "shabad_id", "ang", "match_score", "match_method")


def read_source(path: str, meta: dict) -> dict:
    """
    The reader a source needs: the PDF text layer (the essays), the legacy
    Gurmukhi-font converter (lib/writings_legacy.py), or the merged OCR
    (lib/writings_ocr.py). All three return read_pdf()'s shape.
    """
    reader = meta.get("reader") or "pdf-text"
    if reader == "ocr":
        from lib.writings_ocr import read_ocr_book
        return read_ocr_book(os.path.join(OCR_DIR, meta["book"]))
    if reader == "legacy-font":
        from lib.writings_legacy import read_legacy_pdf
        return read_legacy_pdf(path)
    return read_pdf(path)


def read_one(path: str, manifest: dict | None = None) -> dict:
    """One source to its paragraph records. Runs in a worker process."""
    meta = parse_source(path, manifest)
    try:
        doc = read_source(path, meta)
    except Exception as exc:
        return {"meta": meta, "error": "%s: %s" % (type(exc).__name__, exc), "records": [], "pages": 0}
    records, markers = [], []
    for page in doc["pages"]:
        if page.get("marker"):
            markers.append(page["marker"])
        for i, para in enumerate(page["paragraphs"], start=1):
            rec = {
                "unit_id": "%s:%d:%d:%d" % (meta["work"], meta["part"] or 0, page["page"], i),
                "work": meta["work"], "part": meta["part"], "essay": meta["essay"],
                "page": page["page"], "para_no": i, "marker": page.get("marker"),
                "style": para["style"], "italic": para["italic"], "text": para["text"],
                "lang": meta.get("language", "en"),
            }
            for k in QUOTE_KEYS:
                if para.get(k) is not None:
                    rec[k] = para[k]
            if para.get("routed"):
                rec["routed"] = True
            records.append(rec)
    return {"meta": meta, "records": records, "pages": len(doc["pages"]),
            "markers": markers, "spreads": sum(1 for p in doc["pages"] if p.get("spread"))}


# Real one- and two-letter words. Without these, "a while" cannot be told from a
# broken word, and the first is not a defect. "ji" is one here.
SHORT_WORDS = {"a", "i", "o", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
               "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "am",
               "ah", "oh", "ok", "ye", "ex", "re", "un", "ji"}
LONE = re.compile(r"^[A-Za-z]$")            # a token that is one bare letter
WORDS = re.compile(r"[A-Za-z]+")
MIN_SEEN = 3                                 # times a joined form must occur to be believed


def english_words(min_seen: int = 3) -> set:
    """
    Ordinary English, from the Gurbani translations already in corpus.sqlite.

    An outside arbiter is not a luxury here, it is the whole difficulty. The
    author's own text cannot say whether "imran" is a word, because the defect
    is systematic: the scan broke "Simran" 146 times, so "imran" looks like a
    common word to any count taken over that text. Three rules in a row failed
    on that circularity before this was brought in.

    The 60,403 lines of bdb/ms/ssk are 2.4 million words of ordinary English
    that owe nothing to these PDFs, and they settle it flatly: "and" 89,117,
    "word" 4,636, "inner" 685; "imran" 0, "urmukh" 0, "ubconscious" 0.

    What they cannot settle is the author's own vocabulary -- Simran, Paath and
    Mantar are all 0 here -- so the joined form is still judged by his usage.
    Each vocabulary is asked only what it can answer.
    """
    if not os.path.exists(CORPUS_DB):
        return set()
    con = sqlite3.connect(CORPUS_DB)
    try:
        counts: Counter = Counter()
        for (text,) in con.execute(
                "SELECT text FROM translations WHERE translator IN ('bdb','ms','ssk')"):
            counts.update(w.lower() for w in WORDS.findall(text))
    finally:
        con.close()
    return {w for w, n in counts.items() if n >= min_seen}


def repair_splits(records: list[dict], english: set | None = None) -> int:
    """
    Rejoin words the PDF's own spacing broke: "S imran", "G urmukhs", "thought s".

    Knowing whether a run starts after a real gap means estimating its width,
    and the estimate is occasionally short, which strands a letter. Repairing
    that turns out to be far more delicate than it looks, and three rules failed
    before this one. Each failure is worth keeping, because each is a trap:

    1. A REGEX OVER THE TEXT repairs correct text into nonsense. "\\b(\\w) (\\w+)"
       matches the possessive in "Guru's word" -- there IS a word boundary
       between the apostrophe and the s -- giving "Guru'sword", 91 times, plus
       "one's own" -> "sown" and "one's inner" -> "sinner". Hence tokens: as a
       token "Guru's" is indivisible and the question never arises. A token that
       merely OPENS with an apostrophe is the same trap ("Guru 's word", the
       possessive split by the scan); it is put back on the word before it.

    2. COUNTING THE FRAGMENT defeats itself. Requiring the joined form to be
       commoner than the fragment sounds safe, but the fragment is common
       BECAUSE the defect is systematic: "s" occurs 5,108 times, so "thoughts"
       (919) and "simran" (386) could never clear it. The commoner the defect,
       the less it was repaired.

    3. THE AUTHOR'S OWN TEXT cannot say what a word is, for the same reason. Ask
       it whether "imran" is one and it says yes -- 146 times, every one of them
       this very defect. So an outside arbiter is needed, and the repository
       already holds one: see english_words().

    With that, a stranded letter is offered to both neighbours:

      "S imran"       -> Simran      a lone capital is not a word
      "thought s"     -> thoughts    only the left join is a word
      "touch the e"   -> thee        only the right join is a word
      "influence s"   -> influences  both are words, but a trailing s is a plural
      "blotles s and" -> left alone  "and" plainly stands on its own, so the s
                                     cannot be its first letter
      "the e xternal" -> left alone  "thee" and "external" are both words and
                                     nothing here can say which was meant

    Leaving the last two alone is the point. A wrong join reads as the author's
    own word and is invisible; a stranded letter is visibly a scanning artefact.
    """
    vocab: Counter = Counter()
    for rec in records:
        vocab.update(w.lower() for w in WORDS.findall(rec["text"]))
    english = english_words() if english is None else english
    letters = lambda tok: "".join(WORDS.findall(tok))
    # the joined form only has to be a word THIS author uses (Simran, Paath);
    # whether a piece already stands alone is settled outside his text
    known = lambda w: vocab.get(w.lower(), 0) >= MIN_SEEN or w.lower() in english
    stands_alone = lambda w: w.lower() in english

    fixed = 0
    for rec in records:
        tokens = rec["text"].split(" ")
        out: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            # The stranded letter often arrives wrapped in the quotation marks
            # this author uses constantly -- "'S imran'" -- so the test is on the
            # letters of the token, while the join keeps whatever punctuation
            # was attached to it.
            core = letters(tok)
            # A token opening with an apostrophe is a clitic, not a stranded
            # letter: "Guru 's word" is a possessive the PDF split, and reading
            # its s as stranded joins it forward into "Guru sword". Put it back
            # on the word it belongs to and move on.
            if tok[:1] in "'’" and out and out[-1][-1:].isalpha():
                out[-1] += tok
                i += 1
                fixed += 1
                continue
            if len(core) != 1 or core.lower() in SHORT_WORDS:
                # A split that stranded more than one letter ("wo rld", "Gu rus",
                # "ne gative"). Safe only where NEITHER piece is a word the
                # author uses and the join is: that rules out "in to" and "the
                # rapist", where both pieces stand on their own.
                a, b = letters(tok), letters(nxt)
                if (a and b and not LONE.match(nxt) and tok[-1:].isalpha() and nxt[:1].islower()
                        and not stands_alone(a) and not stands_alone(b) and known(a + b)):
                    out.append(tok + nxt)
                    i += 2
                    fixed += 1
                    continue
                out.append(tok)
                i += 1
                continue
            prev = out[-1] if out else ""
            # a join may only close a gap between letters, never swallow the
            # punctuation that separates two real words
            # A letter never joins forward into a word that already stands on
            # its own: in "blotles s and" the "and" is plainly a word, so the s
            # cannot be its first letter, and joining gave "blotles sand". In
            # "s ubconscious" the remainder is no word at all, so it can.
            head = (core + letters(nxt)
                    if tok[-1:].isalpha() and nxt[:1].isalpha() and nxt[:1].islower()
                    and not stands_alone(letters(nxt)) else "")
            tail = (letters(prev) + core
                    if tok[:1].isalpha() and prev[-1:].isalpha() and prev[-1:].islower() else "")
            hk, tk = bool(head) and known(head), bool(tail) and known(tail)
            if core.isupper():
                take = "head" if hk else None       # a lone capital is never a word
            elif hk and tk:
                take = "tail" if core == "s" else None  # a stranded s is a plural
            elif hk:
                take = "head"
            elif tk:
                take = "tail"
            else:
                take = None
            if take == "head":
                out.append(tok + nxt)
                i += 2
                fixed += 1
            elif take == "tail":
                out[-1] = prev + tok
                i += 1
                fixed += 1
            else:
                out.append(tok)
                i += 1
        rec["text"] = " ".join(out)
    return fixed


def measure(records: list[dict]) -> dict:
    """The W-1 gate: did the paragraphs come back whole and correctly spaced?"""
    body = [r for r in records if r["style"] == "body"]
    quotes = [r for r in records if r["style"] == "quote"]
    words = sum(len(r["text"].split()) for r in records)
    glue = sum(len(GLUE.findall(r["text"])) for r in records)
    mid = sum(1 for r in body if len(r["text"]) > 40 and not ENDS_SENTENCE.search(r["text"]))
    with_ang = 0
    for r in quotes:
        m = ANG_TAIL.search(r["text"])
        if m and 62 <= int(m.group(1)) <= 1430:
            with_ang += 1
    with_ids = sum(1 for r in quotes if r.get("line_ids"))
    return {
        "paragraphs": len(records), "body": len(body), "quotes": len(quotes),
        "headings": sum(1 for r in records if r["style"] == "heading"), "words": words,
        "glue_defects": glue, "glue_pct": round(100.0 * glue / max(words, 1), 3),
        "body_mid_sentence": mid,
        "body_mid_sentence_pct": round(100.0 * mid / max(len(body), 1), 2),
        "quotes_with_ang": with_ang,
        "quotes_with_ang_pct": round(100.0 * with_ang / max(len(quotes), 1), 1),
        "quotes_with_line_ids": with_ids,
        "quotes_with_line_ids_pct": round(100.0 * with_ids / max(len(quotes), 1), 1),
        "routed": sum(1 for r in records if r.get("routed")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder of PDFs: with a manifest.json, or in the essays' root/English/Punjabi convention")
    ap.add_argument("--author", default="Bau Ji", help="for a folder without a manifest (the essays); a manifest names its own")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="first N files only, for a quick look")
    args = ap.parse_args()

    manifest = load_manifest(args.src)
    sources = list_sources(args.src, manifest)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        sys.exit("no PDFs under %s" % args.src)
    if manifest is not None and args.author == ap.get_default("author"):
        args.author = manifest.get("author") or next(
            (w.get("author") for w in manifest["works"] if w.get("author")), args.author)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    print("%d source(s), %d workers%s" % (len(sources), args.workers, ", manifest" if manifest else ""))

    results = []
    reader = functools.partial(read_one, manifest=manifest)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, res in enumerate(pool.map(reader, sources), start=1):
            results.append(res)
            if n % 20 == 0 or n == len(sources):
                print("  %d/%d" % (n, len(sources)))

    by_work = defaultdict(list)
    failures, mismatched = [], []
    for res in results:
        if res.get("error"):
            failures.append({"file": res["meta"]["file"], "error": res["error"]})
            continue
        # The pages carry "L127.3"; the filename says essay 127. A disagreement
        # is reported rather than trusted -- it means one of them is mislabelled.
        essay = res["meta"]["essay"]
        # only the essays carry "L<n>.<page>" markers; a book's marker is its
        # printed page number and says nothing about an essay number
        seen = Counter(int(m[1:].split(".")[0]) for m in res.get("markers", [])
                       if m and m[:1] == "L" and m[1:].split(".")[0].isdigit()) if essay else Counter()
        if essay and seen and seen.most_common(1)[0][0] != essay:
            mismatched.append({"file": res["meta"]["file"], "filename_essay": essay,
                               "marker_essay": seen.most_common(1)[0][0]})
        by_work[res["meta"]["work"]].extend(res["records"])

    # one pass over everything, so a word broken in one essay is judged against
    # the author's usage across all of them. The repair is an English one (its
    # arbiter vocabulary is English); Punjabi records are left as the OCR
    # merge produced them.
    english_records = [r for recs in by_work.values() for r in recs if r.get("lang", "en") == "en"]
    rejoined = repair_splits(english_records) if english_records else 0
    print("rejoined %d words the PDF spacing had split" % rejoined)

    works = []
    for work, records in sorted(by_work.items()):
        records.sort(key=lambda r: (r["part"] or 0, r["page"], r["para_no"]))
        metas = [r["meta"] for r in results if not r.get("error") and r["meta"]["work"] == work]
        title = Counter(m["work_title"] for m in metas).most_common(1)[0][0]
        author = metas[0].get("author") or args.author
        original = all(m["original"] for m in metas)
        policy = metas[0].get("quote_policy") or ("verbatim" if original else "summarise")
        head_meta = {"work": work, "title": title, "author": author, "original": original,
                     "quote_policy": policy, "files": sorted(m["file"] for m in metas),
                     "language": metas[0].get("language", "en"), "licence": metas[0].get("licence"),
                     "source": metas[0].get("reader") or "pdf-text"}
        with open(os.path.join(args.out, work + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"_meta": head_meta}, ensure_ascii=False) + "\n")
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        works.append({"work": work, "title": title, "author": author,
                      "parts": sorted(p for p in {m["part"] for m in metas} if p),
                      "files": len(metas), "folder": metas[0]["folder"],
                      "original": original, "quote_policy": policy,
                      "language": head_meta["language"], "licence": head_meta["licence"],
                      **measure(records)})

    # works.json holds every author's works: this run replaces its own
    # author's entries and keeps the others
    roster_path = os.path.join(args.out, "works.json")
    roster: dict = {"author": args.author, "works": []}
    if os.path.exists(roster_path):
        with open(roster_path, encoding="utf-8") as fh:
            roster = json.load(fh)
    mine = {w["author"] for w in works}
    kept = [w for w in roster.get("works", []) if w.get("author") not in mine]
    roster_works = kept + works
    with open(roster_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"author": roster.get("author") or args.author,
                   "authors": sorted({w["author"] for w in roster_works}),
                   "works": roster_works}, fh, ensure_ascii=False, indent=2)

    every = [r for recs in by_work.values() for r in recs]
    report = {"src": args.src, "author": args.author, "files": len(sources),
              "works": len(works), "failures": failures, "rejoined_words": rejoined,
              "essay_number_mismatch": mismatched,
              "pages": sum(r.get("pages", 0) for r in results),
              "spread_pages": sum(r.get("spreads", 0) for r in results),
              "totals": measure(every)}
    report_path = REPORT if manifest is None else REPORT.replace(
        "writings-report.json", "writings-report-%s.json" % re.sub(r"[^a-z0-9]+", "-", args.author.lower()).strip("-"))
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    t = report["totals"]
    print("\n%d works, %d pages (%d spreads), %d paragraphs, %d words"
          % (report["works"], report["pages"], report["spread_pages"], t["paragraphs"], t["words"]))
    print("  quotes %d (%d with an ang, %s%%)" % (t["quotes"], t["quotes_with_ang"], t["quotes_with_ang_pct"]))
    print("  glue defects %s%%  |  body paragraphs ending mid-sentence %s%%"
          % (t["glue_pct"], t["body_mid_sentence_pct"]))
    if failures:
        print("  FAILED: %d" % len(failures))
    if mismatched:
        print("  essay number disagrees with page marker in %d files" % len(mismatched))
    print("  -> %s  and  %s" % (args.out, report_path))


if __name__ == "__main__":
    main()
