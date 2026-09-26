"""
Ingest an author's essays: PDFs in, one JSONL of paragraphs per work out.

  12_ingest_writings.py --src /path/to/essays --author "Bau Ji"
  12_ingest_writings.py --src /path/to/books --roster rosters/akj.json --out data/akj
  12_ingest_writings.py --src /path/to/scans            # a folder with a manifest.json
  12_ingest_writings.py --src ... --limit 5 --workers 4

Writes <out>/<work>.jsonl (one record per source paragraph, in reading order),
<out>/works.json (the listing a reader is shown), and a report (the numbers
the W-1 gate is judged on): data/raw/writings-report.json for the essays,
<out>/report.json for another corpus.

A source is read by the reader it needs: the PDF's own text layer, the legacy
Gurmukhi-font converter, or the merged OCR of a scanned book (20-22_ocr_*.py),
which a manifest.json beside the PDFs names (lib/writings_manifest.py).

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


def read_source(path: str, meta: dict, furniture: bool = False, italic_quotes: bool = True) -> dict:
    """
    The reader a source needs: the PDF text layer (the essays), the legacy
    Gurmukhi-font converter (lib/writings_legacy.py), or the merged OCR
    (lib/writings_ocr.py). All three return read_pdf()'s shape. `furniture`
    and `italic_quotes` are the text layer's knobs (lib/writings_pdf.py); the
    OCR merge has already decided both.
    """
    reader = meta.get("reader") or "pdf-text"
    if reader == "ocr":
        from lib.writings_ocr import read_ocr_book
        return read_ocr_book(os.path.join(OCR_DIR, meta["book"]))
    if reader == "legacy-font":
        from lib.writings_legacy import read_legacy_pdf
        return read_legacy_pdf(path)
    return read_pdf(path, furniture=furniture, italic_quotes=italic_quotes)


def read_one(path: str, manifest: dict | None = None, roster: dict | None = None,
             furniture: bool = False, italic_quotes: bool = True) -> dict:
    """One source to its paragraph records. Runs in a worker process."""
    meta = parse_source(path, manifest, roster)
    try:
        doc = read_source(path, meta, furniture=furniture, italic_quotes=italic_quotes)
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
            # what the OCR merge already knew about a quoted verse travels
            # with the paragraph (lib/writings_ocr.py); a text layer has none
            for k in QUOTE_KEYS:
                if para.get(k) is not None:
                    rec[k] = para[k]
            if para.get("routed"):
                rec["routed"] = True
            records.append(rec)
    spans = ((roster or {}).get("works", {}).get(meta["file"]) or {}).get("keep")
    if spans:
        records = keep_spans(records, spans, meta["file"])
    return {"meta": meta, "records": records, "pages": len(doc["pages"]),
            "markers": markers, "spreads": sum(1 for p in doc["pages"] if p.get("spread"))}


def keep_spans(records: list[dict], spans: list[dict], name: str = "") -> list[dict]:
    """
    Only the paragraphs inside the roster's spans, each named by its first and
    last paragraph.

    For a file that is mostly something already ingested: Se Kinehiya's summary
    translation retells chapters the full book tells, and what it has that the
    book does not starts and stops mid-page, where no page range can cut. A span
    whose ends are not found is an error -- a changed scan must not quietly
    ingest nothing, or everything.
    """
    kept = []
    for span in spans:
        start = next((i for i, r in enumerate(records)
                      if r["text"].strip().startswith(span["from"])), None)
        if start is None:
            raise ValueError("%s: no paragraph starts with %r" % (name, span["from"]))
        end = next((i for i in range(start, len(records))
                    if records[i]["text"].strip().endswith(span["to"])), None)
        if end is None:
            raise ValueError("%s: no paragraph after %r ends with %r" % (name, span["from"], span["to"]))
        kept.extend(records[start:end + 1])
    return kept


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
    ap.add_argument("--roster", default=None,
                    help="an author's own listing of their files (rosters/*.json); "
                         "without one the filenames are parsed as Bau Ji's are")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="first N files only, for a quick look")
    args = ap.parse_args()

    # Two ways a file gets named, and a folder has one of them. A manifest.json
    # sits beside the PDFs and never enters the repository (a scanned book's
    # author, language, licence and reader); a roster is kept here under
    # rosters/ for a folder that has no manifest. Neither: the filenames are
    # parsed as Bau Ji's are.
    manifest = load_manifest(args.src)
    roster = {}
    if args.roster:
        with open(args.roster, encoding="utf-8") as fh:
            roster = json.load(fh)
        author = roster.get("author") or args.author
        named = set(roster.get("works", {}))
    else:
        author, named = args.author, set()
    if manifest is not None and args.author == ap.get_default("author"):
        author = manifest.get("author") or next(
            (w.get("author") for w in manifest["works"] if w.get("author")), args.author)

    sources = list_sources(args.src, manifest)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        sys.exit("no PDFs under %s" % args.src)
    # A roster that misses a file would ingest it under a slug parsed from
    # another author's naming scheme, and the reader would meet a work nobody
    # named. Better to stop and say which.
    missing = sorted({os.path.basename(p) for p in sources} - named) if named else []
    if missing:
        sys.exit("%s names no entry for: %s" % (args.roster, ", ".join(missing)))
    # A file the roster HOLDS is named but not ingested. Bhai Vir Singh's twelve
    # PDFs are image scans whose OCR damage varies twelve-fold, and three of them
    # are past the point where the text is worth retrieving -- one reads "Goblnd
    # Hal" for the Guru's own name throughout. Holding is not deleting: the entry
    # stays, carrying the measured rate that decided it, so the omission is
    # visible where the roster is read and a better scan just replaces the file.
    held = []
    for path in list(sources):
        why = (roster.get("works", {}).get(os.path.basename(path)) or {}).get("hold")
        if why:
            held.append({"file": os.path.basename(path), "why": why})
            sources.remove(path)
    if held:
        print("held %d of %d files:" % (len(held), len(held) + len(sources)))
        for h in held:
            print("  %s -- %s" % (h["file"], h["why"]))
    if not sources:
        sys.exit("every PDF under %s is held by %s" % (args.src, args.roster))
    os.makedirs(args.out, exist_ok=True)
    # beside the corpus, not in one fixed place: a second author must not
    # overwrite the first author's gate report. A manifest's books share the
    # default folder with the essays, so their report is named for the author.
    if args.out != OUT_DIR:
        report_path = os.path.join(args.out, "report.json")
    elif manifest is not None:
        report_path = REPORT.replace(
            "writings-report.json", "writings-report-%s.json" % re.sub(r"[^a-z0-9]+", "-", author.lower()).strip("-"))
    else:
        report_path = REPORT
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    print("%d source(s), %d workers%s" % (
        len(sources), args.workers,
        ", manifest" if manifest else (", roster %s" % os.path.basename(args.roster) if args.roster else "")))

    # what the italic face means is the book's to say (rosters/rama.json): in
    # every book before that one it set the quoted verse, and that stays the
    # default
    read = functools.partial(read_one, manifest=manifest, roster=roster or None,
                             furniture=bool(roster.get("furniture")),
                             italic_quotes=bool(roster.get("italic_quotes", True)))
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, res in enumerate(pool.map(read, sources), start=1):
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
        original = all(m["original"] for m in metas)
        # a roster or manifest names the author per work: four of the AKJ books
        # are Bhai Sahib Bhai Randhir Singh Ji and the fifth is someone else,
        # and an answer must attribute each passage to the one who wrote it
        work_author = next((m["author"] for m in metas if m.get("author")), author)
        policy = (roster.get("quote_policy") or metas[0].get("quote_policy")
                  or ("verbatim" if original else "summarise"))
        head_meta = {"work": work, "title": title, "author": work_author, "original": original,
                     "quote_policy": policy, "files": sorted(m["file"] for m in metas),
                     "language": metas[0].get("language", "en"), "licence": metas[0].get("licence"),
                     "source": metas[0].get("reader") or "pdf-text"}
        with open(os.path.join(args.out, work + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"_meta": head_meta}, ensure_ascii=False) + "\n")
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        works.append({"work": work, "title": title, "author": work_author,
                      "parts": sorted(p for p in {m["part"] for m in metas} if p),
                      "files": len(metas), "folder": metas[0]["folder"],
                      "original": original, "quote_policy": policy,
                      "language": head_meta["language"], "licence": head_meta["licence"],
                      **measure(records)})

    # works.json holds every author's works in this folder: this run replaces
    # its own authors' entries and keeps the others, so a manifest's books and
    # the essays can share data/writings without one run erasing the other
    listing_path = os.path.join(args.out, "works.json")
    listing: dict = {"author": author, "works": []}
    if os.path.exists(listing_path):
        with open(listing_path, encoding="utf-8") as fh:
            listing = json.load(fh)
    mine = {w["author"] for w in works}
    kept = [w for w in listing.get("works", []) if w.get("author") not in mine]
    listing_works = kept + works
    with open(listing_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"author": listing.get("author") or author,
                   "authors": sorted({w["author"] for w in listing_works}),
                   "works": listing_works}, fh, ensure_ascii=False, indent=2)

    every = [r for recs in by_work.values() for r in recs]
    report = {"src": args.src, "author": author, "files": len(sources),
              "works": len(works), "failures": failures, "held": held,
              "rejoined_words": rejoined,
              "essay_number_mismatch": mismatched,
              "pages": sum(r.get("pages", 0) for r in results),
              "spread_pages": sum(r.get("spreads", 0) for r in results),
              "totals": measure(every)}
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
