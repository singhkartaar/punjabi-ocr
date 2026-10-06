"""
One command from a scanned book to searchable, translated text.

  27_ingest_book.py --src ./books --book santhya-vol-1              everything, in order
  27_ingest_book.py --src ./books --book santhya-vol-1 --dry-run    print the commands, run nothing
  27_ingest_book.py --src ./books --book santhya-vol-1 --from merge resume after a fix
  27_ingest_book.py --src ./books --gt                              sample ground truth, then stop

The numbered scripts each do one thing and are run by hand while a book is
being understood; this driver strings them together once a book is routine.
It reads the folder's manifest.json (lib/writings_manifest.py) for the book's
language, its key under data/ocr/ and its work name, and runs, per book:

  pages      20_ocr_pages.py       render every page at 300 dpi
  ocr        21_ocr_run.py         one run per engine (Tesseract variants by language; --gpu-engines adds more)
  gt         23_ocr_gt.py          only with --gt: sample lines to verify, then STOP -- a person reads the crops
  eval       24_ocr_eval.py        only when gt/lines.jsonl exists: measures each engine so the merge can weight it
  merge      22_ocr_merge.py       corpus match + vote + lexicon + correction -> merged/
and, for a work whose manifest says kind: notation (a keertan notation book), instead of ingest..db-en:
  notation      29_notation_parse.py     layout, shabad, crops -> data/notations/<book>/
  notation-gt   30_notation_gt.py --mid  only with --gt: the mid-book review window, then STOP
  notation-eval 31_notation_eval.py      only when gt/notation-gold.jsonl exists: fields against the gold
  notation-db   32_build_notations_db.py -> artifacts/notations.sqlite
and then, for the folder:
  ingest     12_ingest_writings.py paragraphs -> data/writings/<work>.jsonl
  cite       13_resolve_citations.py   skipped, with a message, when there is no scripture DB
  embed, db  14_embed_writings.py --lang L; 15_build_writings_db.py
                 -> artifacts/corpora/writings-L/ and artifacts/writings-L.sqlite
  translate  26_translate_writings.py  Punjabi/Hindi -> English, locally, for each work whose
             manifest `translate` is on (--translate: all of them; --no-translate: none)
  embed-en, db-en  14 --lang en --translations; 15
                 -> artifacts/corpora/writings-en/ and artifacts/writings.sqlite

A manifest `corpus` ("barusahib") puts the paragraphs in data/<corpus>/ beside
that corpus's roster works, and the indexes in <corpus>-<lang> and
artifacts/<corpus>.sqlite (English), <corpus>-<lang>.sqlite (otherwise).

A step that fails stops the run and names the --from value that resumes it.
Nothing here is clever: plan() is a pure function of the manifest and the
flags, so the sequence can be printed and tested without running anything.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paths import ARTIFACTS, CORPUS_DB, OCR_DIR, ROOT
from lib.notation import mid_window
from lib.writings_manifest import list_sources, load_manifest, parse_source
from lib.writings_works import db_name

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["pages", "ocr", "gt", "eval", "merge", "notation", "notation-gt", "notation-eval", "notation-db",
         "ingest", "cite", "embed", "db", "translate", "embed-en", "db-en"]

# 21_ocr_run.py arguments for each engine name the driver knows; the Tesseract
# variants are what the Punjabi bake-off found clearing the 90% bar
# (docs/ocr-ingestion.md), so a Punjabi book runs both and lets the merge vote.
ENGINE_ARGS = {
    "tesseract": ["--engine", "tesseract"],
    "tesseract-pan": ["--engine", "tesseract", "--tess-lang", "pan"],
    "tesseract-gurmukhi": ["--engine", "tesseract", "--tess-lang", "script/Gurmukhi"],
    # psm 4 (one column of variable-sized text) keeps the sparse rows of a
    # notation grid that psm 3's layout analysis drops; a notation book runs both
    "tesseract-pan-psm4": ["--engine", "tesseract", "--tess-lang", "pan", "--psm", "4"],
    "tesseract-hin": ["--engine", "tesseract", "--tess-lang", "hin"],
    "pdftext": ["--engine", "pdftext"],
    "surya": ["--engine", "surya"],
    "dotsocr": ["--engine", "dotsocr"],
    "indicocr": ["--engine", "indicocr"],
}
DEFAULT_ENGINES = {"pa": ["tesseract-pan", "tesseract-gurmukhi"], "en": ["tesseract"], "hi": ["tesseract"],
                   "pa-notation": ["tesseract-pan", "tesseract-pan-psm4", "tesseract-gurmukhi"]}
# The review window of a notation book (docs/notations.md): pages from about
# 45% of the book, five to seven of them, covering at least two shabads.
NOTATION_GT_PAGES = 7
# Whether the merge looks for ink its lines left uncovered and reads it
# (22_ocr_merge.py --coverage). On since the measurement on the Santhya
# (docs/ocr-ingestion.md): 441 of 440 counted printed lines found against
# 406 without it, word accuracy unchanged, recovered lines 89.4% right, no
# false recovery in 19 non-text crops, 0.47 s a page. A manifest's
# `coverage: false` or --no-coverage turns it off for a book.
COVERAGE_DEFAULT = True


def script(name: str) -> str:
    return os.path.join(HERE, name)


def pdf_pages(path: str) -> int:
    """The page count of a PDF, 0 when it cannot be read (the window then covers the whole book)."""
    if not path or not os.path.exists(path):
        return 0
    try:
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


def has_rows(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def books_in(src: str, only: str | None) -> list[dict]:
    """The works of a folder as parse_source() metas, optionally one book."""
    manifest = load_manifest(src)
    if manifest is None:
        sys.exit("%s has no manifest.json; see docs/ocr-runbook.md for the shape" % src)
    metas = [parse_source(p, manifest) for p in list_sources(src, manifest)]
    if only:
        metas = [m for m in metas if m["book"] == only]
        if not metas:
            sys.exit("no work with book %r in %s/manifest.json" % (only, src))
    return metas


NOTATION_LOOKAHEAD = 2      # pages read past a window, so a notation that runs over the window's end is whole
NOTATION_FRONT_PAGES = 20   # the first pages, read with any window: the index there names the numbered notations


def widen_pages(spec: str, by: int = 1, after: int | None = None) -> str:
    """
    '165-176,200' -> '164-177,199-201': each range grown by `by` pages before
    and `after` (default `by`) pages after, never below 1.
    """
    after = by if after is None else after
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = (part.split("-", 1) + [part])[:2] if "-" in part else (part, part)
        lo, hi = max(1, int(a) - by), int(b) + after
        out.append("%d-%d" % (lo, hi))
    return ",".join(out)


def plan(src: str, metas: list[dict], *, engines: list[str] | None = None, gpu_engines: list[str] = (),
         gt: bool = False, translate: bool | None = None, translate_engine: str = "sarvam", ocr_dir: str = OCR_DIR,
         corpus_db: str = CORPUS_DB, pages: str | None = None, coverage: bool | None = None,
         out_name: str | None = None, correct_agreed: bool | None = None) -> list[tuple[str, list[str] | str]]:
    """
    The (step, argv) sequence for these works. An argv that is a string is a
    message printed in place of a command (a skipped step says why).

    `coverage` None takes each book's manifest key, else COVERAGE_DEFAULT;
    `out_name` writes the merge to <book>/<out_name>/ for a comparison.
    `correct_agreed` None takes each book's `correct_agreed` key (default off).
    `translate` None takes each work's manifest key (`translate`, on unless
    the manifest says otherwise for a work not in English): a work read for
    display in its own language stops at the Punjabi corpus, a work that is
    to be searched in English goes on through 26; True translates every
    foreign work, False none (--translate, --no-translate).
    """
    out: list[tuple[str, list[str] | str]] = []
    ocr_books = [m for m in metas if (m.get("reader") or "ocr") == "ocr" and m.get("kind") != "notation"]
    notation_books = [m for m in metas if m.get("kind") == "notation"]
    for m in notation_books:
        book, lang = m["book"], m.get("language", "pa")
        window = pages
        if gt and not window:
            n = pdf_pages(os.path.join(src, m.get("file") or ""))
            if n:
                a, b = mid_window(n, NOTATION_GT_PAGES)
                window = "%d-%d" % (a, b)
        # a window is READ two pages past its end (a notation that begins on its last page
        # runs on), and RENDERED a page wider still on each side: the review page shows the
        # page before and after a notation, so a reader can see nothing was cut off
        read = widen_pages(window, 0, NOTATION_LOOKAHEAD) if window else None
        if read and not read.startswith("1-"):
            # the index, at the front or the back of the book, names the numbered notations and the raag starts
            n_all = pdf_pages(os.path.join(src, m.get("file") or ""))
            back = ",%d-%d" % (max(1, n_all - NOTATION_FRONT_PAGES + 1), n_all) if n_all > 2 * NOTATION_FRONT_PAGES else ""
            read = "1-%d,%s%s" % (NOTATION_FRONT_PAGES, read, back)
        page_args = ["--pages", read] if read else []
        render_args = ["--pages", widen_pages(read)] if read else []
        out.append(("pages", [script("20_ocr_pages.py"), "--src", src, "--book", book, "--out", ocr_dir] + render_args
                    + (["--bleed"] if m.get("bleed") else [])))
        names = list(engines or DEFAULT_ENGINES.get(lang + "-notation") or DEFAULT_ENGINES.get(lang, ["tesseract"]))
        for name in names:
            if name not in ENGINE_ARGS:
                sys.exit("unknown engine %r; one of %s" % (name, ", ".join(sorted(ENGINE_ARGS))))
            out.append(("ocr", [script("21_ocr_run.py"), "--book", book, "--lang", lang, "--out", ocr_dir]
                        + ENGINE_ARGS[name] + page_args))
        out.append(("merge", [script("22_ocr_merge.py"), "--book", book, "--out", ocr_dir] + page_args))
        out.append(("notation", [script("29_notation_parse.py"), "--book", book] + page_args))
        if gt:
            out.append(("notation-gt", [script("30_notation_gt.py"), "--book", book, "--mid"]))
            continue
        gold = os.path.join(ocr_dir, book, "gt", "notation-gold.jsonl")
        if os.path.exists(gold):
            out.append(("notation-eval", [script("31_notation_eval.py"), "--book", book]))
        else:
            out.append(("notation-eval", "skip: no %s; run with --gt, review gt/notation-review.html and promote "
                                         "the gold before the whole book is built" % gold))
    if notation_books and gt:
        out.append(("notation-gt", "STOP: for each book, open data/ocr/<book>/gt/notation-review.html, judge every "
                                   "notation against its crops, download the candidates, then 30_notation_gt.py "
                                   "--book <book> --promote and 31_notation_eval.py --book <book>"))
        if not ocr_books:
            return out
    elif notation_books:
        book_args = [arg for m in notation_books for arg in ("--book", m["book"])]
        out.append(("notation-db", [script("32_build_notations_db.py")] + book_args + ["--allow-unmeasured"]
                    + (["--gurbani", corpus_db] if has_rows(corpus_db) else [])))
    if not ocr_books:
        return out
    for m in ocr_books:
        book, lang = m["book"], m.get("language", "en")
        page_args = ["--pages", pages] if pages else []
        out.append(("pages", [script("20_ocr_pages.py"), "--src", src, "--book", book, "--out", ocr_dir] + page_args
                    + (["--bleed"] if m.get("bleed") else [])))
        names = list(engines or DEFAULT_ENGINES.get(lang, ["tesseract"]))
        names += [e for e in gpu_engines if e not in names]
        for name in names:
            if name not in ENGINE_ARGS:
                sys.exit("unknown engine %r; one of %s" % (name, ", ".join(sorted(ENGINE_ARGS))))
            out.append(("ocr", [script("21_ocr_run.py"), "--book", book, "--lang", lang, "--out", ocr_dir]
                        + ENGINE_ARGS[name] + page_args))
        gt_path = os.path.join(ocr_dir, book, "gt", "lines.jsonl")
        if gt:
            out.append(("gt", [script("23_ocr_gt.py"), "--book", book, "--sample", "30", "--out", ocr_dir]))
            continue
        if os.path.exists(gt_path):
            out.append(("eval", [script("24_ocr_eval.py"), "--book", book, "--out", ocr_dir]))
        else:
            out.append(("eval", "skip: no %s, so every engine votes with the same weight; "
                                "run with --gt to measure them" % gt_path))
        cover = coverage if coverage is not None else bool(m.get("coverage", COVERAGE_DEFAULT))
        out.append(("merge", [script("22_ocr_merge.py"), "--book", book, "--out", ocr_dir] + page_args
                    + (["--coverage"] if cover else []) + (["--out-name", out_name] if out_name else [])
                    + (["--correct-agreed"] if (correct_agreed if correct_agreed is not None
                                                else m.get("correct_agreed")) else [])))
    if gt:
        out.append(("gt", "STOP: for each book, verify data/ocr/<book>/gt/candidates.jsonl against its crops, "
                          "promote it with 23_ocr_gt.py --book <book> --promote, then resume with --from eval"))
        return out
    # a manifest `corpus` sends the books into another corpus's folder, beside
    # its roster's works (Baru Sahib): every step then names that folder, the
    # citations are resolved for these works only (13 --work keeps the
    # roster's), and the indexes are <corpus>-<lang>
    corpus = next((m["corpus"] for m in metas if m.get("corpus")), None)
    folder = os.path.join(ROOT, "data", corpus) if corpus else os.path.join(ROOT, "data", "writings")
    works = list(dict.fromkeys(m["work"] for m in metas))
    out.append(("ingest", [script("12_ingest_writings.py"), "--src", src]
                + (["--out", folder] if corpus else [])))
    if has_rows(corpus_db):
        if corpus:
            for w in works:
                out.append(("cite", [script("13_resolve_citations.py"), "--src", folder, "--work", w]))
        else:
            out.append(("cite", [script("13_resolve_citations.py")]))
    else:
        out.append(("cite", "skip: no scripture database at %s (CORPUS_DB), so quotations are not resolved; "
                            "the OCR merge's own matches, if any, are kept" % corpus_db))
    langs: list[str] = []
    for m in metas:
        if m.get("language", "en") not in langs:
            langs.append(m.get("language", "en"))
    for lang in langs:
        units = "units.jsonl" if lang == "en" else "units-%s.jsonl" % lang
        out.append(("embed", [script("14_embed_writings.py"), "--lang", lang]
                    + (["--src", folder, "--corpus", "%s-%s" % (corpus, lang)] + keep(corpus, lang) if corpus else [])))
        out.append(("db", [script("15_build_writings_db.py"), "--units", os.path.join(folder, units)]))
    foreign = [m for m in metas if m.get("language", "en") != "en"]
    if translate is None:
        wanted = [m for m in foreign if m.get("translate", True)]
    else:
        wanted = foreign if translate else []
    if wanted:
        seen = set()
        for m in wanted:
            if m["work"] in seen:
                continue
            seen.add(m["work"])
            # the book's terms, prompt and arbiter from the manifest (lib/mt_glossary; 26 --help)
            extra = (["--glossary", os.path.join(src, m["glossary"])] if m.get("glossary") else []) \
                + (["--prompt", m["prompt"]] if m.get("prompt") else []) \
                + (["--arbiter", m["arbiter"]] if m.get("arbiter") else [])
            out.append(("translate", [script("26_translate_writings.py"), "--work", m["work"],
                                      "--src-lang", m.get("language", "pa"), "--engine", translate_engine] + extra
                        + (["--src", folder] if corpus else [])))
        stay = sorted({m["work"] for m in foreign} - seen)
        if stay:
            out.append(("translate", "%s stay in their own language (manifest translate: false)" % ", ".join(stay)))
        out.append(("embed-en", [script("14_embed_writings.py"), "--lang", "en", "--translations"]
                    + (["--src", folder, "--corpus", corpus + "-en"] + keep(corpus, "en") if corpus else [])))
        out.append(("db-en", [script("15_build_writings_db.py")]
                    + (["--units", os.path.join(folder, "units.jsonl")] if corpus else [])))
    elif foreign:
        why = ("--no-translate" if translate is False
               else "the manifest says translate: false for %s" % ", ".join(sorted({m["work"] for m in foreign})))
        out.append(("translate", "skip: %s; the %s text is searchable in its own language only "
                    "(set `translate` per work in the manifest, or pass --translate)"
                    % (why, "/".join(lang for lang in langs if lang != "en"))))
    return out


def keep(corpus: str, lang: str) -> list[str]:
    """
    14 --keep for a corpus already built: its works whose paragraphs are not
    in the folder -- a roster's PDFs read on another machine -- are carried
    over from its database as built, and only the rest is read here.
    """
    db = os.path.join(ARTIFACTS, db_name("%s-%s" % (corpus, lang)))
    return ["--keep", db] if os.path.exists(db) else []


def render(argv: list[str]) -> str:
    def q(a: str) -> str:
        return '"%s"' % a if " " in a else a
    return " ".join(q(os.path.relpath(a, HERE) if a.startswith(HERE) else a) for a in argv)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="folder with manifest.json and the PDFs")
    ap.add_argument("--book", help="one work's book key; default every OCR work in the manifest")
    ap.add_argument("--engines", help="comma list replacing the language default (%s)"
                    % "; ".join("%s: %s" % (k, ",".join(v)) for k, v in DEFAULT_ENGINES.items()))
    ap.add_argument("--gpu-engines", default="",
                    help="comma list added to the default, e.g. dotsocr (slow: minutes a page)")
    ap.add_argument("--pages", help="'1-20,35' for a first look at a few pages")
    ap.add_argument("--gt", action="store_true", help="sample ground truth for the book(s) and stop")
    ap.add_argument("--translate", dest="translate", action="store_true", default=None,
                    help="translate every work not in English; default: each work's manifest `translate`")
    ap.add_argument("--no-translate", dest="translate", action="store_false", help="translate none of them")
    ap.add_argument("--translate-engine", default="sarvam", choices=["sarvam", "indictrans2"])
    ap.add_argument("--from", dest="start", choices=STEPS, help="resume at this step")
    ap.add_argument("--to", dest="stop", choices=STEPS, help="stop after this step")
    ap.add_argument("--coverage", dest="coverage", action="store_true", default=None,
                    help="the merge reads the ink its lines left uncovered (default: the manifest's `coverage` key)")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--out-name", help="write the merge to <book>/<name>/ (merged-<x>) for a side-by-side comparison")
    ap.add_argument("--correct-agreed", dest="correct_agreed", action="store_true", default=None,
                    help="the merge also corrects words every engine misread alike (default: the manifest's "
                         "`correct_agreed` key, else off)")
    ap.add_argument("--no-correct-agreed", dest="correct_agreed", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory")
    args = ap.parse_args()

    metas = books_in(args.src, args.book)
    steps = plan(args.src, metas,
                 engines=[e.strip() for e in args.engines.split(",")] if args.engines else None,
                 gpu_engines=[e.strip() for e in args.gpu_engines.split(",") if e.strip()],
                 gt=args.gt, translate=args.translate, translate_engine=args.translate_engine,
                 ocr_dir=args.out, pages=args.pages, coverage=args.coverage, out_name=args.out_name,
                 correct_agreed=args.correct_agreed)
    lo = STEPS.index(args.start) if args.start else 0
    hi = STEPS.index(args.stop) if args.stop else len(STEPS) - 1
    steps = [(s, a) for s, a in steps if lo <= STEPS.index(s) <= hi]

    print("%d step(s) for %s" % (len(steps), ", ".join("%s (%s)" % (m["book"], m.get("language", "en")) for m in metas)))
    for step, argv in steps:
        if isinstance(argv, str):
            print("  [%s] %s" % (step, argv))
            if argv.startswith("STOP") and not args.dry_run:
                return
            continue
        print("  [%s] python %s" % (step, render(argv)))
        if args.dry_run:
            continue
        code = subprocess.call([sys.executable] + argv, cwd=HERE)
        if code != 0:
            sys.exit("step %s failed (exit %d); fix it and resume with --from %s" % (step, code, step))
    if not args.dry_run:
        print("done")


if __name__ == "__main__":
    main()
