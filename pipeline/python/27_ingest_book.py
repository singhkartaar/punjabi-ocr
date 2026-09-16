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
and then, for the folder:
  ingest     12_ingest_writings.py paragraphs -> data/writings/<work>.jsonl
  cite       13_resolve_citations.py   skipped, with a message, when there is no scripture DB
  embed, db  14_embed_writings.py --lang L; 15_build_writings_db.py     -> artifacts/writings-L/
  translate  26_translate_writings.py  Punjabi/Hindi -> English, locally (--no-translate skips)
  embed-en, db-en  14 --lang en --translations; 15                       -> artifacts/writings-en/

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
from lib.paths import CORPUS_DB, OCR_DIR, ROOT
from lib.writings_manifest import list_sources, load_manifest, parse_source

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["pages", "ocr", "gt", "eval", "merge", "ingest", "cite", "embed", "db", "translate", "embed-en", "db-en"]

# 21_ocr_run.py arguments for each engine name the driver knows; the Tesseract
# variants are what the Punjabi bake-off found clearing the 90% bar
# (docs/ocr-ingestion.md), so a Punjabi book runs both and lets the merge vote.
ENGINE_ARGS = {
    "tesseract": ["--engine", "tesseract"],
    "tesseract-pan": ["--engine", "tesseract", "--tess-lang", "pan"],
    "tesseract-gurmukhi": ["--engine", "tesseract", "--tess-lang", "script/Gurmukhi"],
    "tesseract-hin": ["--engine", "tesseract", "--tess-lang", "hin"],
    "pdftext": ["--engine", "pdftext"],
    "surya": ["--engine", "surya"],
    "dotsocr": ["--engine", "dotsocr"],
    "indicocr": ["--engine", "indicocr"],
}
DEFAULT_ENGINES = {"pa": ["tesseract-pan", "tesseract-gurmukhi"], "en": ["tesseract"], "hi": ["tesseract"]}


def script(name: str) -> str:
    return os.path.join(HERE, name)


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


def plan(src: str, metas: list[dict], *, engines: list[str] | None = None, gpu_engines: list[str] = (),
         gt: bool = False, translate: bool = True, translate_engine: str = "sarvam", ocr_dir: str = OCR_DIR,
         corpus_db: str = CORPUS_DB, pages: str | None = None) -> list[tuple[str, list[str] | str]]:
    """
    The (step, argv) sequence for these works. An argv that is a string is a
    message printed in place of a command (a skipped step says why).
    """
    out: list[tuple[str, list[str] | str]] = []
    ocr_books = [m for m in metas if (m.get("reader") or "ocr") == "ocr"]
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
        out.append(("merge", [script("22_ocr_merge.py"), "--book", book, "--out", ocr_dir] + page_args))
    if gt:
        out.append(("gt", "STOP: for each book, verify data/ocr/<book>/gt/candidates.jsonl against its crops, "
                          "promote it with 23_ocr_gt.py --book <book> --promote, then resume with --from eval"))
        return out
    out.append(("ingest", [script("12_ingest_writings.py"), "--src", src]))
    if has_rows(corpus_db):
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
        out.append(("embed", [script("14_embed_writings.py"), "--lang", lang]))
        out.append(("db", [script("15_build_writings_db.py"), "--units", os.path.join(ROOT, "data", "writings", units)]))
    foreign = [m for m in metas if m.get("language", "en") != "en"]
    if foreign and translate:
        seen = set()
        for m in foreign:
            if m["work"] in seen:
                continue
            seen.add(m["work"])
            out.append(("translate", [script("26_translate_writings.py"), "--work", m["work"],
                                      "--src-lang", m.get("language", "pa"), "--engine", translate_engine]))
        out.append(("embed-en", [script("14_embed_writings.py"), "--lang", "en", "--translations"]))
        out.append(("db-en", [script("15_build_writings_db.py")]))
    elif foreign:
        out.append(("translate", "skip: --no-translate; the %s text is searchable in its own language only"
                    % "/".join(lang for lang in langs if lang != "en")))
    return out


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
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--translate-engine", default="sarvam", choices=["sarvam", "indictrans2"])
    ap.add_argument("--from", dest="start", choices=STEPS, help="resume at this step")
    ap.add_argument("--to", dest="stop", choices=STEPS, help="stop after this step")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--out", default=OCR_DIR, help="the OCR working directory")
    args = ap.parse_args()

    metas = books_in(args.src, args.book)
    steps = plan(args.src, metas,
                 engines=[e.strip() for e in args.engines.split(",")] if args.engines else None,
                 gpu_engines=[e.strip() for e in args.gpu_engines.split(",") if e.strip()],
                 gt=args.gt, translate=not args.no_translate, translate_engine=args.translate_engine,
                 ocr_dir=args.out, pages=args.pages)
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
