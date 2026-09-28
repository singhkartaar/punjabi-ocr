# Working in this repository

Instructions for an AI coding assistant (and a fair summary for a person).
`CONTRIBUTING.md` has the project's principles; this page is the rules that
keep a change usable.

## What this is

A Python pipeline that turns scanned or born-digital books in Punjabi or
English into a searchable, translated corpus. `README.md` has the overview,
`docs/runbook.md` the commands, `docs/output-format.md` the output contract.

The repository is **exported from a larger private one, file for file at the
same paths**. Changes made here are carried back there as patches, and the
next export replaces this repository's files with that one's. So a change
survives only if it can be applied there.

## Rules

1. **Keep every file where it is, under the name it has.** No renames, no
   moves, no splitting a script in two, no new top-level folders. A new
   module goes in `pipeline/python/lib/`; say so in the commit message,
   because the export's file list has to learn about it.
2. **Change only the lines the task needs.** No reformatting, no import
   reordering, no quote-style or line-length passes, no type-annotation
   sweeps. A formatter run over a file makes its patch unusable.
3. **Do not change the output contract** (`docs/output-format.md`):
   the five vector file names and their data types, the manifest's fields,
   the tables and columns of the database, the keys of a paragraph record,
   and `unit_row` being the row in `units.i8`. You may ADD a column, a field
   or a key, after the existing ones. `OutputContractTests` in
   `pipeline/python/test_writings.py` fails if the contract moves; if it
   fails, change the code back, not the test.
4. **Do not change what a command-line flag means, or remove one.** Add a
   new flag with a default that keeps today's behaviour.
5. **Scripture is found, never corrected.** A quoted line is matched against
   a corpus and replaced by the corpus text, or left alone. Nothing may
   "fix" a verse, and quoted scripture is never machine-translated.
6. **No paid call without a budget.** Anything that costs money goes through
   `lib/ocr_route.Budget`: a cap, a ledger, and `--dry-run`.
7. **Commit code only.** Never add a book, a page image, OCR output, a
   database, model weights or a `manifest.json` that names real files. They
   live under `data/`, `artifacts/` and `vendor/`, which are ignored; do not
   use `git add -f`, and do not put such files anywhere else. This
   repository is public.
8. **Nothing personal in the code**: no local paths, account names, tokens
   or keys, in code, comments, tests or commit messages.

## Before every commit

```bash
cd pipeline/python
python -m unittest test_ocr test_writings
python 27_ingest_book.py --src <a folder with a manifest.json> --dry-run
```

Both suites need no scan, no engine binary, no model and no GPU. A test that
reads a rendered book or a scripture database skips itself when they are
absent. If a test fails, the change is not finished.

## How to make a change

- Work on a branch, one change per commit, and say in the commit message
  what changed and why.
- A change to OCR, merging, correction or translation comes with a number:
  the before and after from `24_ocr_eval.py` or `28_translate_bench.py` on
  ground truth (`data/raw/ocr-eval-<book>.json`). Put the numbers in the
  commit message. An improvement that was not measured is a guess.
- New behaviour gets a test in `test_ocr.py` or `test_writings.py`, built
  from synthetic data in the style of the tests already there.
- Match the code around you: its naming, its comment density, plain
  `argparse` and the standard library where they do the job. Comments say
  why, not what.
- Python 3.12. New dependencies go in `requirements-ocr.txt` with a comment
  saying which script needs them, and only when nothing already installed
  will do.

## When running the pipeline on real books

- Keep everything under `data/` after a run. `data/ocr/<book>/` and
  `data/writings/<work>.jsonl` are hours of work; the last two steps
  (`14_embed_writings.py`, `15_build_writings_db.py`) can be run again from
  them in minutes.
- Start with `--pages 1-20`, then `--gt` to measure, before a whole book.
- `--dry-run` first on anything that might call a paid service.
