# Working in this repository

Instructions for an AI coding assistant (and a fair summary for a person).
`CONTRIBUTING.md` has the project's principles; this page is the rules that
keep a change usable.

## What this is

A Python pipeline that turns scanned or born-digital books in Punjabi or
English into a searchable, translated corpus. `README.md` has the overview,
`docs/runbook.md` the commands, `docs/output-format.md` the output contract.
**Read `docs/design.md` before changing the merge, the zones, the matching or
the translation checks**: most of its decisions record something that was
tried the other way first, with the number that settled it.

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
python -m unittest test_ocr test_writings test_notation
# test_notation covers the keertan notation route (lib/notation*.py, scripts 29-32);
# its renderer expectations are regenerated only with: python test_notation.py --write-expected
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

## Keertan notation books (the whole-book runs)

The notation route (`docs/notations.md` in the private repository;
here `docs/runbook.md` §5a and `docs/output-format.md`) is being run
book by book on this machine. What the agent does here, and does not:

- **Work on the series branch** (`music-notations-sangeet-sagar` for
  Prin. Dyal Singh's Gurmat Sangeet Sagar 1-4, `music-notations-tara-singh`
  for Prof Tara Singh's ratnavalis), never on `main`. Pull before a run.
- **The data is not in this repository.** Records, ledgers and fixtures
  live in the private data repository cloned beside this one (the
  `OCR_DIR`, `NOTATIONS_DIR`, `REVIEW_DIR`, `ARTIFACTS_DIR`, `CORPUS_DB`
  variables point into it); commit and push *there* after a run or a
  review (`git add -A && git commit -m "<book>: whole book" && git push`).
  Renders, OCR output and crops are ignored there and remade.
- **The loop per book**, from `pipeline/python` with the venv active:
  ```
  python 27_ingest_book.py --src <books>/<author> --book <book-key> --dry-run
  python 27_ingest_book.py --src <books>/<author> --book <book-key>
  python 34_notation_review.py serve --book <book-key>          # the person reviews; verdicts save as given
  python 34_notation_review.py check --book <book-key> --strict # must be green after any change to the reader
  python 34_notation_review.py status
  ```
  `21_ocr_run.py` and `29_notation_parse.py` use all cores but two; for
  several books at once set `NOTATION_WORKERS` (and `--workers` on 21)
  so they share the machine.
- **Never edit a record, a ledger line or a fixture by hand.** A wrong
  cut is a backlog comment through the review page; a reader fix is a
  rule in `lib/notation_layout.py` with a test in `LinkerRuleTests`
  (`test_notation.py`), written from the cached layout
  (`data/ocr/<book>/notation/pages/NNNN.json`) and the merged lines, as
  small as the case allows. After it: `python -m unittest test_notation`,
  then `34 check --strict` on **every** book that has a ledger -- an
  accepted notation that moves is a regression unless the reviewer's
  own note on it asked for the move (the page shows those again).
- **Flags on a card are facts, not faults.** `long-span` is a notation
  over more than five pages; `continues-next-page` and
  `continued-from-prev` mean the pages read ended or began inside it (a
  sampled window, never a whole book).
- **Commit code on the series branch**, one rule per commit, the rule's
  reason and the card that showed it in the message; the private
  repository ports it back and re-exports. Do not touch the export's file
  set (rule 1), the output contract (rule 3) or the schema
  (`lib/notation.schema.json`, `lib/notation_columns.json`) without
  saying so in the commit, since the app's JavaScript twins must change
  with them.

## When running the pipeline on real books

- Keep everything under `data/` after a run. `data/ocr/<book>/` and
  `data/writings/<work>.jsonl` are hours of work; the last two steps
  (`14_embed_writings.py`, `15_build_writings_db.py`) can be run again from
  them in minutes.
- Start with `--pages 1-20`, then `--gt` to measure, before a whole book.
- `--dry-run` first on anything that might call a paid service.
