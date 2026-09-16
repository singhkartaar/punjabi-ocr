# Contributing

This repository is exported from a larger private one, file for file at the
same paths, so a patch here applies there with `git apply`. Keep the layout
(`pipeline/python/...`) and the LF line endings, and a pull request can be
carried across unchanged.

**Run the tests before you send anything.** They need no scan, no engine and
no model:

```bash
cd pipeline/python
python -m unittest test_ocr test_writings
```

**Measured, not asserted.** An OCR engine, a correction rule or a translator
earns its place by a number on ground truth (`23_ocr_gt.py`, `24_ocr_eval.py`,
`28_translate_bench.py`). A change to the merge should come with the before
and after on a book's `data/raw/ocr-eval-<book>.json`; a new engine with a row
in `docs/engines.md`.

**Scripture is found, never corrected.** A quoted line is matched against a
corpus and replaced by the corpus text or left alone; nothing in the pipeline
"fixes" a verse. Keep it that way.

**No paid call without a budget.** Anything that costs money goes through
`lib/ocr_route.Budget`: a cap, a ledger, and `--dry-run`.

**Nothing personal in the code.** The export refuses files that name a
private path or account; sample books stay out of the repository.

Please open an issue before a large change so the design can be talked through.
