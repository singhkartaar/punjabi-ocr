# Changelog -- punjabi-ocr

## 1.0.0

The first release: scanned or born-digital **Punjabi and English** books to a
documents index the [gurbani-search-api](https://github.com/gurmukhi-repo/gurbani-search-api)
server serves, with local English translation of the Punjabi.

- Rendering (300 dpi, deskew, border crop, show-through suppression) and OCR
  with Tesseract 5 (`pan`, `script/Gurmukhi`, `eng`), Surya 2, dots.ocr,
  IndicOCR and Google Vision behind one adapter shape.
- Quoted scripture found in a corpus and tagged with its ids, never corrected;
  the rest voted across engines, gated by a lexicon, corrected only where
  the engines disagree. An engine measured under 90% word accuracy does not vote.
- Ground-truth sampling with crops and a review page, and an evaluation that
  reports CER, word accuracy and citation precision per engine.
- Paragraphing, retrieval units, int8 vectors with a saved PCA, and a SQLite
  of passages with their citations, in the search API's index format.
- Translation with sarvam-translate (default), IndicTrans2, NLLB, MADLAD or
  any instruction model behind llama-server, checked before it is written,
  and a benchmark against a paid reference that costs cents.
- One driver, `27_ingest_book.py`, that runs the whole sequence from a
  manifest, prints it with `--dry-run`, and resumes with `--from`.

Measured on the books it was built with: Punjabi commentary merged at 94.9%
word accuracy with every quoted verse matched and none falsely, English at
99.1%; see `docs/engines.md`.

## 1.1 (planned)

- **Hindi.** The plumbing is in place (`language: "hi"`, Tesseract `hin`,
  Devanagari checks in the translator) and was measured only on a synthetic
  typeset page (98.2%). Real scans, ground truth and a lexicon are what 1.1 is for.
- A vocabulary-trimmed multilingual embedding model for the Punjabi and
  Hindi corpora, so their vectors ship as compactly as the English ones.
