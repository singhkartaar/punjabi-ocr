# Changelog -- punjabi-ocr

## 1.1.0

The output takes the shape the search API actually serves, and the shared
ingestion scripts catch up with a fortnight of work on the books.

- **Changed: where a corpus lands.** The vectors go to
  `artifacts/corpora/<corpus>/` and the passages to `artifacts/<key>.sqlite`
  (`writings-en` -> `writings.sqlite`, `writings-pa` -> `writings-pa.sqlite`),
  which is the pair `gurbani-search-api`'s `CORPORA` table and its
  `writings-<key>` data packs are built on. 1.0.0 wrote a self-contained
  `artifacts/writings-<lang>/` with `units.sqlite` inside, for a
  `/api/documents` route that was never published. `docs/output-format.md`
  describes the new layout; `14_embed_writings.py --out-dir` and
  `15_build_writings_db.py --out` override it.
- `14_embed_writings.py` cuts a paragraph longer than the cap at its own
  sentence ends instead of handing it whole to the embedder; a heading no
  longer stands alone as a unit, and the verses under a discarded heading are
  carried to the prose that follows. A paragraph may cite several verses.
- `13_resolve_citations.py` reads `(SGGS p. 920)`-style citations, joins a
  quotation a page break cut in two, and can restyle a paragraph that is
  nothing but a resolved quotation (`--restyle`). Scripture the OCR merge
  already matched is still taken as given.
- `12_ingest_writings.py` reads a folder through a manifest (as before) or
  through a roster JSON (`--roster`) that names each file's work, title and
  author and can keep only a span of a file; `--out` puts a corpus in its own
  directory, and `13`/`14` take the same `--src`.
- `pypdf` is a listed dependency; `--text-lang` is kept as an alias of
  `--lang`.

## 1.0.0

The first release: scanned or born-digital **Punjabi and English** books to a
documents index the [gurbani-search-api](https://github.com/singhkartaar/gurbani-search-api)
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

## 1.2 (planned)

- **Hindi.** The plumbing is in place (`language: "hi"`, Tesseract `hin`,
  Devanagari checks in the translator) and was measured only on a synthetic
  typeset page (98.2%). Real scans, ground truth and a lexicon are what 1.1 is for.
- A vocabulary-trimmed multilingual embedding model for the Punjabi and
  Hindi corpora, so their vectors ship as compactly as the English ones.
