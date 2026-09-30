# Changelog -- punjabi-ocr

## 1.2.0

Keertan notation books. A work declared `kind: notation` (a book of
shabads set to raag and taal in Bhatkhande notation) takes its own route
after the OCR merge: `29_notation_parse.py` reads each page's layout,
names the shabad from the corpus and the printed reference, and cuts 1-bit
crops of every grid; `30_notation_gt.py --mid` writes the mid-book review
page; `31_notation_eval.py` measures the fields against the reviewer's gold;
`32_build_notations_db.py` builds `artifacts/notations.sqlite`. The
contract (`lib/notation.schema.json`, `lib/notation.py`), the vocabulary of
raags, taals and symbols (`lib/notation_vocab.json`) and the renderer
(`lib/notation_render.py`) ship with fixtures that pin them.

- `27_ingest_book.py` plans the notation steps for such a work and, with
  `--gt`, the review window (seven pages from the middle of the book).
- `21_ocr_run.py` names a non-default page segmentation mode in the engine
  key (`tesseract-pan-psm4`); `22_ocr_merge.py` has a notation mode;
  `TesseractEngine.recognise_region` reads one region of a page.
- The manifest gains `kind` and `style` (docs/manifest.md); pages.json
  carries them and the work's title.
- Hindi books move to 1.3.

## 1.1.3

`reader: legacy-font` works here. A PDF typed in a pre-Unicode Gurmukhi font
(GurbaniAkhar, AnmolLipi, GurbaniLipi and their kin) has a real text layer
made of that keyboard's keystrokes; `lib/legacy_font.py` turns them into
Unicode, so such a book is read exactly, with no OCR. Until now the reader
needed a helper this repository did not carry, and said so.

- The converter is the ASCII-to-Unicode half of anvaad-js 1.5.1 (Khalis
  Foundation, MIT) in Python, its table unchanged. It returns what that
  library returns on the library's own 52 examples and on 300,000 random
  strings, with one deliberate difference: where that library writes the
  word "undefined" (a sihari before a character its table lacks), this keeps
  the character.
- `NOTICE.md` names the two pieces of code carried from elsewhere, and no
  longer says what a server does with a work's licence.

## 1.1.2

Documentation only.

- `docs/design.md`: the design reference -- what each step reads and writes,
  every engine and translator measured on ground truth, and the decisions
  those numbers led to (an engine below the bar does not vote; wrapped verse
  is matched as a stream; a gutter must be ink-free; vote weights come from
  the measurement; and the rest).
- `docs/engines.md` records Google Vision's measurement (88.6% on the
  Santhya), which it had listed as not yet measured.

## 1.1.1

No change to what the pipeline does; two things that keep it from changing
by accident.

- `OutputContractTests` (`pipeline/python/test_writings.py`) builds a small
  corpus end to end, with a stand-in for the embedding model, and states what
  a reader of it depends on: the five vector files and their sizes, the
  manifest's fields, the database's tables and columns and their order,
  `unit_row` as the row in the vectors, where a citation lands, the default
  locations, and the keys of a paragraph record. Adding passes; renaming,
  removing or reordering fails.
- `GEMINI.md`: the rules for an AI coding assistant working here -- keep
  paths and names, change only what the task needs, leave the contract and
  the flags alone, commit code only, measure before claiming.

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
