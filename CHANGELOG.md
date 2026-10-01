# Changelog -- punjabi-ocr

## 1.3.0

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
- `33_notation_sample.py` reads a few pages of many books at random and
  writes one review page across them: the original page, the crops, the
  grid as read.
- Hindi books move to 1.3.

## 1.2.0

A book is linked to the scripture it explains, a two-column page is read as
verse beside its explanation, no line is dropped without a record of it,
and a new book is one command to a scorecard.

- **Voters**: a Tesseract variant beside a Tesseract pivot always votes; the
  90% bar is for engines of another family (`docs/engines.md`).
- **Manifest** (`docs/manifest.md`): `kind` (essay, translation,
  word-meaning, commentary, reference), `translate` per work, `layout`
  (auto, paired-columns, columns), `angs`, `header_pattern` (now
  implemented), `coverage`.
- **Output** (`docs/output-format.md`): a `links` table -- every relation
  between a passage and the scripture with its `source`, line range and
  `role` (`explains` or `quotes`); `works` gains `kind`, `translate`,
  `ang_from`, `ang_to`; `citations` keeps its shape and only Guru Granth
  Sahib rows. Records gain `pair`, `explains`, `source`, `line_from`,
  `line_to`; pages `layout` and `ang_source`.
- **Reader**: a two-column page is read in bands (`lib/ocr_pairs.py`); a
  verse and the prose beside it share a `pair`, the prose says which lines
  it `explains`; on a single-column page the prose after a verse explains
  it for a translation, word-meaning or commentary, when the verse lies
  inside the work's `angs` (a quotation from elsewhere explains nothing).
  `layout: columns` is the old reading. The embedder links a verse forward
  to what explains it. On the Santhya's first volume: 245 of 530 pages read
  as paired, 3,448 paragraphs explain a verse, and the corpus carries 910
  `explains` and 684 `quotes` links over 155 shabads of angs 1-53
  (`docs/design.md`, "Linked, measured").
- **Merge**: a hairline rule is the gutter (`_meta.columns_source`); an
  orphan is judged in its column; running-header hints are smoothed over
  the book (`_meta.hints`, `hints_raw`, `ang_source`); `--coverage` reads
  the ink the layout left uncovered, one crop at a time, recording every
  region and why (`_meta.coverage`, `recovered` on a line). On by default:
  on the Santhya it finds 441 of 440 counted printed lines where the merge
  without it found 406 and Tesseract alone 398, at unchanged word accuracy,
  with the recovered lines 89.4% right and no false recovery in 19 non-text
  crops (`docs/design.md`, "The coverage pass, measured"). A crop the
  segmenter reads as empty is read again in raw mode. `--out-name` writes a
  merge beside another for comparison.
- **Merge, besides**: the other engines' lines are cut at the gutter where
  the pivot's were, and a voter's wider line is cut to the pivot line's span,
  so a whole row is no longer voted onto half of it; a bar read out of the
  rule belongs to neither half; fragments of the rule the split leaves
  behind are dropped; a line's column is where its centre falls.
- **Correction on a grey scan**: the corrector compares a conjunct as one
  cluster and prices marks one by one, reads an aunkar or dulainkar stacked
  on a vowel sign as the subjoined ra it is, folds the nukta when it looks a
  word up (the Kosh's ਪ੍ਰਸਾਦ is the book's ਪ੍ਰਸ਼ਾਦ) and keeps the book's,
  breaks a tie by corpus frequency, corrects a word without its punctuation
  and a compound part by part, and no longer cuts its candidates at 1,000.
  The merge learns the book's own vocabulary from what two engines agree on,
  refusing a shared misreading. `correct_agreed` (manifest, `22
  --correct-agreed`) puts back a subjoined ra both engines lost. On the Sant
  Attar Singh biography's 120 dpi scan the words carrying a subjoined ra went
  from 23 right and 22 wrong to 42 and 3, unknown words from 40% to 14%, with
  no wrong correction on its ground truth; the Santhya is unchanged
  (`docs/design.md`, "A grey scan's conjuncts, measured"). A prose book needs
  the Mahan Kosh (`00_fetch_mahankosh.py`).
- **Evaluation**: `24 --merged a,b --eval-out` scores merges side by side,
  recovered lines apart from the page's (`by_provenance`), and line recall
  against a reviewer's count of the printed lines (`23 --count-pages`,
  `23 --coverage-sample`). Every correction the merge made on a ground-truth
  line is judged right, wrong or unsure (`corrections`).
- **Reader, besides**: an OCR line box within 12% of the page's body
  height is the body height (the paragraph rules were written for a PDF's
  exact font sizes and cut a grey scan's paragraphs at every line); a long
  bold line the merge left as prose is prose; a recovered line sits where
  its ink is, not where its crop was; a shabad quoted with its raag title
  is one quotation.
- **Embedding**: a corpus with fewer units than the index's 256 dimensions
  (a work of a few pages, a bench sample) keeps as many dimensions as it
  has units; the manifest's `index_dim` says which.
- **Translation**: `26` finds sarvam-translate's weights under
  `vendor/models` before asking the hub, loads them across the card and
  host memory when they do not fit (`--device auto`, the default), and
  retries a batch the card cannot hold one paragraph at a time.
- **Translation, terms**: a book's Sikh terms are data (`glossary` in the
  manifest, `examples/glossary.json`, `lib/mt_glossary.py`). `--prompt terms`
  (the default; the stock prompt when there is no glossary) writes each term
  a paragraph uses in English into the Punjabi, because sarvam-translate
  follows no instruction beyond its trained line; `--prompt rules` states
  them in the instruction, for engines that follow one. The model's
  bracketed explanations are stripped, never more brackets than the source
  has; a term rendered by none of its accepted forms is a refusal reason;
  `--arbiter self|llama|vertex` asks again, another way, for what was
  refused. On the Sant Attar Singh biography: all 30 bench paragraphs
  answered, 58 of 61 terms rendered against 57 before, chrF 55.0 against
  52.5 for the old translations, and "a canopy for the Guru's funeral"
  became "a canopy for Sri Guru Granth Sahib" (`docs/design.md`, "Terms, measured").
  `28_translate_bench.py` scores prompt variants as rows, a term hit rate,
  refusals by reason, and a reference a person writes (`--reference manual`);
  26 takes `--only` and `--out` for such runs.
- **Driver**: `27 --translate` / `--no-translate` override the works'
  `translate`; the English corpus is built only when some work wants it.
  The manifest's `glossary`, `prompt`, `arbiter` and `correct_agreed` reach
  the steps that use them.
- **Bench**: `29_bench_books.py --src <folder>` runs a sample of pages of
  every work and prints one scorecard row per book (`docs/design.md`, "The
  bench"); the runbook's "A first look" says how to read it.
- `GEMINI.md` and the contract tests are unchanged in intent; the contract
  tests now also state the `links` table and the `works` columns.

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

## 1.4 (planned)

- **Hindi.** The plumbing is in place (`language: "hi"`, Tesseract `hin`,
  Devanagari checks in the translator) and was measured only on a synthetic
  typeset page (98.2%). Real scans, ground truth and a lexicon are what 1.1 is for.
- A vocabulary-trimmed multilingual embedding model for the Punjabi and
  Hindi corpora, so their vectors ship as compactly as the English ones.
