# punjabi-ocr

Scanned or born-digital books in **Punjabi or English** in (Hindi in 1.2); a
corpus of passages, **searchable by meaning** and **translated into
English**, out. Runs on one machine, with open-weight models, and sends
nothing to a paid service unless you set a budget for it.

Version 1.1.0 -- `CHANGELOG.md`. Step by step: `docs/runbook.md`.

```
a PDF of scanned pages
  -> rendered at 300 dpi, deskewed                    20_ocr_pages.py
  -> read by several OCR engines                      21_ocr_run.py    (Tesseract; dots.ocr, Surya, IndicOCR; Google Vision under a cap)
  -> quoted scripture found in a corpus, not corrected;
     the rest voted, gated by a lexicon, corrected     22_ocr_merge.py  (scripture matching is optional)
  -> paragraphs, one file per work                    12_ingest_writings.py
  -> embedded, quantised, and a SQLite of passages    14_embed_writings.py, 15_build_writings_db.py
  -> translated into English, locally                 26_translate_writings.py  (sarvam-translate or IndicTrans2)
  -> artifacts/corpora/<corpus>/ + artifacts/<key>.sqlite   a prose corpus, the shape gurbani-search-api serves
```

One command runs all of it for a book:

```bash
cd pipeline/python
python 27_ingest_book.py --src ./books --book my-book --dry-run   # prints the commands
python 27_ingest_book.py --src ./books --book my-book             # runs them
```

## What it was measured on

The pipeline was built on Punjabi religious commentary (Bhai Vir Singh's
*Santhya*, 530 pages of 1-bit scans; Prof. Sahib Singh's grammar) and English
scans of the same period (Prof. Puran Singh), against 29-30 hand-verified lines
per book:

| book | best single engine | merged | notes |
|---|---|---|---|
| Santhya (pa, 300 dpi, 1-bit) | Tesseract `pan` 95.8% word accuracy | **94.9%** with every quoted verse matched to the corpus, 0 false matches | commentary CER 2.5% |
| Gurbani Vayakaran (pa, typeset) | Tesseract `script/Gurmukhi` | **92.0%** | |
| Ten Masters (en, 100 dpi, show-through) | Tesseract `eng` 99.1% | **99.1%** | `bleed: true` in the manifest |
| a synthetic Hindi page | Tesseract `hin` 98.2%, dots.ocr 100% | | Hindi is plumbed but unmeasured on real scans: 1.2 |

Translation: sarvam-translate rendered 8,594 Punjabi paragraphs in 78 minutes
on an RTX 3080 Ti with 146 rejected by the checks. Against a Gemini
reference on thirty paragraphs it scored chrF 53.5, ahead of Gemma 3 12B
(49.8), Qwen3 8B (43.9), IndicTrans2 (41.8) and NLLB (39.4); Google Vision
read the same scans at 88.6% word accuracy against Tesseract's 95.8%.
`docs/engines.md` has the engines, their licences and how they were measured.

## Install

Python 3.12 (3.13 works; 3.14 has no CUDA wheels yet), Tesseract 5, and a GPU
if you want the neural engines or translation to finish in hours rather than
days.

```bash
cd pipeline/python
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements-ocr.txt
```

Tesseract: `apt install tesseract-ocr` / `brew install tesseract` / the UB
Mannheim installer on Windows; then the *best* models for your languages into
a folder `TESSDATA_PREFIX` points at (default `vendor/tessdata`):

```bash
mkdir -p vendor/tessdata/script
for l in pan eng hin; do curl -L -o vendor/tessdata/$l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata; done
curl -L -o vendor/tessdata/script/Gurmukhi.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/script/Gurmukhi.traineddata
```

Embedding models (ONNX, from Hugging Face, no login) into `vendor/models/<name>/`:
`Xenova/bge-small-en-v1.5` for English, `Xenova/multilingual-e5-small` for
Punjabi and Hindi -- `onnx/model_quantized.onnx`, `tokenizer.json`,
`config.json`, `tokenizer_config.json`, `special_tokens_map.json` each.

Translation weights download themselves on first use. Two models ask you to
accept a licence on Hugging Face first (IndicTrans2, IndicOCR); the script
prints the steps when it is refused: accept on the model page, make a Read
token, `hf auth login`.

## Describe a book

A `manifest.json` beside the PDFs (`docs/manifest.md`, `examples/manifest.json`):

```json
{"author": "Bhai Vir Singh", "language": "pa", "reader": "ocr", "licence": "public-domain",
 "works": [{"file": "santhya-vol-1.pdf", "work": "santhya", "part": 1,
            "title": "Santhya Sri Guru Granth Sahib Ji, Vol. 1", "book": "santhya-vol-1"}]}
```

`language` picks the engines and the corpus; `licence` (`public-domain` or
`copyright`) travels with the work into the corpus, for a server to act on;
`reader` is `ocr` for a scan, `pdf-text` for a born-digital PDF.

## Run, then measure

```bash
python 27_ingest_book.py --src ./books --book santhya-vol-1 --pages 1-20   # a first look
python 27_ingest_book.py --src ./books --book santhya-vol-1 --gt           # sample 30 lines and stop
#   read data/ocr/santhya-vol-1/gt/review.html, fix gt/candidates.jsonl
python 23_ocr_gt.py --book santhya-vol-1 --promote
python 27_ingest_book.py --src ./books --book santhya-vol-1 --from eval    # measure, merge, and the rest
```

Without ground truth every engine votes equally and nothing is measured; with
it, `24_ocr_eval.py` prints CER and word accuracy per engine and the merge
weights them by it; an engine measured under 90% does not vote.
`--gpu-engines dotsocr,indicocr` adds the neural engines for a scan Tesseract
reads badly; measure them the same way first.

## Scripture (optional)

If the books quote the Guru Granth Sahib, point `CORPUS_DB` at a database with
a `lines(line_id, shabad_id, ang, gurmukhi_uni, kind)` table -- the
`gurbani.sqlite` that `gurbani-search-api`'s data pack downloads is one -- and
a quoted line is replaced by the corpus text and tagged with its ids, never
"corrected". `MAHANKOSH_DB` adds the Mahan Kosh to the lexicon. Without either
the pipeline runs unchanged and says which steps it skipped.

## Translation

```bash
python 26_translate_writings.py --work santhya --limit 40 --print 5        # a look
python 26_translate_writings.py --work santhya                              # sarvam-translate, resumable
python 26_translate_writings.py --work santhya --engine indictrans2         # the other local engine
python 28_translate_bench.py --work santhya --engines sarvam,indictrans2 --reference vertex --budget-usd 1
python 14_embed_writings.py --lang en --translations && python 15_build_writings_db.py
```

Quoted scripture is never machine-translated (its ids point at real
translations). Every answer is checked -- empty, source script left in, no
Latin letters, far too long, looping -- and rejected rather than written.

## Where it lands, and serving it

| what | where |
|---|---|
| page images, per-engine OCR, merged pages, ground truth, the cost ledger | `data/ocr/<book>/` |
| paragraphs and translations | `data/writings/<work>.jsonl`, `<work>.en.jsonl` |
| the searchable corpus, one per language | `artifacts/corpora/writings-<lang>/` (vectors, manifest) and `artifacts/writings[-<lang>].sqlite` (the passages) |
| reports and measurements | `data/raw/` |

That pair is the contract with
[gurbani-search-api](https://github.com/singhkartaar/gurbani-search-api)
(`docs/output-format.md`): the shape its `writings-<key>` data packs carry.
A deployment serves a corpus once its `CORPORA` table names the directory and
the database, and `GET /api/writings/search?corpus=<key>&q=...` returns whole
passages, the nearest in meaning, with the scripture each one quotes.

## Paid services, if you choose

Google Cloud Vision costs $1.50 per 1,000 pages and, on the clean scans this
was built on, read Gurmukhi worse than Tesseract; Vertex translation is where
a previous project spent thousands. Both run only
through one monthly cap and one ledger (`data/ocr/costs.jsonl`), and
`--dry-run` prints the cost and sends nothing. The benchmarks
(`21_ocr_run.py --gt-pages`, `28_translate_bench.py`) use them on a handful of
pages or paragraphs so you can decide with numbers.

## Status and roadmap

1.1.0 covers Punjabi and English scans end to end, measured. Hindi (`language: "hi"`)
runs through the same code with Tesseract's `hin` model and Devanagari checks in
the translator, but has only been measured on a synthetic typeset page; real
scans, ground truth and a lexicon are 1.2. A legacy-Gurmukhi-font reader
(`reader: legacy-font`) needs a converter that is not in this repository and
says so.

## Contributing

This repository is exported file for file from a larger private one, so a
patch here applies there unchanged -- see `CONTRIBUTING.md`. Run
`python -m unittest test_ocr test_writings` before sending one; the tests need
no scan, engine or model.

## Licence

The code is MIT. The models, a scripture database and the books you feed it
are under their own terms -- see `NOTICE.md`; in particular the default
translator's weights are GPL-3.0, and Gemma 3 12B through `llama-server` is
the measured alternative if that matters for your deployment.
