# Runbook: a scanned book to searchable, translated text

Every step, in order, as you would type it. `README.md` is the overview;
`docs/manifest.md`, `docs/engines.md` and `docs/output-format.md` are the
references this page points into. Commands are shown for a POSIX shell; on
Windows PowerShell use `.venv\Scripts\python.exe` and backslashes.

```bash
cd pipeline/python
py=.venv/bin/python            # Windows: .venv\Scripts\python.exe
```

## 0. Once per machine

**Python 3.12** (3.13 works; 3.14 has no CUDA wheels yet), **Tesseract 5**,
and a GPU with 12 GB or more if you want the neural engines and translation
to finish in hours rather than days. Everything runs on a CPU, slowly.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or https://download.pytorch.org/whl/cpu
pip install -r requirements-ocr.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Tesseract: `apt install tesseract-ocr`, `brew install tesseract`, or the UB
Mannheim installer on Windows. Then the *best* models for your languages into
`vendor/tessdata` (or wherever `TESSDATA_PREFIX` points):

```bash
mkdir -p ../../vendor/tessdata/script
for l in pan eng hin; do curl -L -o ../../vendor/tessdata/$l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata; done
curl -L -o ../../vendor/tessdata/script/Gurmukhi.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/script/Gurmukhi.traineddata
```

Embedding models (ONNX, from Hugging Face, no login) into `vendor/models/<name>/`:
`Xenova/bge-small-en-v1.5` for English, `Xenova/multilingual-e5-small` for
Punjabi and Hindi. Each needs `onnx/model_quantized.onnx` (saved as
`model_quantized.onnx`), `tokenizer.json`, `config.json`,
`tokenizer_config.json` and `special_tokens_map.json`:

```bash
$py - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os
for repo, name in [("Xenova/bge-small-en-v1.5", "bge-small-en-v1.5"), ("Xenova/multilingual-e5-small", "multilingual-e5-small")]:
    d = os.path.join("..", "..", "vendor", "models", name); os.makedirs(d, exist_ok=True)
    shutil.copy(hf_hub_download(repo, "onnx/model_quantized.onnx"), os.path.join(d, "model_quantized.onnx"))
    for f in ["tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(hf_hub_download(repo, f), os.path.join(d, f))
EOF
```

Translation weights download themselves on first use into the Hugging Face
cache. Two of the optional models are gated: accept the licence on the
model page while signed in (https://huggingface.co/ai4bharat/indictrans2-indic-en-1B,
https://huggingface.co/bodhan-ai/indic-ocr), make a **Read** token at
https://huggingface.co/settings/tokens, and log in once:

```bash
hf auth login
```

A refused engine prints these steps instead of a stack trace. IndicTrans2
and MADLAD need transformers 4.x, so give them a second venv
(`python -m venv .venv-t4 && pip install "transformers<5" torch sentencepiece sacremoses indic-nlp-library-itt sacrebleu`)
and run those two engines with it.

**Scripture, optional.** If the books quote the Guru Granth Sahib, point
`CORPUS_DB` at a database with a `lines(line_id, shabad_id, ang, gurmukhi_uni, kind)`
table; the `gurbani.sqlite` that `gurbani-search-api`'s data pack downloads
is one. Quoted lines are then found in it and tagged with their ids rather
than corrected. `MAHANKOSH_DB` adds the Mahan Kosh to the lexicon. Without
either, the pipeline runs unchanged and says which steps it skipped.

## 1. Describe the book

One `manifest.json` beside the PDFs (`docs/manifest.md`; `examples/manifest.json`):

```json
{"author": "Bhai Vir Singh", "language": "pa", "reader": "ocr", "licence": "public-domain",
 "works": [{"file": "santhya-vol-1.pdf", "work": "santhya", "part": 1,
            "title": "Santhya Sri Guru Granth Sahib Ji, Vol. 1", "book": "santhya-vol-1"}]}
```

`language` (`pa`, `en`, `hi`) picks the engines and the corpus; `licence`
(`public-domain` or `copyright`) travels with the work and decides how much
of it a public server shows; `reader` is `ocr` for a scan, `pdf-text` for a
born-digital PDF; `bleed: true` suppresses show-through on thin paper.

## 2. A first look

```bash
$py 27_ingest_book.py --src /path/to/books --book santhya-vol-1 --dry-run     # prints every command, runs nothing
$py 27_ingest_book.py --src /path/to/books --book santhya-vol-1 --pages 1-20  # twenty pages, end to end
```

Open `data/ocr/santhya-vol-1/pages/0001.png` (is it upright and clean?) and
`data/ocr/santhya-vol-1/merged/0001.jsonl` (are the lines what the page
says?). The driver runs, per book: render (`20`), OCR with the language's
engines (`21`; Punjabi runs Tesseract `pan` and `script/Gurmukhi` and lets
the merge vote, English `eng`, Hindi `hin`), evaluate if there is ground
truth (`24`), merge (`22`); then for the folder: paragraphs (`12`), citations
(`13`, skipped without a scripture database), embeddings (`14`), the
database (`15`); then translation for a Punjabi or Hindi book (`26`) and the
English corpus (`14 --lang en --translations`, `15`). A failed step names the
`--from` value that resumes the run.

## 3. Ground truth, so the numbers are yours

Thirty lines a person reads from the crops are enough to rank the engines and
weight their votes. Without them every engine votes equally and nothing is
measured; with them the merge sets aside any engine under 90% word accuracy.

```bash
$py 27_ingest_book.py --src /path/to/books --book santhya-vol-1 --gt          # renders, OCRs, samples 30 lines, stops
#   open data/ocr/santhya-vol-1/gt/review.html; correct each candidate's text in gt/candidates.jsonl
$py 23_ocr_gt.py --book santhya-vol-1 --promote                                # -> gt/lines.jsonl
$py 27_ingest_book.py --src /path/to/books --book santhya-vol-1 --from eval    # measures, merges, and the rest
```

`24_ocr_eval.py --book santhya-vol-1` prints CER, word accuracy and, for
books that quote scripture, how many quoted lines matched the right id, per
engine and for the merge. `--bakeoff --books a,b --engines all` compares
every installed engine across books. `--gpu-engines dotsocr,indicocr` on the
driver adds the neural engines; they vote only if the evaluation puts them
at 90% or better.

## 4. Translation

```bash
$py 26_translate_writings.py --work santhya --limit 40 --print 5       # a look
$py 26_translate_writings.py --work santhya                             # sarvam-translate, resumable
$py 28_translate_bench.py --work santhya --reference vertex --dry-run   # what a paid reference would cost (cents)
$py 28_translate_bench.py --work santhya --engines sarvam --reference vertex --budget-usd 1
$py 28_translate_bench.py --work santhya --engines llama --path llama=http://127.0.0.1:8080 --as llama=gemma3-12b --reference cached
$py 14_embed_writings.py --lang en --translations && $py 15_build_writings_db.py
```

Quoted scripture is never machine-translated; its ids point at real
translations. Every answer is checked (empty, source script left in, no
Latin, far too long, looping) and rejected rather than written. The
benchmark needs `GCP_PROJECT` and `gcloud auth application-default login`
for its one paid reference, cached afterwards; `docs/engines.md` has the
engines and the numbers they reached.

## 5. Where it lands, and serving it

| what | where |
|---|---|
| page images, per-engine OCR, merged pages, ground truth, the cost ledger | `data/ocr/<book>/` |
| paragraphs and translations | `data/writings/<work>.jsonl`, `<work>.en.jsonl` |
| the searchable corpus, one per language | `artifacts/corpora/writings-<lang>/` (vectors, manifest) and `artifacts/writings[-<lang>].sqlite` (the passages) |
| reports and measurements | `data/raw/` |

That pair is the contract with `gurbani-search-api` (`docs/output-format.md`):
the shape its `writings-<key>` data packs carry. Copy both into a deployment's
`ARTIFACTS_DIR`, name them in the server's `CORPORA` table (key, directory,
database), and `GET /api/writings/search?corpus=<key>&q=...` returns whole
passages, the nearest in meaning.

## 5a. A keertan notation book

A book of shabads set to music (a grid of swaras over the sung syllables,
a heading with the raag and taal) is declared with `kind: notation` in the
manifest (docs/manifest.md) and takes its own route. First the review
window -- seven pages from the middle of the book, at least two shabads:

```
python 27_ingest_book.py --src ./books/dyal-singh --book gurmat-sangeet-sagar-1 --gt
```

Open `data/ocr/<book>/gt/notation-review.html`, judge every notation's
shabad, raag, taal, laya and section structure against its crops, download
the candidates over `gt/notation-candidates.jsonl`, then

```
python 30_notation_gt.py --book <book> --check
python 30_notation_gt.py --book <book> --promote
python 31_notation_eval.py --book <book>
```

Fix what the review found (the parser, or the manifest's `style`) and rerun
`29_notation_parse.py --book <book> --pages <window> --force` until every
field is right. Then the whole book:

```
python 27_ingest_book.py --src ./books/dyal-singh --book gurmat-sangeet-sagar-1
python 32_build_notations_db.py --gurbani <path to gurbani.sqlite>
```

Then review the whole book, with every verdict saved as it is given:

```
python 34_notation_review.py serve --book gurmat-sangeet-sagar-1     # tick, comment, or reject each notation
python 34_notation_review.py check --book gurmat-sangeet-sagar-1 --strict   # after a change to the reader
python 34_notation_review.py status                                  # the dashboard
python 32_build_notations_db.py --gurbani <path to gurbani.sqlite> --accepted-only
```

The ledger (`REVIEW_DIR/<book>.jsonl`) keeps each notation's verdict under
a stable key; an accepted notation is frozen and never redone, a commented
one comes back after the next change, and `check` is the regression gate
on everything accepted. The database refuses a book whose review did not
pass; `--allow-unmeasured` is for a pilot, `--accepted-only` builds the
reviewed notations alone. `docs/output-format.md` describes what is written.

On a second machine the same commands run from the same branch; only the
small files move between machines, by git: the records, the ledger and
the fixtures under `REVIEW_DIR`, the page metadata and the layout cache.
Renders, OCR output and images are made again where they are needed, and
`29_notation_parse.py --recut --book <book>` cuts a book's images again
from the bboxes its records store, refusing a crop whose sha differs.

To see how the reader fares across a shelf of books before committing to
any of them, `33_notation_sample.py --library <folder of books> --books 20
--per-book 2` reads a few pages at two random places in each and writes
one review page for all of them (`data/ocr/_sample/review.html`).

## 6. Paid services, only through a cap

Every paid call (Google Vision for OCR, Gemini through Vertex for
translation) goes through one monthly cap and one ledger,
`data/ocr/costs.jsonl`; `--dry-run` prints the cost and sends nothing. Set
`GCP_PROJECT` to the project that pays. Measure on the ground-truth pages
before paying for a book:

```bash
$py 21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --gt-pages --dry-run
$py 21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --gt-pages --budget-usd 0.25
$py 24_ocr_eval.py --book santhya-vol-1
```

On the clean scans this was built on, Vision read Gurmukhi worse than
Tesseract's best model and cost $0.80 a book; measure yours before deciding.
