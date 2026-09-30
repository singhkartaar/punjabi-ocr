# Engines

Every engine is an adapter in `lib/ocr_engines.py` with one method,
`recognise(png, lang) -> lines`, and every engine's output is written to
`data/ocr/<book>/ocr/<engine>/NNNN.jsonl` in the same shape, so the merge
(`22_ocr_merge.py`) treats them alike: it aligns their lines, matches quoted
scripture against the corpus if there is one, and votes on the rest with
weights from the evaluation.

## OCR

| engine (`--engine`) | what | runs on | measured (word accuracy on our own ground truth) |
|---|---|---|---|
| `tesseract` (`--tess-lang pan`, `script/Gurmukhi`, `eng`, `hin`, or the default per language) | Tesseract 5 with `tessdata_best`; `gurmukhifix` reorders its Gurmukhi output | CPU, ~0.35 s a page with 6 workers | Punjabi 92-96%, English 99.1%, Hindi (synthetic page) 98.2% |
| `pdftext` | the text layer a scan already carries, if any | free | a voter, never a pivot |
| `dotsocr` | dots.ocr (3B VLM) through transformers; its own venv (transformers 4.51.3); input capped at 2.6 MP for a 12 GB card | GPU, ~350 s a page | Punjabi 87% at line level, perfect where its blocks align; Hindi 100% |
| `surya` | Surya 2, through `llama-server` (`SURYA_INFERENCE_BACKEND=llamacpp`); returns paragraph blocks the merge cuts at the pivot's lines | GPU, ~13 s a page | English voter |
| `indicocr` | Bodhan / AI4Bharat IndicOCR (layout detector + 1.7 GB recogniser, transformers >= 5.7); gated on Hugging Face (accept the licence, `hf auth login`); the engine fetches the repository and imports its code from there | GPU, ~17 s a page | Punjabi 83.1%: clean Gurmukhi, but two of 29 lines missed and below the bar |
| `vision` | Google Cloud Vision `DOCUMENT_TEXT_DETECTION`, $1.50 per 1,000 pages, whole pages, only through `--budget-usd` and the ledger; `--gt-pages` runs it on the ground-truth pages alone | cloud, ~1.1 s a page | Punjabi 88.6% on a clean 300 dpi scan: below Tesseract and the bar, so set aside as a voter there (`docs/design.md`) |

Which engines a language runs by default is in `27_ingest_book.py`
(`DEFAULT_ENGINES`): Punjabi runs both Tesseract variants and lets the merge
vote; English and Hindi run one. `--gpu-engines dotsocr,indicocr` adds the
neural engines where a scan is hard.

**An engine below the bar does not vote.** On the Santhya's ground-truth
pages the three Tesseract variants merge to 94.9%; adding Surya (92.1%)
leaves 94.9%; adding dots.ocr (88.3%) and IndicOCR (83.1%) drops the merge
to 90.6%, and steeper weights do not recover it -- a block-level engine's
lines align imperfectly with the pivot's, and each misalignment is a wrong
vote at full confidence. So the merge sets aside any engine the evaluation
measured under 90% word accuracy (`MIN_VOTER_ACC` in `22_ocr_merge.py`) and
says so; `--engines` overrides. The neural engines earn their place on a
scan where Tesseract itself is under the bar, or on the pages the merge
routes for a second opinion.

Measure before you trust: `23_ocr_gt.py --sample 30` draws lines to verify
(with crops and a review page), `--promote` records them, and
`24_ocr_eval.py --book B` prints CER and word accuracy per engine and for the
merge, and -- for books that quote scripture -- how many quoted lines were
matched to the right id with how many false matches. `--bakeoff --books a,b
--engines all` compares every installed engine across books.

## Translation

| engine (`--engine`) | model | licence | how it works |
|---|---|---|---|
| `sarvam` (default) | `sarvamai/sarvam-translate`, Gemma-3-4B | GPL-3.0 weights | a paragraph at a time, batches of 8 left-padded, generation capped at 48 + 2x the input's tokens; forgiving of OCR noise; 0.58 s a paragraph on a 3080 Ti; Punjabi and Hindi |
| `indictrans2` | `ai4bharat/indictrans2-indic-en-1B` | MIT, gated (click-through) | a sentence at a time: the paragraph is split on dandas and full stops, translated in batches of 32, joined back; its preprocessor is vendored in plain Python (`lib/indictrans_processor.py`) because the PyPI package needs a C++ compiler |
| `madlad` | `google/madlad400-3b-mt` | Apache-2.0 | Google's 400-language T5, sentence-level, `<2en>` prefix; 6 GB on the card in bf16; needs transformers 4.x (5.x loads its embeddings untied and emits digits) |
| `nllb` | `facebook/nllb-200-distilled-1.3B` | CC-BY-NC-4.0 | the research baseline; non-commercial, so a reference point rather than a default |
| `llama` | any GGUF behind llama.cpp's `llama-server` (Gemma 3, Qwen3, Sarvam-M) | the model's | a paragraph at a time through the OpenAI-compatible endpoint; `--model-path` is the URL; a `<think>` block is stripped |
| `vertex` | Gemini through Vertex | paid | the reference for the benchmark and a metered spot-check; never the default |

Every answer is checked before it is written: empty, the source script left
in the English, no Latin letters, more than four times the source's length,
or a six-word run repeated three times (a small model looping on OCR noise)
-- rejected and counted. `28_translate_bench.py` translates a stratified
30-paragraph sample with each local engine and scores chrF, BLEU and the
length ratio against the Vertex reference, caching the reference so a later
engine is scored for free.

## Notation books

A notation book runs Tesseract `pan` twice, with page segmentation mode 3
and mode 4 (`tesseract-pan-psm4`): mode 3's layout analysis drops the
sparse rows of a grid, mode 4 (one column of variable-sized text) keeps
them, and the merge votes between the two and `script/Gurmukhi`. The swaras
themselves are re-read per row strip and per cell by
`TesseractEngine.recognise_region` (psm 7 and 8) once the grid reader runs.
