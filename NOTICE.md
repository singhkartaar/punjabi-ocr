# Licensing

The **code** in this repository is MIT (`LICENSE`). Almost everything it runs
on comes from other people, under other terms, and this file says which.

## Models the pipeline downloads or calls

| model | used for | licence | notes |
|---|---|---|---|
| Tesseract 5 and `tessdata_best` | OCR (every language) | Apache-2.0 | |
| `gurmukhifix` | reordering Tesseract's Gurmukhi output | see its repository | |
| dots.ocr (`rednote-hilab/dots.ocr`) | OCR, GPU | MIT | |
| Surya 2 (`datalab-to/surya`) | OCR, GPU | see its repository (model weights have their own terms) | |
| IndicOCR (`bodhan-ai/indic-ocr`) | OCR, GPU | **Indic Open Model License v1.0**, gated on Hugging Face | accept on the model page first |
| sarvam-translate (`sarvamai/sarvam-translate`) | Punjabi/Hindi to English | **GPL-3.0** (weights) | the default translator; the pipeline calls it as a separate model, it is not linked into this code |
| IndicTrans2 (`ai4bharat/indictrans2-indic-en-1B`) | Punjabi/Hindi to English | MIT, gated on Hugging Face (a click-through) | its preprocessor (IndicTransToolkit, MIT) is vendored as `lib/indictrans_processor.py` |
| `Xenova/bge-small-en-v1.5`, `Xenova/multilingual-e5-small` | embedding passages | MIT (BAAI, intfloat; ONNX exports by Xenova) | |
| Google Cloud Vision, Gemini via Vertex | optional benchmarks and arbiters | Google's terms; paid | only through a budget cap and a ledger |

## Code carried from elsewhere

`lib/legacy_font.py` is the ASCII-to-Unicode half of
[anvaad-js](https://github.com/KhalisFoundation/anvaad-js) 1.5.1 (Khalis
Foundation, **MIT**), ported to Python with its conversion table unchanged.
`lib/indictrans_processor.py` is IndicTransToolkit's preprocessor (AI4Bharat,
**MIT**), in plain Python.

## Data you may point it at

**A scripture database** (`CORPUS_DB`). The one `gurbani-search-api`
distributes is derived from BaniDB (Khalis Foundation) under **NPOSL-3.0**,
the Non-Profit Open Software License, with source documents by Dr Kulbir
Thind and Sant Singh Khalsa whose notice requires written approval for
commercial use or internet projects. Matching a quotation against it is a
non-profit use of the text; publishing what you built from it is a separate
act -- read their terms.

**The Mahan Kosh** (`MAHANKOSH_DB`): Bhai Kahn Singh Nabha, *Gur Shabad
Ratnakar Mahan Kosh* (1930), digitised by redroyals/mahan-kosh-multilingual
under **CC BY 4.0**. Used as a lexicon at build time; nothing from it enters
the output.

**Dasam Bani and Bhai Gurdas** from BaniDB, if you fetch them: NPOSL-3.0 as
above.

## The books

The books are yours. The pipeline records a `licence` on every work from its
manifest (`public-domain` or `copyright`) and carries it into the output;
what a server does with it is that server's policy. Recording a licence is
not a licence to scan a book. Under Indian copyright law a work enters the public domain sixty
years after the author's death; check the law that applies to you before you
publish anything.

Machine translations are search aids. The output labels them (`text_src` is
the author's words; a translated `text` is not), and a reader should be told
the same.
