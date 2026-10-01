# The manifest

A folder of PDFs carries one `manifest.json`. Top-level keys are defaults;
each entry in `works` may override any of them. `lib/writings_manifest.py`
reads it; nothing else knows a book by name.

```json
{
  "author": "Bhai Vir Singh",
  "language": "pa",
  "reader": "ocr",
  "licence": "public-domain",
  "original": true,
  "quote_policy": "verbatim",
  "kind": "commentary",
  "works": [
    {"file": "santhya-vol-1.pdf", "work": "santhya", "part": 1,
     "title": "Santhya Sri Guru Granth Sahib Ji, Vol. 1", "book": "santhya-vol-1", "angs": [1, 53]},
    {"file": "ten-masters.pdf", "work": "ten-masters", "title": "The Book of the Ten Masters",
     "author": "Prof. Puran Singh", "language": "en", "book": "ten-masters", "bleed": true}
  ]
}
```

| key | values | meaning |
|---|---|---|
| `file` | a filename in the folder | required per work |
| `work` | a slug | the work's id; parts of one work share it. Default: a slug of the title |
| `part` | an integer | for a work in volumes |
| `title` | text | shown with every passage |
| `book` | a slug | the working directory under `data/ocr/`. Default: a slug of the file stem |
| `author` | text | |
| `language` | `pa`, `en`, `hi` | which OCR engines run, which corpus the passages join, whether it is translated |
| `reader` | `ocr`, `pdf-text`, `legacy-font` | a scan; a born-digital PDF with a text layer; a PDF whose text layer is typed in a legacy Gurmukhi font such as GurbaniAkhar or AnmolLipi, converted to Unicode without OCR (`lib/legacy_font.py`). Absent: probed from the file |
| `licence` | `public-domain`, `copyright` | recorded on the work and in the output; a serving API shows a copyright work as excerpts only. Absent counts as copyright |
| `original` | `true`, `false` | whether the text is the author's own or a translation of it |
| `quote_policy` | `verbatim`, `summarise` | how an answer may use the prose; default follows the licence |
| `bleed` | `true` | suppress show-through from the reverse side (thin paper, low dpi) |
| `scripture` | `G`, `D`, `B`, `K` | the source the quoted verse belongs to when it is not the Guru Granth Sahib (BaniDB codes); only matters with a scripture database that holds those sources |
| `kind` | `essay`, `translation`, `word-meaning`, `commentary`, `reference`, `notation` | what the work is to the scripture. An essay is *about* it and quotes it; the other four *explain* a verse, a line or a word, and their passages are linked to the lines they explain (`docs/output-format.md`, the `links` table). Default `essay` |
| | | `notation`: a keertan notation book (shabads set to raag and taal in Bhatkhande notation) takes the notation route: `29_notation_parse.py` .. `32_build_notations_db.py` instead of the prose ingest; every other kind is a writings book |
| `translate` | `true`, `false` | whether the work is rendered into English (`26_translate_writings.py`) so it joins the English corpus for search and answers. Default: yes for a work not in English. A work left in its own language is still embedded and served in that language |
| `layout` | `auto`, `paired-columns`, `columns` | how a page is read: `auto` pairs a verse column with the explanation printed beside it and reads everything else in reading order; `paired-columns` forces the pairing; `columns` is the older order (left column whole, then right) |
| `angs` | `[from, to]` | the angs the work covers: the matching window for a page whose header gives no usable ang, and the range recorded on the work |
| `style` | an object | how a notation book prints its grids, merged over the defaults: `swar_row` (`above`/`below` the bol row), `shabad_position` (`before`/`after`/`either` the grid), `matra_row` (bool), `marker_row` (`below`/`above`/`none`), `table` (`bars`/`ruled`/`none`), `labels` (bool: a ਸਰਗਮ/ਸ਼ਬਦ label column), `script` (`gurmukhi`), `running_header` (the text the book prints at the top of its pages, or a list; dropped before linking). Top-level for the folder, per work to override |
| `header_pattern` | a regular expression | how this book's running header states the page's ang range, with named groups `ang_from`, `ang_to`, `book_page`, `section`; the default fits the Santhya's `section (page) bani-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ 12-14` |
| `coverage` | `true`, `false` | whether the merge reads the ink its lines left uncovered (`22_ocr_merge.py --coverage`); `27_ingest_book.py --coverage` / `--no-coverage` override it |
| `correct_agreed` | `true`, `false` | whether the merge also repairs a word every engine misread alike, by putting back a subjoined ra (ਪੁਸ਼ਾਦ to ਪ੍ਰਸ਼ਾਦ) where that makes a known word. For a grey or low-resolution scan whose ਪ੍ਰ, ਸ੍ਰ, ਗ੍ਰ come out as ਪੁ, ਸੁ, ਗੁ in every engine; a clean scan gains nothing from it. Default off; `27 --correct-agreed` / `--no-correct-agreed` override it |
| `glossary` | a filename beside `manifest.json` | the book's terms (`examples/glossary.json`, `lib/mt_glossary.py`): the ones a paragraph uses are named in the translation prompt when `prompt` is `rules`, the model's own bracketed explanations are taken out, and a translation that renders a term by none of its accepted forms is refused ("term: Karah Prasad") |
| `prompt` | `stock`, `rules` | the translator's instruction: the model's trained one, or that plus the glossary's rule and the paragraph's terms (`26 --prompt`) |
| `arbiter` | `none`, `self`, `llama`, `vertex` | who translates again, with the `rules` prompt, a paragraph the checks refused: the same local model, a `llama-server`, or Gemini through Vertex under a budget (`26 --arbiter`) |
| `corpus` | a corpus name | the corpus the book joins, when it is not the default `data/writings`: paragraphs in `data/<corpus>/`, indexes `<corpus>-<lang>`; the folder's other works and their citations are left as they are, and `14 --keep <db>` carries over works whose sources are not on this machine |
| `title_en` | text | the work's English title, beside a Gurmukhi `title` |

A folder without a manifest is read in the essays convention
(`lib/writings_works.py`): `root/`, `English/`, `Punjabi/`, born-digital PDFs
only, with the author given on the command line.
