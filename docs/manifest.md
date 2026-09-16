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
  "works": [
    {"file": "santhya-vol-1.pdf", "work": "santhya", "part": 1,
     "title": "Santhya Sri Guru Granth Sahib Ji, Vol. 1", "book": "santhya-vol-1"},
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
| `reader` | `ocr`, `pdf-text`, `legacy-font` | a scan; a born-digital PDF with a text layer; a PDF in a legacy Gurmukhi font (needs a converter this repository does not ship). Absent: probed from the file |
| `licence` | `public-domain`, `copyright` | recorded on the work and in the output; a serving API shows a copyright work as excerpts only. Absent counts as copyright |
| `original` | `true`, `false` | whether the text is the author's own or a translation of it |
| `quote_policy` | `verbatim`, `summarise` | how an answer may use the prose; default follows the licence |
| `bleed` | `true` | suppress show-through from the reverse side (thin paper, low dpi) |
| `scripture` | `G`, `D`, `B`, `K` | the source the quoted verse belongs to when it is not the Guru Granth Sahib (BaniDB codes); only matters with a scripture database that holds those sources |
| `header_pattern` | a regular expression | how this book's running header states the page's ang range, with named groups `ang_from`, `ang_to`, `book_page`, `section`; the default fits the Santhya's `section (page) bani-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ 12-14` |

A folder without a manifest is read in the essays convention
(`lib/writings_works.py`): `root/`, `English/`, `Punjabi/`, born-digital PDFs
only, with the author given on the command line.
