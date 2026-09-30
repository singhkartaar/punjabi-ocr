# The output: a prose corpus

What the pipeline produces is what `gurbani-search-api` serves. The contract is
one corpus per language: a directory of vectors under `artifacts/corpora/`,
one level deeper than the server's scripture indexes so that nothing that
scans for those can mistake it for one, and a database of passages beside
them. It is the shape the server's `writings-<key>` data packs carry; a
deployment serves a corpus once its `CORPORA` table names the two.

```
artifacts/corpora/writings-en/
  manifest.json          kind: "documents", index, db, label, roles, order,
                         text_lang, query_scripts, embed_dim, index_dim,
                         and the encoder block (model_dir, tokenizer, pooling,
                         query_prefix, doc_prefix, max_len, pad_id, pad_token,
                         lowercase, strip_accents)
  units.i8               N x D  int8     one vector per passage, row == unit_row
  units.scale.f32        N      float32  per-vector scale
  units.mask.u8          N      uint8    1 = retrievable
  pca.components.f32     E x D  float32  the projection applied to a query
  pca.mean.f32           E      float32  subtracted before projecting
artifacts/writings.sqlite  the passages, their works, their citations
```

The database is named for the corpus without its `-en`: the server keys an
English corpus by its bare name (`writings`, `akj`), so `writings-en` is
served from `writings.sqlite`; a corpus in another language keeps its suffix
(`writings-pa` -> `writings-pa.sqlite`), so the Punjabi and the English of
one folder of books never write the same file.

All numeric files are little-endian, C-ordered, headerless. D is `index_dim`
(256), E is `embed_dim` (384). A query is encoded with the model the manifest
names, centred on `pca.mean`, projected by `pca.components`, L2-normalised,
and scored by dot product against the dequantised rows
(`units.i8[row] * units.scale.f32[row]`).

## The database

```sql
CREATE TABLE works (
  work_id TEXT PRIMARY KEY, title TEXT, author TEXT, folder TEXT, original INTEGER,
  quote_policy TEXT, parts TEXT, files INTEGER, units INTEGER,
  language TEXT DEFAULT 'en',      -- the language the work was written in
  licence TEXT                     -- 'public-domain' | 'copyright' | NULL (treated as copyright)
);
CREATE TABLE units (
  unit_row INTEGER PRIMARY KEY,    -- ALSO the row in units.i8; never renumbered
  unit_id TEXT,                    -- <work>:<part>:<page>:<para>
  work_id TEXT, part INTEGER, page INTEGER, para_no INTEGER, marker TEXT,
  text TEXT,                       -- what is searched and shown (English in writings-en, even for a translated work)
  text_src TEXT                    -- the source-language text of a translated passage, else NULL
);
CREATE TABLE citations (           -- the scripture a passage quotes, as the pipeline resolved it
  unit_row INTEGER, shabad_id INTEGER, line_id INTEGER, ang INTEGER,
  score REAL, method TEXT, span TEXT
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);   -- corpus, authors, units, works, citations, built
```

## The manifest fields the server reads

| field | |
|---|---|
| `kind` | `"documents"` -- a corpus of passages, never a scripture index: the server's registry refuses one wherever it finds it |
| `index` | the corpus's name (`writings-en`); the server's `CORPORA` row pairs it with a key |
| `db` | the database of passages, relative to this directory (`../../writings.sqlite`) |
| `label`, `label_pa`, `order` | presentation |
| `roles` | `["text"]` |
| `text_lang`, `query_scripts` | the language of `text`, and which scripts a query may use (a Gurmukhi query against an English index is refused) |
| `embed_dim`, `index_dim` | E and D |
| `model_dir`, `tokenizer`, `pooling`, `query_prefix`, `doc_prefix`, `max_len`, `pad_id`, `pad_token`, `lowercase`, `strip_accents` | how to encode a query; the same field names a scripture index uses, so the server builds the encoder without a special case |
| `licences` | informational: works per licence status |

## Licence, recorded

The pipeline records what the manifest says: `licence` on each row of
`works`, and a count per status under `licences` in the manifest. What a
server or a data pack does with a copyright work is that server's policy, not
this pipeline's -- `gurbani-search-api` returns whole passages, a bounded
number per query, and records the basis for each author in its NOTICE.md.
Set `licence` in the manifest honestly; an unrecorded one counts as
copyright, and `quote_policy` follows it unless the manifest says otherwise.

## A notation book

A work with `kind: notation` writes no passages. Its output is
`data/notations/<book>/`:

| file | |
|---|---|
| `notations.jsonl` | `_meta` first, then one record per notation: `notation_id` (`<book>:<page>:<seq>`), `kind`, `heading` (raag used, taal, laya as printed and as keys of `lib/notation_vocab.json`), `shabad` (`shabad_id`, `source`, `confidence`, `method`, `ang`, the printed lines and reference), `raag_shabad`, `sections` (the grid: lines of beats, each beat its notes with octave/komal/tivra/length, or an extension, a rest, or unread; and the bol syllable), `images`, `flags`, `verified`. The contract is `lib/notation.schema.json` with the cross-field rules in `lib/notation.py` |
| `images/` | 1-bit PNG crops: first the notation as printed on each of its pages (role `block`, margin to margin, so a taan or a note under the grid is inside), then every grid, shabad text and heading, and a thumbnail of the first block; `images.json` lists them with `sha256`, size and box |
| `raags.jsonl` | what the book says about its raags: one record a description (a heading naming the raag and no taal, then prose), with its crop(s) and the OCR text |
| `data/ocr/<book>/gt/notation-review.html`, `notation-candidates.jsonl`, `notation-gold.jsonl` | the review page, the reviewer's judgements, the promoted gold |
| `data/raw/notation-report-<book>.json`, `notation-eval-<book>.json` | what the parse found; the fields measured against the gold and whether the book passed the bars |
| `artifacts/notations.sqlite`, `notations-images.json` | the database a serving API reads (tables and columns pinned in `lib/notation_columns.json`), and the list of images with their hashes for an upload tool |

## Also written

| file | |
|---|---|
| `data/writings/<work>.jsonl` | one record per source paragraph, in reading order: `unit_id`, `work`, `part`, `page`, `para_no`, `style` (`body`, `heading`, `quote`, `footnote`), `text`, `lang`, and for a quoted verse its `line_ids`, `shabad_id`, `ang`, `match_score` |
| `data/writings/<work>.en.jsonl` | `unit_id`, `en`, `engine`, `model` per translated paragraph |
| `data/writings/units[-<lang>].jsonl` | the retrieval units, `_meta` first, exactly what the database holds |
| `data/writings/citations.jsonl` | the resolved quotations |
| `data/ocr/<book>/merged/NNNN.jsonl` | the merged page: every line with its `text`, `kind`, `zone`, `bbox`, `agreement`, and the corpus match if any |
| `data/raw/ocr-eval-<book>.json`, `ocr-report-<book>.json`, `mt-bench-<work>.json` | the measurements |
