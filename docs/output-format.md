# The output: a documents index

What the pipeline produces is what `gurbani-search-api` serves. The contract is
one directory per language, `artifacts/writings-<lang>/`, in the shape the
server already discovers for its scripture indexes, distinguished by one
field in its manifest. Drop it into a deployment's `ARTIFACTS_DIR` and
`GET /api/documents?q=...` searches it.

```
artifacts/writings-en/
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
  units.sqlite           the passages, their works, their citations
```

All numeric files are little-endian, C-ordered, headerless. D is `index_dim`
(256), E is `embed_dim` (384). A query is encoded with the model the manifest
names, centred on `pca.mean`, projected by `pca.components`, L2-normalised,
and scored by dot product against the dequantised rows
(`units.i8[row] * units.scale.f32[row]`).

## units.sqlite

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
| `kind` | `"documents"` -- what keeps this index out of the scripture routes; a server keeps it in a map of its own |
| `index` | the name used in `?index=` and `INDEXES=` |
| `db` | the database beside the vectors; default `units.sqlite` |
| `label`, `label_pa`, `order` | presentation |
| `roles` | `["text"]` |
| `text_lang`, `query_scripts` | the language of `text`, and which scripts a query may use (a Gurmukhi query against an English index is refused) |
| `embed_dim`, `index_dim` | E and D |
| `model_dir`, `tokenizer`, `pooling`, `query_prefix`, `doc_prefix`, `max_len`, `pad_id`, `pad_token`, `lowercase`, `strip_accents` | how to encode a query; the same field names a scripture index uses, so the server builds the encoder without a special case |
| `licences` | informational: works per licence status |

## Licence, applied twice

A serving API returns a passage of a `public-domain` work whole and any other
passage as an excerpt of at most 300 characters around the sentence closest
to the query. A data pack of the index is built the same way: copyright
passages are cut to a fixed lead excerpt and their `text_src` removed before
the file is published, and `meta.text_policy = excerpt:copyright` records it.
The vectors ship whole -- they are derived data and do not reconstruct the
text. Set `licence` in the manifest honestly; an unrecorded one counts as
copyright.

## Also written

| file | |
|---|---|
| `data/writings/<work>.jsonl` | one record per source paragraph, in reading order: `unit_id`, `work`, `part`, `page`, `para_no`, `style` (`body`, `heading`, `quote`, `footnote`), `text`, `lang`, and for a quoted verse its `line_ids`, `shabad_id`, `ang`, `match_score` |
| `data/writings/<work>.en.jsonl` | `unit_id`, `en`, `engine`, `model` per translated paragraph |
| `data/writings/units[-<lang>].jsonl` | the retrieval units, `_meta` first, exactly what `units.sqlite` holds |
| `data/writings/citations.jsonl` | the resolved quotations |
| `data/ocr/<book>/merged/NNNN.jsonl` | the merged page: every line with its `text`, `kind`, `zone`, `bbox`, `agreement`, and the corpus match if any |
| `data/raw/ocr-eval-<book>.json`, `ocr-report-<book>.json`, `mt-bench-<work>.json` | the measurements |
