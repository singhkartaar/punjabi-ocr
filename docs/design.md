# Design: what each step does, what was measured, and why

`README.md` is the overview and `docs/runbook.md` the commands. This page is
the reference behind them: what each step reads and writes, the numbers that
were measured on real books, and the decisions those numbers led to. Read
"Decisions worth knowing" before changing the merge, the zones or the
matching: most of them record something that was tried the other way first.
The engines are described in `docs/engines.md`, the output in
`docs/output-format.md`, the manifest in `docs/manifest.md`.

## The idea

The Gurbani a book quotes is already in a scripture database (`CORPUS_DB`,
optional). An OCR'd Gurmukhi line is therefore **found**, not corrected:
every engine's reading of it is matched against the corpus (`lib/ocr_match.py`), and a match is replaced
by the corpus text and tagged with its `line_id`s. Archaic spellings survive by
construction and the citation comes out for free. Only the commentary -- modern
Punjabi prose -- is voted across engines, gated by a lexicon, and corrected.

Nothing is sent to a paid service unless it goes through `lib/ocr_route.Budget`:
a monthly cap, an append-only ledger (`data/ocr/costs.jsonl`) and `--dry-run`.

## Steps

| step | reads | writes |
|---|---|---|
| `20_ocr_pages.py --src <folder>` | the PDFs a `manifest.json` lists | `data/ocr/<book>/pages/NNNN.png` (300 dpi, deskewed), `pages.json` |
| `21_ocr_run.py --book B --engine E` | the page images | `data/ocr/<book>/ocr/<engine>/NNNN.jsonl` |
| `22_ocr_merge.py --book B` | every engine's output | `data/ocr/<book>/merged/NNNN.jsonl`, `merged/route.json`, `data/raw/ocr-report-<book>.json` |
| `23_ocr_gt.py --book B --sample 30` | the pivot engine's output | `gt/candidates.jsonl` + crops + `review.html`; `--promote` -> `gt/lines.jsonl` |
| `24_ocr_eval.py --book B` | `gt/lines.jsonl` and every engine | `data/raw/ocr-eval-<book>.json` (`--bakeoff` -> `ocr-bakeoff.json`) |
| `12_ingest_writings.py --src <folder>` | the manifest; merged OCR or a text layer | `data/writings/<work>.jsonl`, `works.json` |
| `13 -> 14 --lang pa -> 15 --units units-pa.jsonl` | as before | `artifacts/corpora/writings-pa/` (vectors, manifest with `kind: documents`) and `artifacts/writings-pa.sqlite`: the Punjabi corpus |
| `26_translate_writings.py --work W` | `<work>.jsonl` | `<work>.en.jsonl` (local sarvam-translate or IndicTrans2; `--src-lang hi` for Hindi) |
| `14 --lang en --translations -> 15` | the English works and every `<work>.en.jsonl` | the `writings-en` corpus and database with the translated books in it |
| `27_ingest_book.py --src <folder>` | the manifest | runs the rows above in order for each book; `--dry-run` prints them |
| `28_translate_bench.py --work W` | a 30-paragraph sample | `data/raw/mt-bench-<work>.json`: each local engine scored against a paid reference, under a cap |
| `29_bench_books.py --src <folder>` | the manifest | a sample of pages through 20-22 per book, then one scorecard row per book (Markdown, and `data/raw/bench-<stamp>.json`) |

A source folder carries a `manifest.json` (`lib/writings_manifest.py`): author,
language, licence, `quote_policy`, and one entry per PDF with its `book` key
and `reader` (`ocr` | `legacy-font` | `pdf-text`). A folder without one is read
by its filenames (`lib/writings_works.py`), or through a roster (`--roster`).

## Measured (ground truth: 29 verified lines per book, `data/ocr/<book>/gt/lines.jsonl`)

| book | engine | CER | word accuracy | Gurbani match |
|---|---|---|---|---|
| Santhya (pa, 1-bit 300 dpi) | tesseract `pan` | 2.5% | 95.8% | 6/6 right, 0 false |
| Santhya | tesseract `pan+eng` | 4.0% | 94.5% | 6/6 |
| Santhya | tesseract `script/Gurmukhi` | 3.9% | 93.6% | 6/6 |
| Santhya | merged (three Tesseract variants) | 3.2% | 94.9% | 6/6 |
| Gurbani Vayakaran (pa with English pages) | tesseract `script/Gurmukhi` | 6.1% | 92.2% | 7/7 |
| Gurbani Vayakaran | tesseract `pan` | 34% | 66.6% (English pages unread) | 7/7 |
| Ten Masters (en, 100 dpi JPEG) | tesseract `eng` | 0.2% | 99.1% | -- |
| Ten Masters | pdftext (the scan's own layer) | 3.8% | 76.6% | -- |

| Santhya | surya (llama.cpp) | 5.0% | 92.1% | 6/6 |
| Santhya | dots.ocr (all 15 truth pages, 354 s/page) | 10.0% | 87.0% | 4/5 |
| Santhya | IndicOCR (15 truth pages, 17 s/page; 27/29 lines found) | 10.7% | 83.1% | 5/6, 0 false |
| Santhya | Google Vision (15 truth pages, 1.1 s/page, $0.0225; 28/29 found) | 13.4% (Gurbani lines 23.2%) | 88.6% | 5/6, 0 false |
| Santhya | merged, final (three Tesseract variants + Surya, stream matching, gutter fixes) | 3.6% | 94.9% (commentary 97.9%) | 6/6 |
| Santhya | merged with dots.ocr and IndicOCR also voting (rejected: see Decisions) | 6.4% | 90.6% | 6/6 |
| Gurbani Vayakaran | merged, final | 5.9% | 92.0% | 7/7 |
| Gurbani Vayakaran | surya (14 lines) | 4.4% | 84.9% | -- |
| Ten Masters | surya (12 lines) | 1.3% | 88.3% | -- |
| Ten Masters | merged (accuracy-weighted vote) | 0.2% | 99.1% | -- |

`24_ocr_eval.py --bakeoff` (`data/raw/ocr-bakeoff.json`): the configurations
that clear the 90% word-accuracy bar on every Punjabi truth set are the
**merge** (94.9% Santhya, 92.0% Vayakaran; judged on prose lines, its Gurbani
lines being corpus text) and **Tesseract `script/Gurmukhi`** alone; `pan`
clears it on the Santhya but not on the Vayakaran's English pages; for English,
**Tesseract** and the **merge** (both 99.1%). dots.ocr reads Gurmukhi at least
as well as any of them where its blocks line up (6/6 perfect on the first
pages) but its paragraph boxes straddle the truth lines on two-column pages,
which the line-level score punishes (87.0% over all 15 pages), and at ~6
minutes a page through transformers on a 12 GB card it is an arbiter for routed
pages, not a full-pass engine, until it runs under vLLM (WSL2/Docker).

Book-level yield after the final merge: on the Santhya, 3,016 verse lines
resolved to the corpus and 1,065 bold lines unresolved (quotations from
sources the scripture database did not hold, raag headings, and pages where
Tesseract merged the two columns). 8,594 of its 8,740 prose paragraphs were
translated in 78 minutes on an RTX 3080 Ti, with 146 rejected by the checks.
Merged Gurbani lines score a higher "CER" than the pivot
because the corpus spelling replaces the book's padchhed by design; the match
precision is the number that matters there. The three Tesseract variants are
correlated, so voting among them adds little; an independent voter (Vision,
dots.ocr) is what the vote is for.

## The bench: a new book in one row

`29_bench_books.py --src <folder>` runs 20-22 on a sample of each work's
pages (the first 12 and 8 spread over the rest) with the coverage pass and
prints two Markdown tables, one row per book, saved to
`data/raw/bench-<stamp>.json`:

```
| book | reader | pages | s/page | columns | lines g/gu/h/c | matched | recovered | uncovered | routes | word acc (prose, gurbani) | line recall |
| book | kind | layout p/c/s | paragraphs | quotes | paired | explains | angs found | angs manifest | ang source | keys set |
```

`columns` counts pages by how the gutter was found (`rule`, `ink`, `words`,
`none`); `lines` are matched Gurbani / unmatched / heading / commentary;
`recovered` and `uncovered` are the coverage pass's lines added and ink it
still could not read; `routes` the three commonest reasons a line was set
aside; `layout` the pages the reader took as paired / plain columns / single;
`paired` and `explains` the paragraphs linked to a verse; `angs found` the
range the running headers give after smoothing, beside the manifest's;
`ang source` how many pages got their ang from the header, from smoothing,
from the manifest, or not at all; `keys set` the manifest keys the work
already sets. Accuracy columns appear only where `gt/lines.jsonl` or
`gt/page-counts.jsonl` exist (`--gt N` samples lines at the same time); no
number is invented without them.

What a row asks for is a manifest key, not code: `words` in `columns` or
zero `paired` on a translation -> look at the pages, then `layout`; `-` in
`angs found` -> `header_pattern`; `uncovered` on two-column pages -> the
coverage pass met a layout it does not handle, which is a decision to record
here with its number. The row for the Santhya (530 pages, the whole book)
is the reference: `columns` rule/ink on its two-column pages, 154 pages
paired, every page's ang from the header or smoothing.

### The coverage pass, measured (2026-09-30, the Santhya, our own Tesseract runs)

Ground truth: 29 page-sampled lines (23_ocr_gt.py --sample), 10 pages whose
printed lines a reviewer counted (--count-pages: 440 lines, 5 two-column
pages), and 40 coverage regions read by eye (--coverage-sample: 22 real
lines, 18 that were rules, slivers or specks).

| | tesseract `pan` | merge, no coverage | merge, `--coverage` |
|---|---|---|---|
| word accuracy, sampled lines | 94.5% | 91.8% | 91.8% |
| prose CER | 1.2% | 1.3% | 1.3% |
| printed lines found, 10 pages | 398 of 440 (0.905) | 406 (0.923) | 441 (0.998, excess 0.005) |
| two-column pages | 0.873 | 0.910 | 1.000 |
| lines recovered over 530 pages | | | 1,455; 8 regions left unread |
| recovered lines, word accuracy | | | 89.4% (CER 4.9%), 20 of 20 found |
| false recoveries | | | 0 of 19 non-text crops |
| cost | | | +0.47 s a page |

The merge's word accuracy is below the pivot's only on Gurbani rows, where
the merge writes the corpus's text for a run of wrapped verse and a printed
row no longer maps to one record (the row's danda goes to the next record,
`ਮਹਾ ਬਲ` becomes the corpus's `ਮਹਾਬਲ`); on prose the two are equal. Of the
lines the layout missed, every one sampled was a real line, and psm 7 read
719 of them as empty until raw mode (psm 13) was tried: that fallback and
the lower height threshold took the recovered count from 259 to 1,455 and
the uncovered from 730 to 8. On a 120 dpi grey scan of another book (the
Sant Attar Singh biography, 20 sample pages) the same change took recovered
from 14 to 98 and uncovered from 90 to 10.

### Linked, measured (the same run, `27_ingest_book.py --src <folder> --book santhya-vol-1`)

The whole volume through ingest, citations, embedding and the database,
with `kind: commentary`, `layout: auto`, `angs: [1, 53]`, `translate:
false` in its manifest:

| | |
|---|---|
| pages read as paired / single / plain columns | 246 / 272 / 12 |
| paragraphs, of them quotations | 12,222, 3,505 (2,646 matched to the corpus, 75.5%) |
| paragraphs with a `pair`, explaining a verse | 6,203, 3,448 |
| units in `writings-pa`, units with an `explains` link | 3,188, 848 |
| `links` rows: explains / quotes | 910 / 684 (all source G) |
| shabads explained inside angs 1-53 | 155; 15 explains links fall outside (a paired band's verse matched elsewhere) |
| `works.ang_from`, `ang_to` | 1, 53 (the manifest's) |

The paragraph count fell from 15,269 to 12,222 when an OCR line box within
12% of the body height stopped counting as a size change (below).
Page 70 reads preface, heading, then each verse of Japji's third pauri
followed by its arth linked to shabad 3; page 201 pairs the verses of
shabad 40 (ang 8) with theirs, 350 shabad 80 (ang 24). Before the
manifest's `angs` were applied on single-column pages, 834 of 1,774
explains links pointed outside the volume: the preface's illustrative
quotations from elsewhere in the Granth, each taking the prose after it
as its explanation.

### A grey scan's conjuncts, measured (2026-09-30, the Sant Attar Singh biography, 20 pages)

A 120 dpi grey scan prints the subjoined ra (ਪ੍ਰ, ਸ੍ਰ, ਗ੍ਰ) as a small hook
that both Tesseract models read as an aunkar, a dulainkar or nothing: every
ਸ੍ਰੀ, ਗ੍ਰੰਥ, ਪ੍ਰਸ਼ਾਦ and ਪ੍ਰਕਾਸ਼ of the sample came out wrong, and the translation
made "the Guru's funeral" of ਸੁੀ ਗੁਰੂ ਗੁੰਥ. Ground truth: 57 page lines and
10 coverage regions read by eye.

| merge | word accuracy | corrections on ground-truth lines, wrong of made | 11 subjoin words over 20 pages, right / wrong | unknown words |
|---|---|---|---|---|
| before | 92.8% | 1 of 1 | 23 / 22 | 40.1% |
| conjuncts, Mahan Kosh, book vocabulary | 92.9% | 1 of 2 | 28 / 19 | 14.5% |
| + agreed words, any seeded confusion | 91.9% | 10 of 11 | | |
| + measured weights, agreed words by the subjoin repair only | 93.0% | 0 of 2 | 42 / 3 | 14.2% |

On the Santhya's clean scan the same code leaves word accuracy (91.8%) and
line recall (0.998) where they were and makes one wrong correction where
the old made two (ਮਿੱਤ੍ਰ to ਸਤ੍ਰ, friend to enemy, is gone).

## Decisions worth knowing

- **An engine below the bar does not vote.** With IndicOCR and dots.ocr
  available for the Santhya's 15 ground-truth pages, the merge was tried
  four ways: the three Tesseract variants, 94.9%; plus Surya (92.1%), 94.9%;
  plus dots.ocr (88.3%) and IndicOCR (83.1%), 90.6%; the same six with
  accuracy^8 weights, 90.8%. The weight is not the problem: a block-level
  engine's lines align imperfectly with the pivot's, and every misalignment
  is a wrong vote at full confidence. So `22_ocr_merge.py` sets aside any
  engine the evaluation measured under `MIN_VOTER_ACC` (0.90) and says so;
  `--engines` overrides. The GPU engines keep their two real uses: a page
  the pivot reads badly (routed), and a scan where Tesseract itself is under
  the bar. A Tesseract variant beside a Tesseract pivot is not held to the bar: it reads the same lines the same way, and on a scan where both variants measure under 90% (Santhya vols. 2, 4, 5, 7: 0.78-0.91) the bar left the pivot voting alone, with no book vocabulary and no corrections; with both voting the merge measured 0.817 -> 0.830, 0.820 -> 0.825, 0.913 -> 0.923, 0.856 -> 0.854.

- **A conjunct is one cluster to the corrector** (`lib/ocr_text.conjuncts`),
  and marks on one letter are priced one by one: split as extended grapheme
  clusters, the repair of ਪੁੇਮ to ਪ੍ਰੇਮ cost 2.25 against a cap of 1.0, and a
  flat price for any mark difference tied ਗੁੰਥ between ਗ੍ਰੰਥ and ਗਾਥ. CER, the
  vote and the lexicon keep grapheme clusters, so published numbers stay
  comparable. An aunkar or dulainkar stacked on another vowel sign (ਸੁੀ) is
  no spelling: it is the subjoin misread, cheap to repair, never an
  inflection. Among equally near candidates the one the corpus uses ten
  times as often wins (ਬ੍ਰਹਮ over the Kosh's ਬਹਮ).
- **A word every engine read alike receives one repair only.** Offered the
  whole confusion table, the corrector bent the author's spelling toward the
  dictionary's ten times in eleven (ਚੱਲਣ to ਚਲਣ, ਖਿਸਕਣ to ਖਿਸਕਣਾ). What both
  engines get wrong the same way on a grey scan is the subjoined ra, so that
  is all `correct_agreed` puts back, and only onto a known word.
- **The book's vocabulary is learnt, and shared misreadings are refused.**
  A word two engines agree on across three pages is the author's; both read
  ਪ੍ਰਸਾਦਿ as ਪੁਸਾਦਿ and ਬ੍ਰਹਮ as ਬੁਹਮ on that scan, so a word that the subjoin
  put back turns into a known word is not admitted.
- **Pages are rendered, never image-extracted**: two sample books tile each page
  into eight JPEG strips.
- **`pan` alone beats `pan+eng` on pure Gurmukhi pages** (Latin fragments creep
  into bold words otherwise) but cannot read the English pages of a Punjabi book;
  the merge takes both.
- **Single danda is prose punctuation, double danda is verse.** Only `॥` marks a
  quotation; `।` is the full stop of modern Punjabi.
- **A bold split needs three lines of each weight.** Stroke widths on the
  100 dpi Ten Masters scan sit within 0.4 px of each other; two faint lines
  made the "regular" group of p.88 and the other 34 came out bold, and since
  bold means quotation, half the book's prose (2,690 of 5,246 lines) was
  left out of the retrieval units. A page with fewer than three lines of a
  weight has one weight.
- **A stamp is a stamp if ANY engine read it as one.** The Gurmukhi pivot reads
  the viewer's "Page 2 of 530" as digits ("। 2੩06 2 0! 530") and the zone pass
  cannot know; the merge drops a line once one engine's reading is a stamp,
  and `is_stamp` also accepts a few specks of noise before "Page N of M".
  Before this, five stamps reached the Santhya's text and eight watermarks the
  Ten Masters'.
- **Corpus headings are excluded from matching**: "ਮਾਰੂ ਮਹਲਾ ੫ ॥" would match
  the citations the books print after a quote.
- **A shorter corpus line contained in a longer one is not an ambiguity**: a
  runner-up must claim a comparable stretch of the reading.
- **Correction touches only words the engines disagreed on.** A word every
  engine read the same way and the lexicon does not know is the author's
  spelling (ਉਪਲਖਤ) until proven otherwise; a dropped syllable (ਪੰਜਵੇਂ -> ਪੰਜ)
  is never a "confusion".
- **Bold is read from stroke width** (regular 4.0-4.2 px, bold 4.6-4.7 px at
  300 dpi on the Santhya); matched corpus text is the stronger signal and is
  used regardless.
- **Wrapped verse is matched as a stream.** On the Santhya's two-column pages
  one corpus line wraps over two or three OCR lines in the narrow left column,
  so runs of consecutive bold lines in a column are concatenated and matched
  as one text (`lib/ocr_match.align_stream`) BEFORE the per-line pass. The
  corpus line lands on the first OCR line (`matches`), the continuation lines
  keep only what lies outside the match and point back (`merged_into`).
  Matching fragments one by one instead gave the first fragment the whole
  corpus line and left the rest to duplicate it. Full-book effect: 2,090 →
  3,865 matched Gurbani lines, 1,727 → 1,198 unmatched bold lines.
- **A line Tesseract's segmenter skips is adopted from another engine**
  (`adopt_orphans`), judged against every pivot line including headers so the
  other engines' header readings are not adopted as body. All three Tesseract
  variants share the segmenter, so this pays off only with an independent
  engine (Vision, dots.ocr).
- **A gutter must be ink-free, not just free of word starts.** A justified
  English page shows 24 px runs with no word beginning in them by chance; the
  Santhya's real gutter is a hairline with ~55 px of white either side. The
  column detector now needs 2% of the page width and under 1% ink in the band.
  Before this, Ten Masters was cut at a false gutter and its merged accuracy
  fell from 99.1% to 97.3%; after it, merged equals the pivot.
- **Vote weights come from the measurement.** With equal weights the scan's
  own 77% text layer and Surya's 88% blocks outvoted a 99% Tesseract pivot.
  `22_ocr_merge.py --weights auto` (the default) weights each engine by its
  measured word accuracy to the fourth power, 0.5 where unmeasured, so run
  `24_ocr_eval.py --book B` (every engine) before merging a book.
- **Routing thresholds are only partly calibrated.** On the Vayakaran truth
  set low cross-engine agreement caught 6 of 7 lines with CER over 5%
  (precision 0.5) -- but those are the English pages the pan-only variant
  cannot read, so the signal is the variants' script mismatch, not OCR noise.
  On the Santhya the three correlated variants agree on their shared
  mistakes and routing caught none of the four bad lines. Re-measure once
  Vision or dots.ocr votes; for English the OOV gate is relaxed to 0.6 because
  the lexicon (the Gurbani translations) lacks the proper nouns such a book is
  made of.

- **The page layout stays the pivot; the coverage pass recovers only the ink
  it left uncovered.** Tesseract's layout drops lines on two-column pages
  (a right-column line here, a table column there), and the first answer
  tried elsewhere was to replace the layout with a projection-profile line
  finder and read every line as a crop (`psm 13`): coverage went to 100% and
  word accuracy from 94.9% to 79.7% on the same book, because a line finder
  that cannot tell a line from a rule or a smudge reads everything as text.
  So the merge keeps the layout and, with `--coverage`, draws the ink no
  recognised box covers (`lib/ocr_coverage.py`: every box padded a quarter
  line height, header and footer bands, stamps, the page edges and a halo
  round both rules excluded), cuts the residual into row regions per column,
  rejects what is not a line by shape (too short, tall, narrow, sparse,
  dense, solid) and reads the rest one crop at a time with the same
  Tesseract voters (`recognise_region`, `psm 7`, or `psm 6` for a block).
  A crop that reads empty is read again in raw mode (`psm 13`, or `psm 4`
  for a block): Tesseract's segmenter refuses the crop for the same reason
  its page layout skipped the line (a grey band behind the text, a line it
  takes for a picture), and the raw modes have no segmenter to refuse it
  with -- 719 crops on the Santhya and 61 on a 120 dpi scan came back
  empty at `psm 7`, every one sampled a real line, and `psm 13` read them
  all. A reading with no letters in it is rejected. A region's height is
  judged against the line height at 0.35 (a run cut at the valley under
  its headline is the x-height alone, 0.41-0.48 of the line; 0.5 refused
  968 real lines), and the cap is 40 regions a page, after the shape tests,
  so a picture page is not the cost but a page whose layout dropped half
  its lines gets them all. Every region is recorded in `_meta.coverage` with
  its status and why, a recovered line carries `recovered`, and the
  evaluation scores recovered lines apart from the page's
  (`by_provenance`). On by default since the measurement below
  (`COVERAGE_DEFAULT` in `27_ingest_book.py`; a manifest's `coverage: false`
  or `--no-coverage` turns it off for a book).
- **A hairline rule is the gutter.** The Santhya prints a vertical rule
  between the verse column and its arth; the ink-free-band test found 163 of
  the ruled pages and the rule detector (`lib/ocr_zones.vertical_rule`: per
  column of pixels the longest dark stroke with gaps of a few rows bridged,
  at most 1% of the page wide, white on both sides over its own rows) finds
  258, the extra 95 confirmed by eye. Where a rule is found the crossing
  test is skipped (a line that crosses the rule is what `split_at_gutter`
  is for, and it splits only within the rule's span) and `_meta.columns_source`
  says `rule`, `ink` or `words`.
- **An orphan is judged in its column.** `adopt_orphans` compared vertical
  overlap only, so a left-column line another engine found was "covered" by
  the right-column line at the same height and never adopted; it now needs
  horizontal overlap too.
- **Header hints are smoothed over the book.** The running header gives the
  page's ang range and is the matching window, and Tesseract misreads its
  digits often enough to matter (12 -> 92, 42 -> 491: 84 of 437 headers on
  the Santhya). `smooth_hints` runs a window over the neighbouring pages
  with a tolerance that scales with the book's own rate of angs per page,
  replaces a value outside it by the interpolation between trusted
  neighbours, fills pages without a header, and fixes the printed page
  number by the modal offset: 48 headers corrected and 93 filled over 530
  pages, no jumps left, and running it again changes nothing. Each page
  records `ang_source` (`header`, `smoothed`, `manifest`, or none);
  `header_pattern` in the manifest describes another book's header.
- **A two-column page is read in bands, and a verse pairs with the
  explanation beside it.** Reading the left column whole and then the right
  ran the verse chunks together, put every arth after all of them, and
  returned a full-width line the gutter had split as two half lines far
  apart: 50% of the commentary paragraphs ended mid-sentence. `lib/ocr_pairs.py`
  cuts the page into rows and bands (full-width prose rejoined; a verse half
  beside a prose half is a paired row; two or more make a paired band), cuts
  the prose column where an explanation starts (level with a verse chunk,
  or after a wider baseline step) and gives each prose paragraph the verse
  lines printed beside it by overlap. 154 of the Santhya's 530 pages read as
  paired, 3,299 explanations, 451 of them continuing over a page. `layout:
  columns` in the manifest is the old reading, unchanged, for every corpus
  built before.
- **A verse's links go forward to what explains it.** An essay quotes a tuk
  to make its point, so its citation belongs to the prose before the verse;
  a translation, word-meaning or commentary (the manifest's `kind`) puts the
  verse first and its explanation after, so the link belongs to the prose
  after it, in the `explains` role. `14_embed_writings.py` holds a verse's
  citations until the prose that explains its pair comes; one cite per
  (source, shabad, role), its line range widened over every mention. The
  `links` table carries every relation with its scripture (`source`), range
  and role; `citations` keeps its old shape and only Guru Granth Sahib rows,
  because its `line_id` is documented as an id of that scripture.
- **Whether a work is translated is the work's decision.** A commentary
  kept in Punjabi for display beside the verse and a translation meant for
  English search can share a folder; each work's `translate` says which, and
  the driver runs 26 for the ones that want it.

## Hindi: plumbing done, measured on a synthetic page only

No Hindi scans were available, so the measurement is on a synthetic page
(six Devanagari lines, printed to PDF through a browser, with their truth). The
whole chain runs for `language: "hi"` (`tessdata_best/hin`, `--lang hi`).
Tesseract reads that page at **98.2% word accuracy** (CER 0.6%, one word:
पैंदीस for पैंतीस), `hin` alone or `hin+eng`; **dots.ocr reads it perfectly**
(CER 0, 191 s for the page); `data/raw/ocr-eval-hindi-sample.json`.

A first version of that page was drawn with Pillow, which does no Indic
shaping, so the image itself showed every i-matra after its consonant and
every conjunct unformed; all three engines then "scored" 71-73% by reading
faithfully what was drawn (साहबि, धरम). A fixture for a complex script has to
come from a shaping renderer (a browser, LibreOffice, a real scan), never
from `ImageDraw.text`. There is still no Hindi lexicon to gate on, and the
bar is only met on typeset text: real scans, when they arrive, decide between
Tesseract, dots.ocr and IndicOCR (gated) the same way the Punjabi bake-off did.

## Translation: the commentary in English, locally

A reader searching in English is served from `writings-en`, so a Punjabi
book's prose is translated once, on a local GPU, by `sarvamai/sarvam-translate`
(Gemma-3-4B fine-tune, GPL-3.0 weights):

```
26_translate_writings.py --work santhya --limit 40 --print 5     # a look first
26_translate_writings.py --work santhya                          # resumable; ~0.4 s a paragraph in batches of 8
14_embed_writings.py --lang en --translations                    # English works + every work with a .en.jsonl
15_build_writings_db.py                                          # artifacts/writings.sqlite: units.text English, units.text_src the Punjabi
```

Only body, heading and footnote paragraphs are sent; a quote paragraph is
Gurbani, already carrying line_ids that point at the corpus's real English
translations, so it is never machine-translated. Every answer is checked
(empty, Gurmukhi left in, no Latin letters), plus two rules a 4B model needs: an answer more than four times the
Punjabi's length, or one that repeats a six-word run three times, is the
model looping on a fragment of OCR noise ("This is the translation of the
given Punjabi text to English:" until the token cap -- 37 of the first 1,187
answers before the rule), and is rejected rather than written; the generation
cap is also tied to the input's length (48 + 2x its tokens) so a loop costs
seconds, not the full 768 tokens. A paragraph without a translation is
dropped from the English corpus, not embedded in Gurmukhi. Paragraphs are
sorted by length into batches of eight (left-padded), about 0.5 s each on an
RTX 3080 Ti. The translation
inherits the OCR's mistakes (the sample's "ਨਰਕ" for a misread word became
"hell"), which is one more reason the merge's accuracy matters and why a
Punjabi original rides beside every translated unit.

The Vertex engine (`--engine vertex --budget-usd N`) stays for a measured
spot-check through the same ledger as the OCR arbiters; it is never the
default: an earlier project's paid translation run is where thousands went.

A second local engine, `--engine indictrans2` (AI4Bharat's
`indictrans2-indic-en-1B`, MIT, a 1B seq2seq model), translates a sentence at
a time: `split_sentences()` cuts a paragraph on dandas and full stops, the
sentences go through the model in batches of 32, and each paragraph is joined
back from its own. Its preprocessor is vendored as
`lib/indictrans_processor.py` (the PyPI `IndicTransToolkit` is a Cython source
build that needs a C++ compiler; the copy is the same code without the
typing). The model is gated on Hugging Face (a click-through). `--src-lang hi`
switches both engines to Hindi: sarvam-translate covers it with the same
prompt, IndicTrans2 with the `hin_Deva` tag, and the rejection rule looks for
Devanagari instead of Gurmukhi in the answer.

Which engine to use is decided by `28_translate_bench.py`: a stratified
30-paragraph sample translated by each local engine and scored (chrF, BLEU,
length ratio; sacrebleu) against Gemini through Vertex, cents under a cap,
the reference cached in `data/raw/mt-bench-<work>.json` so later engines are
scored for free.

Measured on 2026-09-16 (28 paragraphs scored; Gemini 2.5 Flash rejected two
of its own thirty references by the same checks, and they are left out):

| work | engine | licence | chrF | BLEU | length ratio | rejected of 30 |
|---|---|---|---|---|---|---|
| santhya | **sarvam-translate** 4B (fresh, batches of 8) | GPL-3.0 weights | **53.5** | **29.2** | 1.30 | 0 |
| santhya | sarvam-translate (the finished full-book run) | | 53.0 | 29.4 | 1.28 | 0 |
| santhya | Gemma 3 12B instruct, Q4_K_M via llama-server | Gemma terms | 49.8 | 27.8 | 1.20 | 0 |
| santhya | Qwen3 8B, Q4_K_M via llama-server | Apache-2.0 | 43.9 | 21.4 | 1.24 | 1 (transliterates some lines) |
| santhya | IndicTrans2 1B (fp32) | MIT | 41.8 | 20.9 | 1.26 | 5 (runaways) |
| santhya | NLLB-200 distilled 1.3B | CC-BY-NC-4.0 | 39.4 | 17.4 | 1.09 | 0 |
| santhya | MADLAD-400 3B | Apache-2.0 | unusable | | 2.7 | 24 (dots to the cap, invented sentences; the same in bf16 on the GPU and fp32 on the CPU, under transformers 4 and 5) |

A chrF in the fifties against a different model's rendering of free
commentary is agreement on words, not a grade; the pairs printed with the
score are what to read. sarvam's misses are paraphrases of noisy source
("The performer then begins the sukha" for a line whose OCR left "ਦਾਤਾਰਗੀ"),
not inventions. The verdict: sarvam-translate stays the default; Gemma 3 12B
is the alternative if the GPL weights ever matter for a deployment, at the
same speed once loaded; the small purpose-built translators trail the
instruction models on commentary full of parenthetical glosses, which is
not the newswire they were trained on.

Two of them need transformers 4: IndicTrans2's model code imports
`transformers.onnx` (gone in 5), and transformers 5 loads MADLAD's decoder
embeddings untied and emits digits. Both run from `.venv-dots` (4.51.3)
with `sentencepiece`, `sacremoses` and `indic-nlp-library-itt` added.
IndicTrans2 in fp16 ran away (thirty paragraphs unfinished after 25 minutes
at 100% GPU); fp32 fits at 1B. MADLAD is broken on this text in every
precision tried, so the failure is the model's, not the loader's.

### Terms, measured (2026-09-30, the Sant Attar Singh biography, 30 paragraphs)

A 4B translator rendered ਆਸਾ ਜੀ ਦੀ ਵਾਰ ਦਾ ਭੋਗ ਪਿਆ as "Asa Ji's time of death
was near" and a canopy for the Sri Guru Granth Sahib as one "for the Guru's
funeral". Scored against references translated from the Punjabi, with the
book's glossary:

| | chrF | answered of 30 | Sikh terms rendered |
|---|---|---|---|
| before (unrepaired OCR, stock prompt) | 52.5 | 29 | 57 of 61 |
| stock prompt | 53.9 | 27 | 57 |
| the terms in the instruction (`rules`) | 55.8 on the 19 answered | 19 (10 copied back in Punjabi) | 44 |
| the terms written into the Punjabi (`terms`) | 55.4 | 27 | 58 |
| `terms`, the refused asked again with the stock prompt | 55.0 | 30 | 58 |

sarvam-translate is tuned on one system line and follows nothing else: given
the rule and the term pairs it translated the instruction into Punjabi,
given one line of English it copied a third of the paragraphs back. It keeps
a Latin word, so the glossary reaches it through the text. The check that
refuses a translation for a missing term flagged 21 of 213 older
translations, nearly all of them wrong ("the ascetic community" for ਸਾਧ
ਸੰਗਤ, "This sect has plundered"); it sees terms, not meaning ("the Bhog was
consumed" passes it), and its accepted lists stay narrow on purpose: widening
ਸੰਗਤ to "gathering" let a sentence with its negation inverted through.

## The paid arbiter (Google Vision): measured, and not needed on this scan

The API was enabled on 2026-09-16 and the benchmark run: `--gt-pages`
restricted it to the 15 ground-truth pages ($0.0225, in the ledger), and
`24_ocr_eval.py` put it in the table above. On the Santhya's clean 300-dpi
1-bit scan Vision reads Gurmukhi at 88.6% word accuracy, worse than
Tesseract's `pan` model (95.8%) and under the 90% bar, so the merge sets it
aside as a voter by the same rule as dots.ocr and IndicOCR, and the plan's
"third voter on every Punjabi page" is withdrawn for this kind of scan: it
would cost $0.80 a book to lower the result. It stays available for a scan
Tesseract cannot read, measured the same way first. The recipes, for that
case:

```
21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --gt-pages --budget-usd 0.05   # the benchmark
21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --dry-run             # 530 pages, $0.80, sends nothing
21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --budget-usd 5        # the full third-voter pass
21_ocr_run.py --book santhya-vol-1 --engine vision --lang pa --routed              # only merged/route.json's pages (297, $0.45)
24_ocr_eval.py --book santhya-vol-1 && 22_ocr_merge.py --book santhya-vol-1       # measure, then re-merge with it voting
```

`gcloud auth application-default login` and `GCP_PROJECT` first. Every request
is checked against the monthly cap and written to `data/ocr/costs.jsonl`.

## Keertan notations

A notation book is the same OCR up to the merge, and a different reading
after it. The merge runs in notation mode (no footnote rule, no gutter
split, no lexicon correction: a grid's rules and gaps and its all-OOV
swara rows would trip each of them) and keeps the corpus matching of the
shabad's lines, which is what names the shabad. `lib/notation_layout.py`
classifies each merged line by its tokens and its ink -- heading, section
label, marker row, grid row, shabad text, reference, note -- and strings
consecutive pages into spans; `lib/notation_text.py` reads the headings
and the eight printed forms of a reference; `lib/notation_resolve.py`
weighs three signals for the shabad (text 0.5, reference 0.3, bol row 0.2)
and flags disagreement instead of picking; `lib/ocr_grid.py` cuts 1-bit
crops. The record is script-neutral (`lib/notation.schema.json`), rendered
to Gurmukhi or English by `lib/notation_render.py`, whose JavaScript twin is
pinned to the same bytes by the fixtures.

The decisions: the scan is stored and shown as the authority, the parse
beside it with a confidence; nothing is built from a book until a person
has judged a mid-book window of it against the crops (`30_notation_gt.py
--mid`, `31_notation_eval.py`), and `32_build_notations_db.py` refuses a
book under the bars; glyph ambiguity (× is the sam in a marker row and a
rest in a swar row) is resolved by a row's role, never by the glyph; and a
number is an ang only when nothing says it is a mahala, a patshahi, a vaar,
a pauri or a footnote.
