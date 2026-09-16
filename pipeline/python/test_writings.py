"""
Tests for reading an author's essays out of PDF.

The pure tests build lines and runs by hand, so they need no PDF on disk and
state the geometry they depend on. The one integration test runs only when the
ingested corpus is present.
"""
import json
from importlib import import_module
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paths import ROOT
from lib.writings_pdf import columns, group_lines, is_spread, join_runs, left_margin, paragraphs
from lib.citations import clean_quote, find_ang, looks_quoted, resolve_lexical, tokens
from lib.writings_works import parse_filename

WRITINGS = os.path.join(ROOT, "data", "writings")


def run(x, text, size=10.0, italic=False, y=100.0):
    return {"x": x, "y": y, "size": size, "italic": italic, "text": text}


def line(y, x0, text, size=10.0, italic=False):
    return {"y": y, "x0": x0, "size": size, "italic": italic, "text": text}


class FilenameTests(unittest.TestCase):
    def test_stamp_and_essay_number(self):
        d = parse_filename("1492003384127-Dhooja-Bhau-Part-2.pdf")
        self.assertEqual((d["essay"], d["part"], d["work"]), (127, 2, "dhooja-bhau"))
        self.assertEqual(d["title"], "Dhooja Bhau, Part 2")

    def test_lekh_form_and_the_bilingual_tag(self):
        d = parse_filename("Lekh-110-Simran-Part-1-SS-English-_-Punjabi.pdf")
        self.assertEqual((d["essay"], d["part"], d["work"]), (110, 1, "simran"))
        # the "-SS-English-_-Punjabi" tag names the scan, not the essay
        self.assertEqual(d["work_title"], "Simran")

    def test_a_book_has_no_essay_number(self):
        d = parse_filename("1491937634Universal_Religion.pdf")
        self.assertEqual((d["essay"], d["part"]), (None, None))
        self.assertEqual(d["work"], "universal-religion")

    def test_typists_initials_are_not_part_of_a_title(self):
        self.assertEqual(parse_filename("1563125358Lekh_8_Gurprasad_SS_JK.pdf")["work"], "gurprasad")
        self.assertEqual(parse_filename("Lekh_4_Maya_Dhari_JK-SS.pdf")["work"], "maya-dhari")

    def test_an_editing_round_leaves_a_number_that_is_not_a_title(self):
        # "Shabad-Part-6-edit-2-JK-fixed" is part 6 of Shabad, not "Shabad 2"
        d = parse_filename("1491995000063-Shabad-Part-6-edit-2-JK-fixed.pdf")
        self.assertEqual((d["work"], d["part"]), ("shabad", 6))

    def test_a_number_nobody_touched_stays_in_the_title(self):
        # nothing was stripped here, so the trailing 2 is the author's own
        self.assertEqual(parse_filename("1491937024power_of_thougts_2.pdf")["work"], "power-of-thougts-2")

    def test_one_essay_spelt_two_ways_is_still_one_work(self):
        # Hukum Part 1 and Hukam Part 2 are the same essay; without the alias
        # they would be two works and a reader would find half of each
        self.assertEqual(parse_filename("1491990249025-Hukum-Part-1-.pdf")["work"], "hukam")
        self.assertEqual(parse_filename("1491990270026-Hukam-Part-2.pdf")["work"], "hukam")
        self.assertEqual(parse_filename("1491993320047-Dharan-Ja-Mazab-Part-2.pdf")["work"], "dharam-ja-mazab")

    def test_a_filename_typo_is_corrected_for_display_only(self):
        d = parse_filename("1491936939divine_wil.pdf")
        self.assertEqual(d["work"], "divine-wil")          # the id follows the file
        self.assertEqual(d["work_title"], "Divine Will")   # the reader sees it spelt


class JoinRunsTests(unittest.TestCase):
    def test_a_run_split_mid_word_is_joined_without_a_space(self):
        # pypdf splits runs mid-word; "or h" + "e stops" must not become "or h e stops"
        runs = [run(412.3, "mother's love, or h"), run(491.1, "e stops depending on her")]
        self.assertEqual(join_runs(runs), "mother's love, or he stops depending on her")

    def test_a_real_gap_becomes_a_space(self):
        # the quote's number sits well left of its text
        self.assertEqual(join_runs([run(412.3, "1"), run(430.9, "If the son")]), "1 If the son")

    def test_a_right_aligned_ang_is_not_glued_to_the_verse(self):
        runs = [run(430.9, "his mother does not hold it against him in her mind."), run(700.0, "478")]
        self.assertEqual(join_runs(runs), "his mother does not hold it against him in her mind. 478")


class ParagraphTests(unittest.TestCase):
    def test_italic_separates_a_quoted_verse_from_the_prose(self):
        lines = [line(200, 430, "When a child becomes self-willed, he becomes"),
                 line(188, 412, "indifferent to the mother's love."),
                 line(174, 412, "1 If the son, in anger, runs away, 478", size=8.5, italic=True),
                 line(160, 430, "Exactly in the same way, when we turn our back.")]
        out = paragraphs(lines, body_size=10.0, margin=412.0)
        self.assertEqual([p["style"] for p in out], ["body", "quote", "body"])
        self.assertTrue(out[0]["text"].endswith("mother's love."))

    def test_a_first_line_indent_starts_a_paragraph_and_the_rest_continues_it(self):
        # breaking on "x differs" rather than "x is indented" would split every
        # paragraph after its first line
        lines = [line(200, 430, "First paragraph opens here"),
                 line(188, 412, "and continues on the margin."),
                 line(176, 430, "Second paragraph opens here"),
                 line(164, 412, "and also continues.")]
        out = paragraphs(lines, body_size=10.0, margin=412.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "First paragraph opens here and continues on the margin.")

    def test_a_trailing_hyphen_keeps_its_compound(self):
        lines = [line(200, 412, "he leads a life that is self-"), line(188, 412, "willed and insipid.")]
        out = paragraphs(lines, body_size=10.0, margin=412.0)
        self.assertEqual(out[0]["text"], "he leads a life that is self-willed and insipid.")

    def test_a_wide_gap_starts_a_new_paragraph(self):
        # The threshold is a multiple of the PAGE's median line gap, so it needs
        # a page's worth of gaps to know what normal is: here four lines at the
        # ordinary leading of 12, then one set 40 below.
        lines = [line(200 - i * 12, 412, "a line at the ordinary leading of this page") for i in range(4)]
        lines.append(line(lines[-1]["y"] - 40, 412, "a separate thought begins well below it"))
        out = paragraphs(lines, body_size=10.0, margin=412.0)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[1]["text"].startswith("a separate thought"))

    def test_one_gap_alone_cannot_be_judged_wide(self):
        # A known limit of measuring against the page median rather than the font
        # size: with a single gap on the page, that gap IS the median. It costs
        # nothing here -- a real page carries dozens -- and the alternative was
        # measured to be worse (see LEADING_BREAK in lib/writings_pdf.py).
        lines = [line(200, 412, "The first thought ends here."),
                 line(150, 412, "A separate thought begins here.")]
        self.assertEqual(len(paragraphs(lines, body_size=10.0, margin=412.0)), 1)


class SpreadTests(unittest.TestCase):
    def test_a_two_up_spread_is_recognised_by_where_its_lines_start(self):
        english = [line(200 - i * 12, 412, "a line of English prose on the right half") for i in range(6)]
        self.assertTrue(is_spread(english + [line(550, 31.9, "L127.1")]))

    def test_an_ordinary_column_is_not_a_spread(self):
        self.assertFalse(is_spread([line(200 - i * 12, 43, "a line of an ordinary single column") for i in range(6)]))

    def test_the_margin_is_where_most_lines_start_not_where_the_first_one_does(self):
        lines = [line(200, 430, "an indented first line of some length here")] + \
                [line(200 - i * 12, 412, "a continuation line of some length here") for i in range(1, 5)]
        self.assertEqual(left_margin(lines), 412.0)


class ColumnTests(unittest.TestCase):
    def two_column_page(self):
        left = [run(43 + (i % 3) * 60, "left column text here", y=200 - i * 12) for i in range(20)]
        right = [run(413 + (i % 3) * 60, "right column text here", y=200 - i * 12) for i in range(20)]
        return left + right

    def test_a_gutter_is_found_from_where_runs_begin(self):
        bands = columns([self.two_column_page()])
        self.assertEqual(len(bands), 2)
        self.assertTrue(bands[0][1] > 220 and bands[0][1] < 413)

    def test_a_few_strays_do_not_close_the_gutter(self):
        # one run beginning in the gutter, on one page of many, must not hide it
        page = self.two_column_page()
        self.assertEqual(len(columns([page, page, page + [run(350, "a stray heading", y=560)]])), 2)

    def test_a_spread_is_not_two_columns(self):
        # its left half is a scanned image: one page marker against 40 runs of English
        page = [run(412 + (i % 3) * 60, "the English half of a spread", y=200 - i * 12) for i in range(40)]
        self.assertEqual(columns([page + [run(31.9, "L127.1", y=550)]]), [])

    def test_an_ordinary_column_has_no_gutter(self):
        page = [run(43 + (i % 7) * 40, "an ordinary single column of prose", y=200 - i * 12) for i in range(40)]
        self.assertEqual(columns([page]), [])


class GroupLinesTests(unittest.TestCase):
    def test_runs_at_the_same_height_become_one_line_in_reading_order(self):
        lines = group_lines([run(491.1, "e stops", y=526.3), run(412.3, "mother's love, or h", y=526.3),
                             run(412.3, "the next line", y=514.5)])
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["text"], "mother's love, or he stops")
        self.assertEqual(lines[0]["x0"], 412.3)

    def test_a_line_is_quoted_when_most_of_its_ink_is_slanted(self):
        lines = group_lines([run(412.3, "1", y=300, italic=True),
                             run(430.9, "If the son, in anger, runs away", y=300, italic=True),
                             run(700.0, "478", y=300)])
        self.assertTrue(lines[0]["italic"])


class AngTests(unittest.TestCase):
    def test_the_forms_the_essays_actually_use(self):
        self.assertEqual(find_ang("Says Kabeer, my fear has gone. [Pg. 1349]")[0], 1349)
        self.assertEqual(find_ang("with Satguru's grace, with satsangat. (661)")[0], 661)
        self.assertEqual(find_ang("his mother does not hold it against him in her mind. 478")[0], 478)
        # the PDF glues a right-aligned ang to the last word
        self.assertEqual(find_ang("emotional attachment and love of duality well up.921")[0], 921)
        # an ang range cites its first page
        self.assertEqual(find_ang("I see the Immaculate Lord pervading everywhere. 1349-1350")[0], 1349)
        # the translator may be named after the number
        self.assertEqual(find_ang("I see the Lord everywhere. 1349 Bh. Kabir")[0], 1349)

    def test_the_citation_is_removed_from_the_quote(self):
        ang, rest = find_ang("even then, his mother does not hold it against him. 478")
        self.assertEqual(ang, 478)
        self.assertNotIn("478", rest)

    def test_nothing_outside_the_granth_is_read_as_an_ang(self):
        self.assertIsNone(find_ang("there were 5000 of them")[0])
        self.assertIsNone(find_ang("a paragraph with no citation at all")[0])

    def test_counters_and_the_essays_own_numbering_are_stripped(self):
        got = clean_quote("2 As is the fire within the womb, so is Maya outside. || 1 || Pause || 921")
        self.assertEqual(got, "As is the fire within the womb, so is Maya outside.".rstrip("."))

    def test_a_page_number_is_not_a_quotation_but_a_stanza_counter_is(self):
        # every page carries a number; without this every page footer is a citation
        self.assertFalse(looks_quoted("the prose runs on and ends here 12"))
        self.assertTrue(looks_quoted("detached, like the lotus upon the water. || 1 ||"))
        self.assertTrue(looks_quoted("Says Kabeer. [Pg. 289]"))


class ResolveTests(unittest.TestCase):
    def by_ang(self):
        return {
            478: [(1779, 900, tokens("If the son in anger runs away even then his mother does not hold it against him in her mind")),
                  (1780, 905, tokens("The cranes fly away and land again on the shores of distant seas"))],
            479: [(1781, 910, tokens("Some are made beggars and some are given great kingdoms to rule"))],
        }

    def test_a_verbatim_quotation_resolves_to_its_shabad(self):
        span = {"ang": 478, "text": "If the son, in anger, runs away, even then his mother does not hold it against him in her mind"}
        hit = resolve_lexical(span, self.by_ang())
        self.assertEqual(hit["shabad_id"], 1779)
        self.assertEqual(hit["method"], "lexical_ang")
        self.assertGreater(hit["score"], 0.9)

    def test_without_an_ang_nothing_is_resolved(self):
        # the ang is the window the match runs inside; guessing without it is
        # exactly the confident-looking error this must never make
        self.assertIsNone(resolve_lexical({"ang": None, "text": "his mother does not hold it against him"}, self.by_ang()))

    def test_two_shabads_scoring_alike_are_reported_ambiguous_not_guessed(self):
        by_ang = {5: [(1, 10, tokens("shabad guru sangat naam simran hukam")),
                      (2, 20, tokens("shabad guru sangat naam simran hukam"))]}
        hit = resolve_lexical({"ang": 5, "text": "the shabad guru sangat naam simran hukam together"}, by_ang)
        self.assertTrue(hit["ambiguous"])
        self.assertNotIn("shabad_id", hit)

    def test_a_short_corpus_line_cannot_win_on_containment_alone(self):
        # a three-word line scores a perfect 1.00 against any quote using those
        # three words; without a minimum overlap the same few shabads came back
        # for unrelated quotations and precision sat at 84%
        by_ang = {7: [(1, 10, tokens("burning fire rages"))]}
        span = {"ang": 7, "text": "the burning fire of desire rages within the mind of every mortal being"}
        self.assertIsNone(resolve_lexical(span, by_ang))

    def test_an_ang_one_out_still_resolves(self):
        # a shabad can run over a page break, so the cited ang can be one off
        span = {"ang": 480, "text": "Some are made beggars and some are given great kingdoms to rule over"}
        self.assertEqual(resolve_lexical(span, self.by_ang())["shabad_id"], 1781)


class RepairSplitTests(unittest.TestCase):
    """
    Rejoining words the PDF's spacing broke.

    Every case here is one that a previous version of the rule got wrong, so
    each is a trap rather than a hypothetical.
    """

    ENGLISH = {"and", "word", "own", "inner", "not", "the", "company", "holy", "mind",
               "is", "of", "to", "thee", "external", "influence", "influences", "thought",
               "thoughts", "blotless", "saints", "becomes"}

    def repair(self, *texts):
        mod = import_module("12_ingest_writings")
        records = [{"text": t} for t in texts]
        n = mod.repair_splits(records, english=self.ENGLISH)
        return [r["text"] for r in records], n

    def test_a_stranded_capital_joins_the_word_it_was_cut_from(self):
        # "Simran" and "Paath" are absent from ordinary English, so only the
        # author's own usage can vouch for them
        out, _ = self.repair("doing P aath or S imran",
                             "Simran and Paath", "Simran and Paath", "Simran and Paath")
        self.assertEqual(out[0], "doing Paath or Simran")

    def test_a_word_seen_only_once_is_not_believed(self):
        # the joined form has to be established somewhere other than the broken
        # spot itself, or the repair would simply invent vocabulary
        out, n = self.repair("doing P aath daily")
        self.assertEqual(out[0], "doing P aath daily")
        self.assertEqual(n, 0)

    def test_a_possessive_is_never_read_as_a_stranded_letter(self):
        # "Guru's word" has a word boundary between the apostrophe and the s.
        # A regex repaired 91 of these into "Guru'sword".
        out, _ = self.repair("the Guru’s word is true")
        self.assertEqual(out[0], "the Guru’s word is true")

    def test_a_possessive_the_scan_split_is_put_back_together(self):
        out, _ = self.repair("the Guru ’s word is true")
        self.assertEqual(out[0], "the Guru’s word is true")

    def test_a_letter_never_joins_into_a_word_that_stands_on_its_own(self):
        # "blotles s and" -> "blotles sand" was the failure. "and" is plainly a
        # word, so the s cannot be its first letter; the left join is not a word
        # this author uses either, so nothing is done.
        out, _ = self.repair("becomes blotles s and is rendered")
        self.assertNotIn("sand", out[0])

    def test_a_trailing_s_is_read_as_a_plural(self):
        out, _ = self.repair("our thought s and our influence s are many")
        self.assertIn("thoughts", out[0])
        self.assertIn("influences", out[0])

    def test_a_letter_joins_backward_when_only_that_side_is_a_word(self):
        out, _ = self.repair("Death shall touch the e not")
        self.assertEqual(out[0], "Death shall touch thee not")

    def test_a_genuinely_ambiguous_letter_is_left_alone(self):
        # "thee" and "external" are both words; nothing here can say which was
        # meant, and a wrong join would read as the author's own word
        out, _ = self.repair("with the e xternal music")
        self.assertEqual(out[0], "with the e xternal music")

    def test_a_frequent_defect_is_still_repaired(self):
        # Judging a join by whether the joined form beats the FRAGMENT's count
        # defeats itself: the fragment is common because the defect is
        # systematic. Here "s" outnumbers "thoughts" and must not block it.
        texts = ["thought s here"] * 4 + ["s s s s s s s s"] * 4
        out, _ = self.repair(*texts)
        self.assertIn("thoughts", out[0])

    def test_real_short_words_are_not_swallowed(self):
        out, n = self.repair("he waited a while and I saw it")
        self.assertEqual(out[0], "he waited a while and I saw it")
        self.assertEqual(n, 0)


class UnitTests(unittest.TestCase):
    """The retrieval unit, built by 14_embed_writings.py."""

    @staticmethod
    def build(records, cites=None, target=60, hard=220):
        mod = import_module("14_embed_writings")
        return mod.build_units(records, cites or {}, target, hard)

    @staticmethod
    def para(text, style="body", work="w", part=1, page=1, no=1, uid=None):
        return {"unit_id": uid or "%s:%d:%d:%d" % (work, part, page, no), "work": work,
                "part": part, "page": page, "para_no": no, "marker": None,
                "style": style, "italic": style == "quote", "text": text}

    def test_the_cascading_lists_are_merged_into_something_retrievable(self):
        # half this author's paragraphs are one or two words; a unit reading
        # "recognise" retrieves nothing and means nothing
        records = [self.para("This two way love between mother and child cannot be:-", no=1)]
        records += [self.para(w, no=i + 2) for i, w in enumerate(
            ["studied or cause to be studied", "learnt or taught", "taken or given", "bought"])]
        units = self.build(records, target=10)
        self.assertEqual(len(units), 1)
        self.assertIn("studied", units[0]["text"])
        self.assertIn("bought", units[0]["text"])

    def test_a_unit_does_not_close_in_the_middle_of_a_sentence(self):
        records = [self.para("A sentence that runs well past the target length without ending yet and", no=1),
                   self.para("carries on to finish here.", no=2)]
        units = self.build(records, target=5)
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0]["text"].endswith("finish here."))

    def test_a_unit_never_spans_two_parts(self):
        records = [self.para("The end of part one.", part=1, no=1),
                   self.para("The start of part two.", part=2, no=1)]
        units = self.build(records, target=100)
        self.assertEqual(len(units), 2)
        self.assertEqual([u["part"] for u in units], [1, 2])

    def test_a_quoted_verse_becomes_a_citation_on_the_prose_that_introduced_it(self):
        records = [self.para("we are getting burnt in the fire of Maya -", no=1),
                   self.para("The world is burning in the fire of Maya. 1049", style="quote", no=2),
                   self.para("In this way we forget the Creator.", no=3)]
        cites = {"w:1:1:2": {"shabad_id": 3838, "line_id": 1, "ang": 1049, "score": 1.0,
                             "method": "lexical_ang", "span": "The world is burning"}}
        units = self.build(records, cites, target=100)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["cites"][0]["shabad_id"], 3838)
        # the verse's own words stay out of the unit: they are Gurbani's, and the
        # `en` index over the whole Granth already finds them
        self.assertNotIn("burning in the fire", units[0]["text"])

    def test_an_unresolved_quotation_attaches_nothing(self):
        records = [self.para("he writes -", no=1),
                   self.para("a verse nobody could resolve", style="quote", no=2)]
        self.assertEqual(self.build(records, {}, target=100)[0]["cites"], [])

    def test_a_translated_work_embeds_its_english_and_keeps_the_punjabi_beside_it(self):
        import json
        import tempfile
        mod = import_module("14_embed_writings")
        records = [self.para("ਇਹ ਪਹਿਲਾ ਪੈਰਾ ਹੈ।", no=1), self.para("ਬਾਣੀ ਦੀ ਤੁਕ", style="quote", no=2),
                   self.para("ਇਸ ਦਾ ਕੋਈ ਤਰਜਮਾ ਨਹੀਂ।", no=3)]
        with tempfile.NamedTemporaryFile("w", suffix=".en.jsonl", delete=False, encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"work": "w"}}) + "\n")
            fh.write(json.dumps({"unit_id": "w:1:1:1", "en": "This is the first paragraph."}) + "\n")
        out, translated, dropped = mod.apply_translations(records, fh.name)
        os.unlink(fh.name)
        self.assertEqual((translated, dropped), (1, 1))
        self.assertEqual([r["style"] for r in out], ["body", "quote"])   # the verse passes through
        units = self.build(out, target=100)
        self.assertEqual(units[0]["text"], "This is the first paragraph.")
        self.assertEqual(units[0]["text_src"], "ਇਹ ਪਹਿਲਾ ਪੈਰਾ ਹੈ।")
        # an untranslated English work carries no source text at all
        self.assertNotIn("text_src", self.build(records[:1], target=100)[0])


class IngestedCorpusTests(unittest.TestCase):
    """Runs only where the ingested corpus is present (it is gitignored)."""

    @classmethod
    def setUpClass(cls):
        cls.works_file = os.path.join(WRITINGS, "works.json")
        if not os.path.exists(cls.works_file):
            raise unittest.SkipTest("data/writings not built; run 12_ingest_writings.py")
        with open(cls.works_file, encoding="utf-8") as fh:
            cls.roster = json.load(fh)

    def test_every_work_has_a_jsonl_whose_header_names_it(self):
        for work in self.roster["works"]:
            path = os.path.join(WRITINGS, work["work"] + ".jsonl")
            self.assertTrue(os.path.exists(path), path)
            with open(path, encoding="utf-8") as fh:
                head = json.loads(fh.readline())
            self.assertEqual(head["_meta"]["work"], work["work"])
            self.assertEqual(head["_meta"]["title"], work["title"])

    def test_records_are_in_reading_order_and_uniquely_identified(self):
        path = os.path.join(WRITINGS, self.roster["works"][0]["work"] + ".jsonl")
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh][1:]
        keys = [(r["part"] or 0, r["page"], r["para_no"]) for r in rows]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(r["unit_id"] for r in rows)), len(rows))

    def test_the_parts_of_a_work_are_stitched_together(self):
        # the essays with ten-odd parts are Bau Ji's; a roster built from other
        # authors' books (each one or two PDFs) has nothing to stitch
        if not any(w.get("author") == "Bau Ji" for w in self.roster["works"]):
            self.skipTest("no Bau Ji essays in this roster")
        multi = [w for w in self.roster["works"] if len(w["parts"]) > 1]
        self.assertTrue(multi, "no work has several parts")
        self.assertGreaterEqual(max(len(w["parts"]) for w in multi), 10)


if __name__ == "__main__":
    unittest.main()
