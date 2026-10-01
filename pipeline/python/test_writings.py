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
from lib.writings_pdf import (columns, group_lines, is_spread, join_runs, left_margin,
                              collapse_overprint, drop_shadows, legacy_font_run, page_columns, strip_legacy_words, paragraphs, read_pdf, running_heads)
from lib.citations import (clean_quote, detect, find_ang, looks_quoted, opened_by, resolve_lexical,
                           tokens, wholly)
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

    def test_a_font_that_states_its_widths_is_believed_over_the_estimate(self):
        # Where a PDF gives the advance of every glyph, the end of a run is
        # known rather than guessed. CHAR_W = 0.46 is calibrated for Bau Ji's
        # scans and over-estimates a narrower face, which closes the gap and
        # drops the space: the AKJ Autobiography came out as
        # "On hearingthis,theInspectorGeneralwas" until this was read.
        # "this," is 5 narrow glyphs: 22.0 wide in the real font, but the
        # estimate makes it 5 x 12 x 0.46 = 27.6 and so runs past the next word.
        pair = [run(0.0, "this,", size=12.0), run(24.7, "the", size=12.0)]
        exact = [{**pair[0], "width": 22.0, "space": None},
                 {**pair[1], "width": 14.0, "space": None}]
        self.assertEqual(join_runs(exact), "this, the")
        # the same pair with no metrics falls back, and the estimate loses it
        self.assertEqual(join_runs(pair), "this,the")

    def test_an_exact_run_split_mid_word_still_joins(self):
        # the tighter threshold must not turn a mid-word split into two words
        runs = [{**run(412.3, "or h"), "width": 20.0, "space": None},
                {**run(432.3, "e stops"), "width": 30.0, "space": None}]
        self.assertEqual(join_runs(runs), "or he stops")


class FurnitureTests(unittest.TestCase):
    """Running heads, printed page numbers and watermarks -- read_pdf(furniture=True)."""

    @staticmethod
    def book(pages):
        return [[{"text": t} for t in page] for page in pages]

    def test_a_head_that_recurs_and_a_watermark_that_wanders_are_both_found(self):
        # six lines a page, so the middle two are nowhere near an edge
        pages = self.book([["WWW.AKJ.ORG", "%d CHAPTER ONE" % n, "real prose here",
                            "www.stamp", "and more prose", "%d" % n]
                           for n in range(1, 11)])
        heads, offset, stamps, repeated = running_heads(pages)
        self.assertIn("WWW.AKJ.ORG", heads)
        # the stamp sits in the middle of the page, so only recurrence finds it
        self.assertIn("www.stamp", stamps)
        # the printed number tracks the page at a fixed offset
        self.assertEqual(offset, 0)
        self.assertIn("CHAPTER ONE", repeated)
        self.assertNotIn("real prose here", heads)

    def test_a_numbered_line_that_does_not_recur_is_left_alone(self):
        # "2. Where does it come from?" opens page 2 of a Bau Ji essay. It
        # carries the page number at the right offset and sits at the top, so
        # position and number alone would discard a real line of the essay.
        pages = self.book([["%d. Where does it come from?" % n, "prose"] for n in range(1, 11)])
        _, _, _, repeated = running_heads(pages)
        self.assertNotIn(". Where does it come from?", repeated)

    def test_nothing_is_stripped_unless_asked(self):
        # Bau Ji's corpus is built and shipped from this reader; these filters
        # move its output, so they are opt-in and he stays byte-identical.
        import inspect
        self.assertIs(inspect.signature(read_pdf).parameters["furniture"].default, False)


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

    def test_where_italic_is_not_the_quotation_face_a_title_is_a_heading_and_a_term_is_nothing(self):
        # In Search of the True Guru sets its chapter titles and its Punjabi
        # terms in italic, and no verse. Read as quotes, the title stood as a
        # verse and the short last line "did not do Naam simran." -- mostly one
        # italic term by ink -- was cut off its paragraph, 236 times.
        lines = [line(200, 412, "Charitable Acts", italic=True),
                 line(188, 412, "There are many people who do kind things for others but most"),
                 line(176, 412, "did not do Naam simran.", italic=True)]
        out = paragraphs(lines, body_size=10.0, margin=412.0, italic_quotes=False)
        self.assertEqual([p["style"] for p in out], ["heading", "body"])
        self.assertTrue(out[1]["text"].endswith("did not do Naam simran."))
        # the default reading is unchanged for every book before it
        self.assertEqual([p["style"] for p in paragraphs(lines, body_size=10.0, margin=412.0)],
                         ["quote", "body", "quote"])


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


class PageColumnTests(unittest.TestCase):
    """
    A magazine's columns, found page by page (rosters/barusahib.json).

    Eternal Voice sets one article in two columns, the next in four, a title
    page in one, and puts a pull quote or an indented verse in the gutter; the
    whole-document profile above finds nothing in any of its sixteen articles.
    """

    WIDTH = 595.0

    def column(self, x, n, top=780):
        return [run(x, "a line of the column", y=top - i * 12) for i in range(n)]

    def test_two_columns_split_where_the_right_one_starts(self):
        page = self.column(40, 40) + self.column(300, 40)
        bands = page_columns(page, self.WIDTH)
        self.assertEqual(len(bands), 2)
        self.assertTrue(40 < bands[0][1] <= 300)

    def test_four_columns_are_four(self):
        page = self.column(36, 45) + self.column(180, 17) + self.column(300, 26) + self.column(432, 16)
        self.assertEqual(len(page_columns(page, self.WIDTH)), 4)

    def test_a_pull_quote_in_the_gutter_does_not_move_it(self):
        page = self.column(40, 40) + self.column(300, 40) + [run(150, "a pull quote", y=400)] * 3
        bands = page_columns(page, self.WIDTH)
        self.assertEqual(len(bands), 2)
        self.assertGreater(bands[0][1], 150)

    def test_a_single_column_with_runs_starting_mid_line_is_one(self):
        # an italic word or a bold name starts a run wherever it falls in the line
        page = self.column(80, 40) + [run(120 + 37 * i, "word", y=700 - 30 * i) for i in range(10)]
        self.assertEqual(page_columns(page, self.WIDTH), [])

    def test_an_indented_verse_is_not_a_column(self):
        page = self.column(80, 30) + self.column(130, 10, top=500)
        self.assertEqual(page_columns(page, self.WIDTH), [])


class OverprintTests(unittest.TestCase):
    """Fake bold, cut to one copy (Sikh Faith draws its headings five times over)."""

    def collapsed(self, *texts, copies=5):
        runs = [run(90 + 0.3 * i, t, y=500) for i, t in enumerate(texts)]
        collapse_overprint(runs, copies)
        return [r["text"] for r in runs]

    def test_a_run_holding_its_words_five_times_holds_them_once(self):
        self.assertEqual(self.collapsed("Union with the Divine (God)" * 5), ["Union with the Divine (God)"])

    def test_a_label_inside_body_text_is_cut_and_the_text_kept(self):
        self.assertEqual(self.collapsed("Inert Matter : " * 5 + "This class consists of suns"),
                         ["Inert Matter : This class consists of suns"])

    def test_copies_drawn_as_separate_runs_at_one_spot_are_one_run(self):
        # a drop shadow, as the Eternal Voice articles set their headings
        runs = [run(56.0, "Need for ", y=610.55), run(56.0, "Persons", y=396.04),
                run(54.67, "Need for ", y=612.26), run(54.67, "Persons", y=397.75)]
        self.assertEqual(drop_shadows(runs), 2)
        self.assertEqual([r["text"] for r in runs], ["Need for ", "Persons"])
        # the same words elsewhere on the page are text, not a shadow
        runs = [run(56.0, "Need for ", y=610.55), run(90.0, "Need for ", y=300.0)]
        self.assertEqual(drop_shadows(runs), 0)

    def test_in_a_fake_bold_run_a_lone_word_and_a_stuttered_capital_are_cut_too(self):
        self.assertEqual(self.collapsed("Guru " * 5 + "Arjan Dev states:" * 5), ["Guru Arjan Dev states:"])
        self.assertEqual(self.collapsed("W WW WWilliam " + "William " * 4 + "W " + "arburt" * 5),
                         ["William Warburt"])

    def test_a_word_repeated_as_verse_or_chant_is_left_alone(self):
        # "Har " five times is how a verse is written, not fake bold
        self.assertEqual(self.collapsed("Har Har Har Har Har gun gaavahu"), ["Har Har Har Har Har gun gaavahu"])
        # and a count other than the book's is not its fake bold
        self.assertEqual(self.collapsed("SaintSaint"), ["SaintSaint"])


class LegacyFontTests(unittest.TestCase):
    """Gurmukhi in fonts nothing here converts (Sant Waryam Singh Ji's books)."""

    def test_a_whole_line_in_either_font_is_legacy_and_english_is_not(self):
        self.assertTrue(legacy_font_run("ÕÇð ÕÇð ÔÅÇðú ÁÇéÕ"))
        self.assertTrue(legacy_font_run("frqj ;'fJB uzdB[ ;[rzX bkfJ w'sh jho/.."))
        for english in ("strength and rhythm", "In holy company are these effaced.’ P. 206",
                        "Sabh meh jot jot hai soi", "P.", " 707"):
            self.assertFalse(legacy_font_run(english), english)

    def test_legacy_words_inside_a_line_of_english_are_cut_and_the_english_kept(self):
        line = ("vzvT[fs pzdB nfBe pko ;op ebk ;woE.. v'bB s/ okyj[ gqG{ BkBe d/ efo jE.. "
                "P. 256 'After wandering and wandering O Lord, I have come")
        self.assertEqual(strip_legacy_words(line), "P. 256 'After wandering and wandering O Lord, I have come")

    def test_english_punctuation_is_not_the_fonts(self):
        for english in ("Kindly save me, O Lord; kindly keep me in Thy refuge.",
                        "he/she may be; whose births whirls them; round and round",
                        "[The idea that I am the Creator.] and [Considering the world as Brahm]"):
            self.assertEqual(strip_legacy_words(english), english)


class LinkTranslationTests(unittest.TestCase):
    """A translation of a bani, linked to its shabads in order (29_link_translations.py)."""

    def link(self):
        return import_module("29_link_translations")

    def test_the_path_never_goes_back_and_moving_costs_something(self):
        seg = self.link().segment
        # paragraph 2 reads slightly more like shabad 0 again: noise, not a return
        sim = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.12, 0.1, 0.0], [0.0, 0.0, 0.5]]
        self.assertEqual(seg(sim, 0.15), [0, 1, 1, 2])
        path = seg([[0.3, 0.2], [0.2, 0.3], [0.3, 0.2]], 0.0)
        self.assertEqual(path, sorted(path))
        # a single paragraph's small preference does not pay for a move
        self.assertEqual(seg([[0.5, 0.0], [0.0, 0.05], [0.5, 0.0]], 0.15), [0, 0, 0])

    def test_two_renderings_of_one_line_share_their_content_words(self):
        words = self.link().words
        a = set(words("By bowing the head what is achieved, when at heart one goes about impure?"))
        b = set(words("What can be achieved by bowing the head, when the man goes with filthy heart?"))
        self.assertTrue({"bow", "head", "achiev", "heart"} <= a & b)
        self.assertNotIn("the", a)

    def test_a_translation_aligns_to_its_bani_in_order(self):
        m = self.link()
        corpus = {1: "fear of the lord the wind blows rivers flow fire works",
                  2: "the clay of the muslim falls into the potter lump bricks burned",
                  3: "bowing the head achieves nothing when the heart is impure filthy"}
        space = m.Space({k: m.words(v) for k, v in corpus.items()})
        dv = {k: space.vec(m.words(v)) for k, v in corpus.items()}
        book = ["In fear the wind blows and the rivers flow", "In fear the fire does its work",
                "The clay of a Mussulman finds its way into the potter's lump",
                "What is the use of bowing the head, when the heart is impure?"]
        sim = [[m.cos(space.vec(m.words(t)), dv[k]) for k in (1, 2, 3)] for t in book]
        self.assertEqual(m.segment(sim, 0.05), [0, 0, 1, 2])


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

    def test_the_forms_in_search_of_the_true_guru_uses(self):
        # the whole bracket is the citation, whatever else it names
        ang, rest = find_ang('"Says Nanak, sing this True Bani forever. || 23 ||" (Anand Sahib: SGGS p. 920)')
        self.assertEqual(ang, 920)
        self.assertNotIn("SGGS", rest)
        self.assertNotIn("Anand Sahib", rest)
        self.assertEqual(find_ang('"...on the palm of your hand." (Guru Nanak Dev Ji SGGS p. 1412).')[0], 1412)
        self.assertEqual(find_ang('"...the world is drowning." ( SGGSp. 27)')[0], 27)
        self.assertTrue(looks_quoted('"...furnace of the tenth gate." (SGGS p. 1123)'))
        # a bracket with the number missing cites nothing
        self.assertIsNone(find_ang('"Naam is the medicine for all types of disease." (Sukhmani: GGS p. )')[0])


class PageBreakTests(unittest.TestCase):
    """A verse quoted at the foot of a page arrives as two paragraphs."""

    def rec(self, unit_id, text, style="body"):
        return {"unit_id": unit_id, "work": "w", "part": None, "page": 1, "para_no": 1,
                "style": style, "text": text}

    def test_the_second_half_is_scored_with_the_first_in_front_of_it(self):
        first = self.rec("w:1", '"Where there is no mother, father, children, friends or brothers, O my mind,')
        second = self.rec("w:2", 'there, only the Naam shall be with you as your help and support. || 1 ||" (SGGS p. 264)')
        self.assertTrue(opened_by(first, second))
        spans = detect([first, second])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["unit_id"], "w:2")
        self.assertEqual(spans[0]["opened_by"], "w:1")
        self.assertEqual(spans[0]["ang"], 264)
        text = spans[0]["text"]
        self.assertLess(text.index("Where there is no mother"), text.index("help and support"))

    def test_a_page_of_dialogue_does_not_qualify(self):
        # the first paragraph closes its quotation, so the second is not its rest
        first = self.rec("w:1", '"Bhai Sahib Ji, is it important to live the life of a householder?"')
        second = self.rec("w:2", '"Yes, if an individual is seeking salvation." (SGGS p. 264)')
        self.assertFalse(opened_by(first, second))
        # and a paragraph that merely mentions a number is not a half either
        third = self.rec("w:3", "There were 700 of them.")
        self.assertFalse(opened_by(third, second))

    def test_only_a_paragraph_that_is_nothing_but_the_quotation_is_restyled(self):
        self.assertTrue(wholly('"Come, O beloved Sikhs of the True Guru. || 23 ||" (Anand Sahib: SGGS p. 920)'))
        self.assertTrue(wholly('“Sewa of the True Guru bears fruit only if done with total devotion." (SGGS p. 552)'))
        # a verse inside his own sentence keeps the sentence
        self.assertFalse(wholly('As Guru Nanak Dev Ji says: "contentment was the chariot." (SGGS p. 470) (See Yug.)'))
        self.assertFalse(wholly('"Science has proven that Gurbani has spread. As Guru Ji says: "fire is the chariot." (SGGS p. 470) (See also Yug in the Glossary.)'))


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


class CorpusIndexTests(unittest.TestCase):
    """lib/citations.corpus_by_ang on the scripture database that ships."""

    def test_a_corpus_without_translations_indexes_nothing_and_fails_nowhere(self):
        import sqlite3
        from lib.citations import corpus_by_ang
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE lines (line_id INTEGER, shabad_id INTEGER, ang INTEGER, kind TEXT)")
        self.assertEqual(corpus_by_ang(con), {})


class KeepSpanTests(unittest.TestCase):
    """What a roster keeps of a file that mostly retells another."""

    RECORDS = [{"text": t} for t in (
        "4. Leaving Santokh Daas", "Baba Ji did selfless seva...", "Darshan",
        "I was reading a in Se Kanehaa last night...",
        "Jesus: My only wish is that all my Panth go and follow Guru Nanak's teachings.",
        "5. The Search for Sikh 'Saints'", "32. What is Maya according to Gurbani?",
        "Maya, or mammon, is that ignorance...", "(p938)", "33. The Initiation of Langar")]

    def keep(self, spans):
        return [r["text"] for r in import_module("12_ingest_writings").keep_spans(
            self.RECORDS, spans, "summary.pdf")]

    def test_each_span_runs_from_the_paragraph_it_opens_with_to_the_one_it_closes_with(self):
        kept = self.keep([{"from": "Darshan", "to": "follow Guru Nanak's teachings."},
                          {"from": "32. What is Maya", "to": "(p938)"}])
        self.assertEqual(kept[0], "Darshan")
        self.assertEqual(len(kept), 6)
        self.assertNotIn("5. The Search for Sikh 'Saints'", kept)
        self.assertEqual(kept[-1], "(p938)")

    def test_an_end_that_is_not_found_is_an_error_not_an_empty_file(self):
        with self.assertRaises(ValueError):
            self.keep([{"from": "Darshan", "to": "no such ending"}])
        with self.assertRaises(ValueError):
            self.keep([{"from": "no such start", "to": "(p938)"}])


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

    def test_a_paragraph_quoting_several_verses_keeps_all_of_them(self):
        # The legacy-font resolver emits one citation per paragraph; the
        # transliteration one finds every verse in a paragraph, and Bandginama
        # runs three into one. Keyed singly, two of the three were overwritten
        # and lost with no error anywhere.
        records = [self.para("he writes -", no=1),
                   self.para("three verses run together here", style="quote", no=2)]
        cites = {"w:1:1:2": [
            {"shabad_id": 11, "line_id": 1, "ang": 239, "score": 1.0,
             "method": "skeleton", "span": "one"},
            {"shabad_id": 22, "line_id": 2, "ang": 982, "score": 1.0,
             "method": "skeleton", "span": "two"},
            {"shabad_id": 33, "line_id": 3, "ang": 1309, "score": 1.0,
             "method": "skeleton", "span": "three"}]}
        units = self.build(records, cites, target=100)
        self.assertEqual(sorted(c["ang"] for c in units[0]["cites"]), [239, 982, 1309])

    def test_a_verse_quoted_inside_a_prose_paragraph_is_still_cited(self):
        # Bhai Raghbir Singh quotes a tuk mid-sentence, and the resolver leaves
        # such a paragraph as prose on purpose so his own words stay
        # retrievable. Before this, only paragraphs restyled as quotes were
        # linked at all -- 35 citations of Bandginama's 80 instead of 77.
        records = [self.para("Disease and pain are shed dukh bharam dard bhau nasia "
                             "when the Creator dwells in our mind.", no=1)]
        cites = {"w:1:1:1": [{"shabad_id": 44, "line_id": 9, "ang": 240, "score": 1.0,
                              "method": "skeleton", "span": "dukh bharam dard bhau nasia"}]}
        units = self.build(records, cites, target=100)
        self.assertEqual(units[0]["cites"][0]["ang"], 240)
        # and the paragraph itself is kept, unlike a standalone quotation
        self.assertIn("Disease and pain", units[0]["text"])

    def test_the_older_one_citation_per_paragraph_shape_still_works(self):
        # iterating a dict yields its keys, so a bare citation must be wrapped
        # rather than trusted to be a list
        records = [self.para("he writes -", no=1),
                   self.para("a verse", style="quote", no=2)]
        cites = {"w:1:1:2": {"shabad_id": 7, "line_id": 1, "ang": 100, "score": 1.0,
                             "method": "letters", "span": "a verse"}}
        units = self.build(records, cites, target=100)
        self.assertEqual(units[0]["cites"][0]["shabad_id"], 7)

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


    # -- a verse and the prose that explains it (a translation, a commentary) --

    @staticmethod
    def explains(pair, shabad, line_from, line_to, ang=3, source="G"):
        return {"pair": pair, "shabad_id": shabad, "line_from": line_from, "line_to": line_to,
                "ang": ang, "source": source, "score": 0.9, "method": "ocr-corpus-match"}

    @staticmethod
    def cite(shabad, line_from, line_to, ang=3, source="G"):
        return {"shabad_id": shabad, "line_id": line_from, "line_from": line_from, "line_to": line_to,
                "ang": ang, "source": source, "role": "quotes", "score": 0.9,
                "method": "ocr-corpus-match", "span": ""}

    def test_a_verse_with_an_explanation_after_it_links_forward_to_that_explanation(self):
        # the Santhya's page: the preface, then a verse and its arth beside it.
        # The preface did not lead into the verse; the arth is about it.
        records = [self.para("The preface says what this pauri is about.", no=1),
                   self.para("ਮੂਲ", style="heading", no=2),
                   {**self.para("ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥", style="quote", no=3), "pair": 1},
                   {**self.para("Sochai: by washing, cleanliness of mind does not come.", no=4),
                    "pair": 1, "explains": [self.explains(1, 5, 10, 10)]}]
        cites = {"w:1:1:3": [self.cite(5, 10, 10)]}
        units = self.build(records, cites, target=100)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["cites"], [])
        self.assertEqual([(c["shabad_id"], c["role"]) for c in units[1]["cites"]], [(5, "explains")])
        self.assertIn("Sochai", units[1]["text"])

    def test_a_verse_no_prose_explains_is_still_the_previous_passage_s_quotation(self):
        # an essay quotes a tuk in passing: the pair number alone changes nothing
        records = [self.para("As the Guru says of washing -", no=1),
                   {**self.para("ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ॥", style="quote", no=2), "pair": 1},
                   self.para("and the essay carries on with its argument here.", no=3)]
        units = self.build(records, {"w:1:1:2": [self.cite(5, 10, 10)]}, target=100)
        self.assertEqual(len(units), 1)
        self.assertEqual([(c["shabad_id"], c["role"]) for c in units[0]["cites"]], [(5, "quotes")])

    def test_two_explanations_of_one_shabad_in_one_passage_widen_the_range(self):
        records = [{**self.para("First tuk explained.", no=1), "pair": 1, "explains": [self.explains(1, 5, 10, 11)]},
                   {**self.para("Second tuk explained.", no=2), "pair": 2, "explains": [self.explains(2, 5, 12, 13)]},
                   {**self.para("A quotation of the same shabad, in another role.", no=3)},
                   {**self.para("ਤੁਕ ॥", style="quote", no=4)}]
        cites = {"w:1:1:4": [self.cite(5, 20, 20)]}
        units = self.build(records, cites, target=100)
        self.assertEqual(len(units), 1)
        by_role = {c["role"]: (c["line_from"], c["line_to"]) for c in units[0]["cites"]}
        self.assertEqual(by_role, {"explains": (10, 13), "quotes": (20, 20)})

    def test_an_explanation_of_a_verse_the_merge_could_not_place_links_nothing_yet(self):
        records = [{**self.para("ਤੁਕ ॥", style="quote", no=1), "pair": 1},
                   {**self.para("Its explanation, three words long enough.", no=2), "pair": 1,
                    "explains": [{"pair": 1, "unmatched": True}]}]
        units = self.build(records, target=100)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["cites"], [])

    def test_an_explanation_running_on_from_the_page_before_links_to_the_same_verse(self):
        # page 2 opens with prose that continues page 1's arth (writings_ocr
        # marks it `continues`): it is about the same verse
        records = [{**self.para("ਤੁਕ ॥", style="quote", no=1), "pair": 1},
                   {**self.para("The arth begins on this page.", style="body", page=1, no=2), "pair": 1,
                    "explains": [self.explains(1, 5, 10, 10)]},
                   self.para("A heading between", style="heading", page=2, no=1),
                   {**self.para("and the arth carries on over the page.", page=2, no=2), "pair": 2,
                    "explains": [{**self.explains(1, 5, 10, 10), "pair": 2, "continues": True}]}]
        units = self.build(records, {"w:1:1:1": [self.cite(5, 10, 10)]}, target=100)
        self.assertEqual(len(units), 2)
        self.assertEqual([[(c["shabad_id"], c["role"]) for c in u["cites"]] for u in units],
                         [[(5, "explains")], [(5, "explains")]])


class SmallCorpusTests(unittest.TestCase):
    """14_embed_writings.py on a work of a few pages."""

    def test_a_corpus_smaller_than_the_index_keeps_as_many_dimensions_as_it_has_units(self):
        mod = import_module("14_embed_writings")
        self.assertEqual(mod.index_dim_for(3198, 384), 256)
        self.assertEqual(mod.index_dim_for(77, 384), 77)               # a twenty-page bench sample
        self.assertEqual(mod.index_dim_for(1, 384), 1)
        self.assertEqual(mod.index_dim_for(500, 128), 128)


class OutputContractTests(unittest.TestCase):
    """
    The shape of what the pipeline writes, which a server reads without asking.

    Nothing else in these suites would notice a file renamed, a column dropped
    or the rows renumbered: the scripts would run, every test would pass, and
    the corpus would fail to load somewhere else, later. These cases build a
    small corpus end to end -- with a stand-in for the embedding model, so no
    weights are needed -- and state what a reader of it depends on
    (docs/output-format.md; packages/search-core's corpus reader in the
    repository this one is exported from).

    ADDING a column, a manifest field or a record key passes. Renaming,
    removing or reordering one fails, and that is the point: change the
    reader and the document first, then these cases.
    """

    N = 300                     # PCA to 256 dimensions needs more rows than that
    EMBED_DIM, INDEX_DIM = 384, 256

    VECTOR_FILES = {            # name -> bytes, for N passages
        "units.i8": lambda n: n * OutputContractTests.INDEX_DIM,
        "units.scale.f32": lambda n: n * 4,
        "units.mask.u8": lambda n: n,
        "pca.components.f32": lambda n: OutputContractTests.INDEX_DIM * OutputContractTests.EMBED_DIM * 4,
        "pca.mean.f32": lambda n: OutputContractTests.EMBED_DIM * 4,
    }
    MANIFEST_FIELDS = {"kind", "index", "corpus", "db", "model", "model_dir", "tokenizer", "pooling",
                       "query_prefix", "doc_prefix", "pad_token", "pad_id", "lowercase", "strip_accents",
                       "max_len", "embed_dim", "index_dim", "units", "works", "citations",
                       "text_lang", "query_scripts", "files"}
    COLUMNS = {
        "works": ["work_id", "title", "title_en", "author", "folder", "original", "quote_policy",
                  "parts", "files", "units", "language", "licence", "kind", "translate", "ang_from", "ang_to"],
        "units": ["unit_row", "unit_id", "work_id", "part", "page", "para_no", "marker", "text", "text_src"],
        "citations": ["unit_row", "shabad_id", "line_id", "ang", "score", "method", "span"],
        "links": ["unit_row", "source", "shabad_id", "line_from", "line_to", "ang", "role", "kind",
                  "method", "score", "page"],
        "meta": ["key", "value"],
    }
    META_KEYS = {"corpus", "author", "authors", "units", "works", "citations", "links", "built"}
    RECORD_KEYS = {"unit_id", "work", "part", "essay", "page", "para_no", "marker",
                   "style", "italic", "text", "lang"}
    UNIT_KEYS = {"unit_row", "unit_id", "work", "part", "page", "para_no", "marker", "text",
                 "cites", "title", "author", "quote_policy"}

    @classmethod
    def setUpClass(cls):
        import contextlib
        import io
        import sqlite3
        import tempfile
        import numpy as np

        cls.tmp = tempfile.TemporaryDirectory()
        src = os.path.join(cls.tmp.name, "src")
        cls.out_dir = os.path.join(cls.tmp.name, "artifacts", "corpora", "writings-en")
        cls.db_path = os.path.join(cls.tmp.name, "artifacts", "writings.sqlite")
        cls.units_path = os.path.join(src, "units.jsonl")
        os.makedirs(src)

        # one work: a heading, a paragraph that leads into a verse, the verse,
        # and enough prose that every paragraph closes a unit of its own
        def rec(no, text, style="body", **extra):
            page, para = no // 10 + 1, no % 10 + 1
            return {"unit_id": "book:1:%d:%d" % (page, para), "work": "book", "part": 1, "essay": None,
                    "page": page, "para_no": para, "marker": None, "style": style,
                    "italic": style == "quote", "text": text, "lang": "en", **extra}
        records = [rec(0, "The Opening", style="heading"),
                   rec(1, "The Guru says this of the Name, as the verse below shows."),
                   rec(2, "By the Name alone is one carried across. 468", style="quote")]
        records += [rec(i, "Passage number %d speaks of the Name and nothing else." % i)
                    for i in range(3, cls.N + 2)]
        # a paired page (lib/writings_ocr): a verse, then the prose that explains
        # it, the two sharing a pair; the prose says which lines it explains
        records.append(rec(cls.N + 2, "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥", style="quote", pair=1))
        records.append(rec(cls.N + 3, "By washing, cleanliness of the mind does not come, though one wash a lakh times.",
                           pair=1, explains=[{"pair": 1, "shabad_id": 9, "line_from": 300, "line_to": 301, "ang": 1,
                                              "source": "G", "score": 0.9, "method": "ocr-corpus-match"}]))
        # a verse of another scripture, matched by the OCR merge (source D, its
        # own id space), quoted after the last prose
        records.append(rec(cls.N + 4, "ਦਸਮ ਬਾਣੀ ਦੀ ਇਕ ਤੁਕ ॥", style="quote"))
        with open(os.path.join(src, "book.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"_meta": {
                "work": "book", "title": "A Book", "author": "An Author", "original": True,
                "quote_policy": "verbatim", "files": ["book.pdf"], "language": "en",
                "licence": "public-domain", "source": "pdf-text"}}) + "\n")
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(src, "citations.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"_meta": {}}) + "\n")
            fh.write(json.dumps({"unit_id": records[2]["unit_id"], "shabad_id": 1712, "line_id": 20101,
                                 "ang": 468, "score": 0.91, "method": "lexical_ang",
                                 "span": "By the Name alone is one carried across."}) + "\n")
            fh.write(json.dumps({"unit_id": records[-3]["unit_id"], "shabad_id": 9, "line_id": 300,
                                 "line_ids": [300, 301], "line_from": 300, "line_to": 301, "source": "G",
                                 "ang": 1, "score": 0.9, "method": "ocr-corpus-match",
                                 "span": "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥"}) + "\n")
            fh.write(json.dumps({"unit_id": records[-1]["unit_id"], "shabad_id": 3, "line_id": 50,
                                 "line_ids": [50, 51], "line_from": 50, "line_to": 51, "source": "D",
                                 "ang": 12, "score": 0.95, "method": "ocr-corpus-match",
                                 "span": "ਦਸਮ ਬਾਣੀ ਦੀ ਇਕ ਤੁਕ ॥"}) + "\n")

        embed = import_module("14_embed_writings")
        build = import_module("15_build_writings_db")
        rng = np.random.default_rng(0)
        real = (embed.Embedder, embed.embed_long, sys.argv)
        # the model is the one thing a test cannot carry: what it returns is
        # stood in for, what is done with it is not
        embed.Embedder = lambda name, max_len=256: object()
        embed.embed_long = lambda emb, texts, max_len: (
            rng.standard_normal((len(texts), cls.EMBED_DIM)).astype(np.float32), 0)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sys.argv = ["14_embed_writings.py", "--src", src, "--out-dir", cls.out_dir,
                            "--model", "bge-small-en-v1.5", "--target-tokens", "3", "--hard-tokens", "40"]
                embed.main()
                sys.argv = ["15_build_writings_db.py", "--units", cls.units_path, "--out", cls.db_path]
                build.main()
        finally:
            embed.Embedder, embed.embed_long, sys.argv = real

        with open(os.path.join(cls.out_dir, "manifest.json"), encoding="utf-8") as fh:
            cls.manifest = json.load(fh)
        with open(cls.units_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        cls.head, cls.units = rows[0]["_meta"], rows[1:]
        cls.con = sqlite3.connect(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_the_vector_files_have_the_names_and_sizes_a_reader_maps(self):
        n = len(self.units)
        self.assertGreaterEqual(n, self.INDEX_DIM)
        for name, size in self.VECTOR_FILES.items():
            with self.subTest(file=name):
                path = os.path.join(self.out_dir, name)
                self.assertTrue(os.path.exists(path), name)
                self.assertEqual(os.path.getsize(path), size(n))
                self.assertEqual(self.manifest["files"][name], size(n))

    def test_the_manifest_carries_every_field_a_server_reads(self):
        self.assertEqual(self.MANIFEST_FIELDS - set(self.manifest), set())
        self.assertEqual(self.manifest["kind"], "documents")
        self.assertEqual(self.manifest["corpus"], "writings-en")
        self.assertEqual((self.manifest["embed_dim"], self.manifest["index_dim"]),
                         (self.EMBED_DIM, self.INDEX_DIM))
        self.assertEqual(self.manifest["units"], len(self.units))
        self.assertEqual(self.manifest["text_lang"], "en")
        self.assertEqual(self.manifest["query_scripts"], ["latin"])

    def test_the_database_has_the_tables_and_columns_a_reader_selects(self):
        for table, wanted in self.COLUMNS.items():
            with self.subTest(table=table):
                have = [r[1] for r in self.con.execute("PRAGMA table_info(%s)" % table)]
                self.assertEqual([c for c in wanted if c not in have], [])
        # works and units are inserted by position: the columns a reader knows
        # come first and in this order, and anything new goes after them
        for table in ("works", "units", "citations", "links"):
            with self.subTest(order=table):
                have = [r[1] for r in self.con.execute("PRAGMA table_info(%s)" % table)]
                self.assertEqual(have[:len(self.COLUMNS[table])], self.COLUMNS[table])
        keys = {k for (k,) in self.con.execute("SELECT key FROM meta")}
        self.assertEqual(self.META_KEYS - keys, set())

    def test_unit_row_is_the_row_in_the_vectors_and_nothing_renumbers_it(self):
        n = len(self.units)
        self.assertEqual([u["unit_row"] for u in self.units], list(range(n)))
        self.assertEqual([r for (r,) in self.con.execute("SELECT unit_row FROM units ORDER BY unit_row")],
                         list(range(n)))
        # the same passage under the same number in both
        for row in (0, n // 2, n - 1):
            (text,) = self.con.execute("SELECT text FROM units WHERE unit_row=?", (row,)).fetchone()
            self.assertEqual(text, self.units[row]["text"])

    def test_a_unit_and_its_work_say_what_a_passage_is_shown_with(self):
        self.assertEqual(self.UNIT_KEYS - set(self.units[0]), set())
        self.assertRegex(self.units[0]["unit_id"], r"^book:1:\d+:\d+$")
        self.assertEqual(self.head["corpus"], "writings-en")
        work = self.con.execute(
            "SELECT work_id, title, author, quote_policy, language, licence, units FROM works").fetchall()
        self.assertEqual(work, [("book", "A Book", "An Author", "verbatim", "en", "public-domain",
                                 len(self.units))])

    def test_a_match_against_another_scripture_keeps_its_source_and_stays_out_of_citations(self):
        last = self.units[-1]
        other = [c for c in last["cites"] if c.get("source") == "D"]
        self.assertEqual(len(other), 1)
        self.assertEqual((other[0]["line_from"], other[0]["line_to"], other[0]["shabad_id"], other[0]["role"]),
                         (50, 51, 3, "quotes"))
        # the Guru Granth Sahib cite says so too, by default
        self.assertTrue(all(c.get("source") == "G" for u in self.units[:-1] for c in u["cites"]))
        # citations.line_id is a Guru Granth Sahib id and nothing else goes in that column
        self.assertEqual(self.con.execute("SELECT count(*) FROM citations WHERE line_id = 50").fetchone()[0], 0)
        self.assertEqual(self.con.execute("SELECT count(*) FROM citations").fetchone()[0], 2)

    def test_a_verse_with_its_explanation_after_it_is_linked_to_the_explanation(self):
        # the verse's own citation and the prose's `explains` are one link, in
        # the explains role, on the passage that explains it -- not on the
        # passage before the verse, which was about something else
        (row,) = self.con.execute("SELECT unit_row FROM units WHERE text LIKE '%lakh times%'").fetchone()
        rows = self.con.execute(
            "SELECT unit_row, role, line_from, line_to FROM links WHERE shabad_id = 9").fetchall()
        self.assertEqual(rows, [(row, "explains", 300, 301)])
        self.assertEqual(self.units[row - 1]["cites"], [])

    def test_every_link_between_a_passage_and_the_scripture_is_in_links(self):
        rows = self.con.execute(
            "SELECT source, shabad_id, line_from, line_to, ang, role, kind, method FROM links ORDER BY source, shabad_id").fetchall()
        self.assertEqual(rows, [("D", 3, 50, 51, 12, "quotes", "essay", "ocr-corpus-match"),
                                ("G", 9, 300, 301, 1, "explains", "essay", "ocr-corpus-match"),
                                ("G", 1712, 20101, 20101, 468, "quotes", "essay", "lexical_ang")])
        self.assertEqual(self.con.execute("SELECT value FROM meta WHERE key='links'").fetchone()[0], "3")
        self.assertEqual(self.manifest["links"], 3)
        # the work says what it is, how far it goes, and which angs it covers: the
        # verses it explains (ang 1), not the one it quotes in passing (468)
        self.assertEqual(self.con.execute("SELECT kind, translate, ang_from, ang_to FROM works").fetchall(),
                         [("essay", 0, 1, 1)])

    def test_the_verse_is_a_citation_on_the_passage_that_led_into_it(self):
        rows = self.con.execute(
            "SELECT unit_row, shabad_id, line_id, ang, method FROM citations WHERE shabad_id = 1712").fetchall()
        self.assertEqual(len(rows), 1)
        unit_row, shabad_id, line_id, ang, method = rows[0]
        self.assertEqual((shabad_id, line_id, ang, method), (1712, 20101, 468, "lexical_ang"))
        (text,) = self.con.execute("SELECT text FROM units WHERE unit_row=?", (unit_row,)).fetchone()
        self.assertIn("as the verse below shows", text)
        # and the verse itself is in no passage's text
        (n,) = self.con.execute("SELECT count(*) FROM units WHERE text LIKE '%carried across%'").fetchone()
        self.assertEqual(n, 0)

    def test_where_a_corpus_lands_by_default(self):
        from lib.writings_works import db_name
        embed = import_module("14_embed_writings")
        self.assertEqual([db_name(c) for c in ("writings-en", "akj-en", "writings-pa", "treatises-pa")],
                         ["writings.sqlite", "akj.sqlite", "writings-pa.sqlite", "treatises-pa.sqlite"])
        self.assertEqual([embed.units_name(lang) for lang in ("en", "pa", "hi")],
                         ["units.jsonl", "units-pa.jsonl", "units-hi.jsonl"])
        self.assertEqual({lang: d["corpus"] for lang, d in embed.LANG_DEFAULTS.items()},
                         {"en": "writings-en", "pa": "writings-pa", "hi": "writings-hi"})

    def test_a_paragraph_record_has_the_keys_every_later_step_reads(self):
        ingest = import_module("12_ingest_writings")
        doc = {"path": "x.pdf", "body_size": 10.0, "pages": [{
            "page": 7, "marker": "12", "spread": False, "paragraphs": [
                {"text": "Prose.", "style": "body", "italic": False},
                {"text": "ਨਾਨਕ ਨਾਮੁ", "style": "quote", "italic": False, "line_ids": [20101, 20102],
                 "shabad_id": 1712, "ang": 468, "match_score": 0.97, "match_method": "exact"}]}]}
        real = ingest.read_source
        ingest.read_source = lambda path, meta, **kw: doc
        try:
            res = ingest.read_one("A-Book.pdf")
        finally:
            ingest.read_source = real
        prose, verse = res["records"]
        self.assertEqual(self.RECORD_KEYS - set(prose), set())
        self.assertEqual(prose["unit_id"], "%s:0:7:1" % prose["work"])
        self.assertEqual((prose["page"], prose["para_no"], prose["marker"]), (7, 1, "12"))
        self.assertIn(prose["style"], ("body", "heading", "quote", "footnote"))
        # what the OCR merge matched travels with the verse, under these names
        self.assertEqual((verse["line_ids"], verse["shabad_id"], verse["ang"]), ([20101, 20102], 1712, 468))
        self.assertNotIn("line_ids", prose)


class SharedCitationsTests(unittest.TestCase):
    def test_one_works_citations_are_replaced_and_the_folders_others_kept(self):
        import tempfile
        cite = import_module("13_resolve_citations")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "citations.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for r in ({"_meta": {}}, {"unit_id": "sikh-faith:0:12:3", "shabad_id": 1},      # the legacy resolver's
                          {"unit_id": "jiwan:1:4:2", "work": "jiwan", "shabad_id": 2}):
                    fh.write(json.dumps(r) + "\n")
            kept = cite.kept_citations(path, {"jiwan"})
            self.assertEqual([r["shabad_id"] for r in kept], [1])
            self.assertEqual(cite.kept_citations(os.path.join(d, "none.jsonl"), {"jiwan"}), [])


class LineCommentaryTests(unittest.TestCase):
    """30_line_commentary.py: a commentary printed line by line, as a teeka."""

    def test_each_line_gets_the_prose_that_explains_it_and_nothing_else(self):
        lc = import_module("30_line_commentary")
        def rec(no, text, style="body", line=None, ang=5, part=1):
            ex = [{"pair": no, "shabad_id": 26, "line_from": line, "line_to": line, "ang": ang, "source": "G"}] if line else []
            return {"part": part, "page": 143, "para_no": no, "style": style, "text": text, "explains": ex}
        records = [rec(1, "ਅਮੁਲ ਆਵਹਿ ਅਮੁਲ ਲੈ ਜਾਹਿ ॥", style="quote"),
                   rec(2, "(ਫੇਰ ਮਾਨ ਦਾਅਵੇ ਦਾ ਤਿਆਗ ਕਰਕੇ) ਮੁੱਲ ਰਹਿਤ ਆਉਂਦੇ ਹਨ", line=243),
                   rec(3, "ਤੇ ਅਮੁਲ ਲੈ ਜਾਂਦੇ ਹਨ।", line=243),
                   rec(4, "2੭", line=244),                                   # a stray numeral, not prose
                   rec(5, "ਸ਼ਬਦ ਦਾ ਭਾਵ", style="footnote", line=244),
                   rec(6, "ਓਅੰਕਾਰ ਤੋਂ ਵੇਦ ਰਚੇ ਗਏ।", line=40000, ang=929)]    # quoted in passing
        lines, seen = lc.line_texts(records, {1: [1, 53]})
        self.assertEqual(list(lines), [243])
        self.assertEqual(lines[243]["text"], "(ਫੇਰ ਮਾਨ ਦਾਅਵੇ ਦਾ ਤਿਆਗ ਕਰਕੇ) ਮੁੱਲ ਰਹਿਤ ਆਉਂਦੇ ਹਨ ਤੇ ਅਮੁਲ ਲੈ ਜਾਂਦੇ ਹਨ।")
        self.assertEqual(lines[243]["pages"], ["1:143"])
        self.assertEqual((seen["not prose"], seen["outside the volume's angs"], seen["taken"]), (1, 1, 2))
        self.assertIn(40000, lc.line_texts(records)[0])                        # unbounded without the manifest


class KeepTests(unittest.TestCase):
    """
    14 --keep: a scanned book joins a corpus whose other works were built on
    another machine (Baru Sahib's PDFs), carried over from its database.
    """

    def test_the_built_works_are_kept_as_they_were_and_the_book_is_added_after_them(self):
        import contextlib
        import io
        import sqlite3
        import tempfile
        import numpy as np

        embed = import_module("14_embed_writings")
        build = import_module("15_build_writings_db")
        with tempfile.TemporaryDirectory() as d:
            # the published database: no links table, files a count, a pooled article's section
            old = os.path.join(d, "old.sqlite")
            con = sqlite3.connect(old)
            con.executescript("""
                CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, title_en TEXT, author TEXT, folder TEXT,
                  original INTEGER, quote_policy TEXT, parts TEXT, files INTEGER, units INTEGER, language TEXT, licence TEXT);
                CREATE TABLE units (unit_row INTEGER PRIMARY KEY, unit_id TEXT, work_id TEXT, part INTEGER, page INTEGER,
                  para_no INTEGER, marker TEXT, text TEXT, text_src TEXT, section TEXT);
                CREATE TABLE citations (unit_row INTEGER, shabad_id INTEGER, line_id INTEGER, ang INTEGER, score REAL,
                  method TEXT, span TEXT);""")
            con.executemany("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                ("articles", "Articles", None, "Banner", "root", 1, "verbatim", "[]", 17, 2, "en", None),
                ("old-book", "An Old Book", None, "Banner", "root", 1, "verbatim", "[]", 1, 1, "en", None)])
            con.executemany("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (0, "articles:1:1:1", "articles", 1, 1, 1, None, "Aim of life.", None, "Aim of Life, by X"),
                (1, "articles:2:1:1", "articles", 2, 1, 1, None, "Grammar of life.", None, "Grammar of Life"),
                (2, "old-book:0:4:2", "old-book", None, 4, 2, None, "Rebuilt from src, not kept.", None, None)])
            con.execute("INSERT INTO citations VALUES (1, 1712, 20101, 468, 0.9, 'lexical_ang', 'carried across')")
            con.commit()
            con.close()

            src = os.path.join(d, "src")
            os.makedirs(src)

            def write(name, meta, recs):
                with open(os.path.join(src, name), "w", encoding="utf-8") as fh:
                    if meta:
                        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
                    for r in recs:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            def rec(work, page, text, lang):
                return {"unit_id": "%s:1:%d:1" % (work, page), "work": work, "part": 1, "essay": None, "page": page,
                        "para_no": 1, "marker": None, "style": "body", "italic": False, "text": text, "lang": lang}
            write("jiwan.jsonl", {"work": "jiwan", "title": "ਜੀਵਨ", "title_en": "The Life", "author": "Banner",
                                  "original": True, "quote_policy": "summarise", "files": ["v1.pdf", "v2.pdf"],
                                  "language": "pa", "licence": "copyright", "source": "ocr"},
                  [rec("jiwan", 1, "ਸੰਤ ਜੀ ਆਏ।", "pa"), rec("jiwan", 2, "ਕੀਰਤਨ ਹੋਇਆ।", "pa")])
            write("jiwan.en.jsonl", {"work": "jiwan"},
                  [{"unit_id": "jiwan:1:1:1", "en": "Sant Ji came."}, {"unit_id": "jiwan:1:2:1", "en": "Kirtan was sung."}])
            write("old-book.jsonl", {"work": "old-book", "title": "An Old Book", "author": "Banner", "original": True,
                                     "quote_policy": "verbatim", "files": ["b.pdf"], "language": "en"},
                  [rec("old-book", 4, "The old book, read again.", "en")])

            rng = np.random.default_rng(0)
            real = (embed.Embedder, embed.embed_long, sys.argv)
            embed.Embedder = lambda name, max_len=256: object()
            embed.embed_long = lambda emb, texts, max_len: (rng.standard_normal((len(texts), 384)).astype(np.float32), 0)
            units_path, db = os.path.join(src, "units.jsonl"), os.path.join(d, "new.sqlite")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    sys.argv = ["14", "--src", src, "--out-dir", os.path.join(d, "vec"), "--corpus", "banner-en",
                                "--model", "bge-small-en-v1.5", "--translations", "--keep", old,
                                "--target-tokens", "1", "--hard-tokens", "40"]
                    embed.main()
                    sys.argv = ["15", "--units", units_path, "--out", db]
                    build.main()
            finally:
                embed.Embedder, embed.embed_long, sys.argv = real

            con = sqlite3.connect(db)
            rows = con.execute("SELECT unit_row, unit_id, work_id, text, text_src, section FROM units "
                               "ORDER BY unit_row").fetchall()
            # the kept articles first, as built; then the src works, the stale copy of old-book not among them
            self.assertEqual(rows[0], (0, "articles:1:1:1", "articles", "Aim of life.", None, "Aim of Life, by X"))
            self.assertEqual(rows[1][1:4], ("articles:2:1:1", "articles", "Grammar of life."))
            self.assertEqual([r[2] for r in rows], ["articles", "articles", "jiwan", "jiwan", "old-book"])
            self.assertNotIn("Rebuilt from src, not kept.", [r[3] for r in rows])
            self.assertEqual(rows[2][3:5], ("Sant Ji came.", "ਸੰਤ ਜੀ ਆਏ।"))
            # the kept citation stays on its passage
            self.assertEqual(con.execute("SELECT unit_row, shabad_id, span FROM citations").fetchall(),
                             [(1, 1712, "carried across")])
            works = {w[0]: w[1:] for w in con.execute(
                "SELECT work_id, title_en, original, quote_policy, files, units, language FROM works")}
            self.assertEqual(works["articles"], (None, 1, "verbatim", 17, 2, "en"))
            # a machine's English is not the author's own
            self.assertEqual(works["jiwan"], ("The Life", 0, "summarise", 2, 2, "pa"))
            self.assertEqual(os.path.getsize(os.path.join(d, "vec", "units.scale.f32")), 5 * 4)
            con.close()


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


ROSTERS = os.path.join(ROOT, "pipeline", "python", "rosters")


class RosterTests(unittest.TestCase):
    """
    What each roster says about attribution, asserted rather than assumed.

    A roster exists because writings_works.py:100 infers `original = folder ==
    "root"`, and a flat folder of PDFs counts as root. Left to that rule every
    book in these four folders would have been marked the author's own English
    and quotable verbatim -- five AKJ translations, twelve of Bhai Vir Singh's,
    Bandginama. Puran Singh is the one author for whom `verbatim` is true, and
    it is written down here rather than arrived at by accident.
    """

    POLICY = {"akj": "summarise", "puran": "verbatim",
              "virsingh": "summarise", "raghbir": "summarise",
              "bariaran": "summarise", "rama": "summarise",
              "rampurkhera": "summarise", "barusahib": "verbatim", "ratwara": "summarise"}

    @classmethod
    def setUpClass(cls):
        # the rosters name the private corpora's files and do not ship with
        # the public pipeline (tools/export-ingest.mjs), where a book is
        # described by a manifest.json instead
        if not os.path.isdir(ROSTERS):
            raise unittest.SkipTest("no rosters/ in this checkout")

    def roster(self, name):
        with open(os.path.join(ROSTERS, name + ".json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_every_roster_states_its_quote_policy_explicitly(self):
        for name, policy in self.POLICY.items():
            with self.subTest(roster=name):
                self.assertEqual(self.roster(name)["quote_policy"], policy)

    def test_a_translated_corpus_is_never_marked_the_authors_own_words(self):
        for name, policy in self.POLICY.items():
            if policy == "verbatim":
                continue
            for fname, work in self.roster(name)["works"].items():
                with self.subTest(roster=name, file=fname):
                    self.assertNotEqual(work.get("original"), True)

    def test_every_work_names_a_slug_and_a_title(self):
        for name in self.POLICY:
            for fname, work in self.roster(name)["works"].items():
                with self.subTest(roster=name, file=fname):
                    self.assertTrue(work.get("work"))
                    self.assertTrue(work.get("title"))

    def test_a_held_book_says_why_and_carries_the_measurement(self):
        # Holding is not deleting: the entry stays, so the omission is visible
        # where the roster is read and a better scan just replaces the file.
        held = {f: w for f, w in self.roster("virsingh")["works"].items() if w.get("hold")}
        self.assertEqual(len(held), 3, "the 1.0%% gate holds three of the twelve")
        for fname, work in held.items():
            with self.subTest(file=fname):
                self.assertGreater(work["glue_pct"], 1.0)
                self.assertIn("glue", work["hold"])
        kept = [w for w in self.roster("virsingh")["works"].values() if not w.get("hold")]
        self.assertEqual(len(kept), 9)

    def test_the_akj_second_author_is_named_on_his_own_book(self):
        # An answer names the author of the passage it used, so a book by
        # somebody else inside one author's folder has to say so.
        works = self.roster("akj")["works"]
        self.assertEqual(works["GodasviewedthroughGurbaniandScience.pdf"]["author"],
                         "Subedar Dharam Singh Sujjon")


if __name__ == "__main__":
    unittest.main()
