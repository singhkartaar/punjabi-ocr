"""
Tests for the OCR ingestion modules (20_ocr_pages.py .. 24_ocr_eval.py).

Everything here is synthetic: made-up Gurmukhi strings, hand-built line
boxes, a PDF drawn on the fly. No book text and no engine binary is needed.
The one place a real dependency is exercised (PyMuPDF rendering) skips when
the library is absent.
"""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ocr_text import cer, graphemes, match_key, normalise, script_of, strip_footnote_marker, wer, words
from lib.ocr_zones import classify_zones, header_hints, header_of, is_stamp, page_columns
from lib.ocr_match import CorpusIndex, match_line, match_text
from lib.ocr_route import Budget, needs_arbiter, page_needs_arbiter
from lib.ocr_engines import TesseractEngine, VisionEngine, number, smart_resize, split_block
from lib.ocr_pages import parse_pages
from lib.writings_manifest import list_sources, load_manifest, parse_source

SIHARI, U_SIGN = "\u0a3f", "\u0a41"
URA, U = "\u0a73", "\u0a09"


def line(y0, y1, text, x0=100, x1=900, zone=None, words_=None, conf=0.9):
    return {"bbox": [x0, y0, x1, y1], "text": text, "conf": conf, "words": words_ or [], "zone": zone}


class NormaliseTests(unittest.TestCase):
    def test_carrier_becomes_independent_vowel(self):
        self.assertEqual(normalise(URA + U_SIGN + "ਪਰ"), U + "ਪਰ")

    def test_leading_sihari_moves_after_its_consonant(self):
        self.assertEqual(normalise(SIHARI + "ਸਮਰਿ"), "ਸ" + SIHARI + "ਮਰਿ")

    def test_medial_sihari_between_consonants_is_left_alone(self):
        # ਸਿਮ is valid logical order; only the engine-specific fixer may move it
        self.assertEqual(normalise("ਸਿਮ"), "ਸਿਮ")

    def test_nfc_applies_to_both_sides(self):
        self.assertEqual(normalise("\u0a36"), normalise("\u0a38\u0a3c"))    # ਸ਼ either way

    def test_idempotent_and_whitespace(self):
        s = normalise("  ਨਾਮੁ   ਜਪੋ \n")
        self.assertEqual(s, "ਨਾਮੁ ਜਪੋ")
        self.assertEqual(normalise(s), s)

    def test_graphemes_keep_marks_with_their_base(self):
        self.assertEqual(graphemes("ਨਿਰਭਉ"), ["ਨਿ", "ਰ", "ਭ", "ਉ"])
        self.assertEqual(len(graphemes(normalise("\u0a36ਬਦ"))), 3)      # ਸ਼ is one cluster

    def test_match_key_folds_counters_and_nasal(self):
        k1 = match_key("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ ॥੧॥ ਰਹਾਉ ॥")
        k2 = match_key("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ")
        self.assertEqual(k1, k2)
        self.assertEqual(match_key("ਸੰਤ"), match_key("ਸਂਤ"))

    def test_footnote_marker(self):
        self.assertEqual(strip_footnote_marker("ਸ਼ਬਦ੧"), ("ਸ਼ਬਦ", "੧"))
        self.assertEqual(strip_footnote_marker("word12"), ("word", "12"))
        self.assertEqual(strip_footnote_marker("1930"), ("1930", None))
        self.assertEqual(strip_footnote_marker("੧"), ("੧", None))

    def test_script(self):
        self.assertEqual(script_of("ਨਾਮੁ"), "gurmukhi")
        self.assertEqual(script_of("ਨਾਮੁ (Declension)"), "mixed")
        self.assertEqual(script_of("12 ."), "none")

    def test_cer_counts_a_dropped_matra_as_one(self):
        self.assertAlmostEqual(cer("ਨਿਰਭਉ", "ਨਰਭਉ"), 0.25)
        self.assertEqual(cer("abc", "abc"), 0.0)
        self.assertEqual(wer("a b c", "a x c"), 1 / 3)
        self.assertEqual(words("ਨਾਮੁ, ਜਪੋ ॥"), ["ਨਾਮੁ", "ਜਪੋ"])


class ConjunctTests(unittest.TestCase):
    """A subjoined ra misread on a grey scan is one confusion from the word."""

    def test_a_halant_joins_its_cluster_to_the_next(self):
        from lib.ocr_text import conjuncts
        self.assertEqual(conjuncts("ਪ੍ਰੇਮ"), ["ਪ੍ਰੇ", "ਮ"])
        self.assertEqual(conjuncts("ਸ੍ਰੀ"), ["ਸ੍ਰੀ"])
        self.assertEqual(conjuncts(normalise("ਪ੍ਰਸ਼ਾਦ")), ["ਪ੍ਰ", "ਸ਼ਾ", "ਦ"])
        self.assertEqual(conjuncts(normalise("ਪੁਸ਼ਾਦ")), ["ਪੁ", "ਸ਼ਾ", "ਦ"])
        self.assertEqual(conjuncts("ਨਿਰਭਉ"), graphemes("ਨਿਰਭਉ"))       # no halant: the same clusters
        self.assertEqual(graphemes("ਪ੍ਰੇਮ")[0], "ਪ੍")                     # CER and the vote keep theirs

    def test_the_subjoined_ra_misreads_are_repaired(self):
        from lib.ocr_correct import Corrector
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(modern=["ਪ੍ਰਸ਼ਾਦ", "ਸ੍ਰੀ", "ਗ੍ਰੰਥ", "ਪ੍ਰੇਮ", "ਪ੍ਰੇਮੀ", "ਪ੍ਰਕਾਸ਼"])
        c = Corrector(lex)
        for bad, good in [("ਪੁਸ਼ਾਦ", "ਪ੍ਰਸ਼ਾਦ"), ("ਸੁੀ", "ਸ੍ਰੀ"), ("ਸੀ", "ਸ੍ਰੀ"), ("ਗੁੰਥ", "ਗ੍ਰੰਥ"),
                          ("ਗੰਥ", "ਗ੍ਰੰਥ"), ("ਪੁੇਮ", "ਪ੍ਰੇਮ"), ("ਪੁਕਾਸ਼", "ਪ੍ਰਕਾਸ਼")]:
            self.assertEqual(c.correct_word(bad), (good, "confusion"), bad)
        self.assertEqual(c.correct_word("ਪ੍ਰੇਮੀ"), ("ਪ੍ਰੇਮੀ", None))             # known: never touched
        known = Corrector(Lexicon.from_words(gurbani=["ਸੀ"], modern=["ਸ੍ਰੀ"]))
        self.assertEqual(known.correct_word("ਸੀ"), ("ਸੀ", None))                # ਸੀ "was" is a word

    def test_marks_are_priced_one_by_one(self):
        from lib.ocr_correct import Confusions, MARK_COST, STACKED
        c = Confusions()
        self.assertEqual(c.sub("ਗੁੰ", "ਗ੍ਰੰ"), 0.5)                  # one seeded confusion
        self.assertEqual(c.sub("ਗੁੰ", "ਗਾ"), 1.0)                     # two marks changed: a whole substitution
        self.assertEqual(c.sub("ਕਿ", "ਕੇ"), MARK_COST)                # one unseeded mark, as before
        self.assertEqual(c.sub("ਸੁੀ", "ਸ੍ਰੀ"), STACKED)               # a stacked aunkar is a subjoined ra
        self.assertEqual(c.sub("ਸੁੀ", "ਸੀ"), MARK_COST)               # and dropping it is no seeded confusion

    def test_the_book_vocabulary_learns_agreement_and_refuses_shared_misreads(self):
        from lib.ocr_correct import subjoin_variants
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(gurbani=["ਪ੍ਰਸਾਦਿ"])
        pages = [["ਮਸਤੂਆਣੇ ਸੁੀ ਪੁਸਾਦਿ", "ਮਸਤੂਆਣੇ ਸੀ ਪੁਸਾਦਿ"]] * 3           # three pages, two engines each
        refuse = lambda w: not lex.knows(w) and any(lex.knows(v) for v in subjoin_variants(w))
        lex.add_book_vocab(pages, refuse=refuse)
        self.assertIn("ਮਸਤੂਆਣੇ", lex.book)                                # agreed, three pages: the author's word
        self.assertNotIn("ਸੁੀ", lex.book)                                  # the engines disagree
        self.assertNotIn("ਪੁਸਾਦਿ", lex.book)                               # they agree, but ਪ੍ਰਸਾਦਿ is known
        self.assertEqual(subjoin_variants("ਬੁਹਮ"), ["ਬ੍ਰਹਮ"])
        from lib.ocr_correct import subjoin_repairs
        self.assertIn("ਗ੍ਰੰਥ", subjoin_repairs("ਗੰਥ"))                         # the dropped subjoin too
        from lib.ocr_correct import shared_misread
        lex2 = Lexicon.from_words(modern=["ਗ੍ਰੰਥ", "ਪ੍ਰਿਆ"])
        pages = [["ਗੰਥ ਸਾਹਿਬ ਪਿਆ", "ਗੰਥ ਸਾਹਿਬ ਪਿਆ"]] * 3 + [["ਗੁੰਥ", "ਗੰਥ"]]
        seen = {w for p in pages for t in p for w in t.split()}
        lex2.add_book_vocab(pages, refuse=lambda w: shared_misread(w, lex2.knows, seen))
        self.assertNotIn("ਗੰਥ", lex2.book)                                     # ਗੁੰਥ was read too
        self.assertIn("ਪਿਆ", lex2.book)                                        # ਪੁਿਆ never: ਪ੍ਰਿਆ proves nothing
        self.assertIn("ਸਾਹਿਬ", lex2.book)

    def test_an_agreed_word_is_corrected_only_on_strict_terms(self):
        from lib.ocr_correct import Corrector, is_subjoin_repair
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(gurbani=["ਗੁਰਿ", "ਪ੍ਰਸਾਦਿ"], modern=["ਪ੍ਰਸਾਦ", "ਗ੍ਰੰਥ", "ਚਲਣ"])
        c = Corrector(lex)
        text, ch = c.correct_line("ਕੜਾਹ-ਪੁਸ਼ਾਦ, ਗੁੰਥ। ਚੱਲਣ ਪੁਸਾਦਿ", only=set(),
                                  agreed={"ਕੜਾਹ-ਪੁਸ਼ਾਦ", "ਗੁੰਥ", "ਚੱਲਣ", "ਪੁਸਾਦਿ"})
        self.assertEqual(text, "ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ, ਗ੍ਰੰਥ। ਚੱਲਣ ਪ੍ਰਸਾਦਿ")         # part by part, punctuation kept
        self.assertEqual([x[:2] for x in ch], [["ਪੁਸ਼ਾਦ", "ਪ੍ਰਸ਼ਾਦ"], ["ਗੁੰਥ", "ਗ੍ਰੰਥ"], ["ਪੁਸਾਦਿ", "ਪ੍ਰਸਾਦਿ"]])
        self.assertTrue(all(x[2] == "confusion-agreed" for x in ch))
        # ਚੱਲਣ -> ਚਲਣ is a seeded confusion too (a dropped addak), and the
        # author's spelling: an agreed word receives the subjoin repair only
        self.assertEqual(c.correct_word("ਚੱਲਣ"), ("ਚਲਣ", "confusion"))
        for a, b, ok in [("ਗੁਰਪੁਸਾਦਿ", "ਗੁਰਪ੍ਰਸਾਦਿ", True), ("ਗੰਥ", "ਗ੍ਰੰਥ", True), ("ਸੁੀ", "ਸ੍ਰੀ", True),
                         ("ਪੂਦੇਸ਼", "ਪ੍ਰਦੇਸ਼", True),
                         ("ਚੱਲਣ", "ਚਲਣ", False), ("ਖਿਸਕਣ", "ਖਿਸਕਣਾ", False)]:
            self.assertEqual(is_subjoin_repair(a, b), ok, a)
        # a disputed word before a danda is corrected now (it was compared with its danda)
        self.assertEqual(c.correct_line("ਗੁੰਥ।", only={"ਗੁੰਥ"})[0], "ਗ੍ਰੰਥ।")

    def test_a_tie_goes_to_the_word_the_corpus_uses(self):
        from lib.ocr_correct import Corrector
        from lib.ocr_lexicon import Lexicon
        words_ = ["ਬ੍ਰਹਮ", "ਬਹਮ", "ਪ੍ਰੇਮ", "ਪੇਸ"]
        self.assertEqual(Corrector(Lexicon.from_words(modern=words_)).correct_word("ਬੁਹਮ"), ("ਬੁਹਮ", "ambiguous"))
        lex = Lexicon.from_words(modern=words_, freq={"ਬ੍ਰਹਮ": 240, "ਬਹਮ": 0, "ਪ੍ਰੇਮ": 300, "ਪੇਸ": 2})
        c = Corrector(lex)
        self.assertEqual(c.correct_word("ਬੁਹਮ"), ("ਬ੍ਰਹਮ", "confusion"))
        self.assertEqual(c.correct_word("ਪੇਮ"), ("ਪ੍ਰੇਮ", "confusion"))
        even = Corrector(Lexicon.from_words(modern=words_, freq={"ਬ੍ਰਹਮ": 40, "ਬਹਮ": 12}))
        self.assertEqual(even.correct_word("ਬੁਹਮ")[1], "ambiguous")              # not ten times as often: no call

    def test_the_stem_is_repaired_and_the_ending_kept(self):
        from lib.ocr_correct import Corrector, is_subjoin_repair
        from lib.ocr_lexicon import Lexicon
        c = Corrector(Lexicon.from_words(modern=["ਪ੍ਰਸਾਦ"]))
        self.assertEqual(c.correct_word("ਪੁਸ਼ਾਦਾ"), ("ਪ੍ਰਸ਼ਾਦਾ", "confusion"))       # ਪ੍ਰਸ਼ਾਦਾ is known only as ਪ੍ਰਸਾਦ + ਾ
        ok = lambda b: is_subjoin_repair("ਪੁਸ਼ਾਦਾ", b)
        self.assertEqual(c.correct_word("ਪੁਸ਼ਾਦਾ", max_cost=0.5, allow=ok), ("ਪ੍ਰਸ਼ਾਦਾ", "confusion"))

    def test_two_spellings_of_one_word_are_one_candidate(self):
        from lib.ocr_correct import Corrector
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(modern=["ਪ੍ਰਸਾਦ"])
        lex.book = {"ਪ੍ਰਸ਼ਾਦ"}                                                  # the Kosh's and the book's
        self.assertEqual(Corrector(lex).correct_word("ਪੁਸ਼ਾਦ"), ("ਪ੍ਰਸ਼ਾਦ", "confusion"))

    def test_the_nukta_is_folded_for_lookup_and_kept_in_the_text(self):
        from lib.ocr_correct import Corrector
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(modern=["ਪ੍ਰਸਾਦ", "ਸੁ"])
        self.assertTrue(lex.knows("ਪ੍ਰਸ਼ਾਦ"))                          # the Kosh's spelling, a modern book's nukta
        self.assertFalse(lex.knows("ਸੁੀ"))                             # not ਸੁ + ੀ: stacked signs are no inflection
        self.assertTrue(lex.knows("ਸੁ-ਪ੍ਰਸ਼ਾਦ") and not lex.knows("ਸੁ-ਸੁੀ"))   # a compound, by its parts
        self.assertEqual(Corrector(lex).correct_word("ਪੁਸ਼ਾਦ"), ("ਪ੍ਰਸ਼ਾਦ", "confusion"))


class ZoneTests(unittest.TestCase):
    def test_santhya_header(self):
        h = header_hints("ਸੋਦਰ ਆਸਾ ਮ:੧ ( ੧੮੩ ) (ਰਹਿਰਾਸ-ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ ੮-੯")
        self.assertEqual((h["ang_from"], h["ang_to"], h["book_page"]), (8, 9, 183))
        self.assertEqual(h["section"], "ਸੋਦਰ ਆਸਾ ਮ:੧")

    def test_single_ang_and_plain_page_number(self):
        self.assertEqual(header_hints("ਜਪੁ ( ੨੩ ) ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ ੨")["ang_to"], 2)
        h = header_hints("ਗੁਰਬਾਣੀ ਵਿਆਕਰਣ    ੬੧")
        self.assertEqual((h["ang_from"], h["book_page"]), (None, 61))

    def test_stamps(self):
        self.assertTrue(is_stamp("Page 41 of 530"))
        self.assertTrue(is_stamp("www.sikhbookclub.com"))
        self.assertTrue(is_stamp("- ਕ - ! Page 8 of 530"))
        self.assertFalse(is_stamp("ਸਾਧਨ ਪੱਖ"))
        self.assertFalse(is_stamp("the account continues on page 8 of 530 in the second volume of the history"))

    def test_a_translation_that_loops_or_balloons_is_rejected(self):
        from importlib import import_module
        mod = import_module("26_translate_writings")
        self.assertIsNone(mod.check("A short sentence.", "ਇਕ ਛੋਟਾ ਵਾਕ।"))
        loop = " ".join(["This is the translation of the given Punjabi text to English:"] * 5)
        self.assertIn("longer", mod.check(loop, "ਤਰਜਮਾ:-"))
        self.assertIn("longer", mod.check("word " * 60, "ਸ਼ਬਦ"))
        self.assertEqual(mod.check(loop, loop), "repeats itself")

    def test_zones_by_geometry(self):
        lines = [line(20, 50, "ਜਪੁ ( ੨੩ ) ਗੁਰੂ ਗ੍ਰੰਥ ਪੰਨਾ ੨-੩"), line(400, 440, "ਭੈ ਕਾਹੂ ਕਉ ਦੇਤ ਨਹਿ"),
                 line(2900, 2930, "੧ ਨਿਰਮਲ ਭਉ"), line(3200, 3230, "Page 41 of 530"), line(3100, 3130, "੨੩")]
        out = classify_zones(lines, 2560, 3300, rule_y=2850)
        self.assertEqual([ln["zone"] for ln in out], ["header", "body", "footnote", "stamp", "pageno"])
        self.assertEqual(header_of(out)["ang_from"], 2)

    def test_stamp_by_box_and_footnote_without_rule(self):
        stamps = [{"bbox": [1000, 3190, 1500, 3240], "text": "Page 41 of 530"}]
        lines = [line(400, 450, "ਸਾਧਨ ਪੱਖ"), line(500, 550, "ਮੂਲ ਮੰਤ੍ਰ ਦੇ ਭਾਵ"),
                 line(2900, 2930, "੧ ਨਿਰਮਲ ਭਉ ਪਾਇਆ"), line(2940, 2970, "ਹਰਿ ਗੁਣ ਗਾਇਆ"),
                 line(3195, 3235, "Paqe 41 of 53O", x0=1010, x1=1490)]
        out = classify_zones(lines, 2560, 3300, stamps=stamps)
        self.assertEqual([ln["zone"] for ln in out], ["body", "body", "footnote", "footnote", "stamp"])

    def test_engine_zone_is_kept(self):
        out = classify_zones([line(1500, 1540, "ਪੰਨਾ", zone="header")], 2560, 3300)
        self.assertEqual(out[0]["zone"], "header")

    def test_columns(self):
        left = [{"bbox": [100 + (i % 7) * 60, 0, 0, 0], "text": "word"} for i in range(60)]
        right = [{"bbox": [1400 + (i % 7) * 60, 0, 0, 0], "text": "word"} for i in range(60)]
        self.assertEqual(len(page_columns(left + right, 2560)), 2)
        self.assertEqual(page_columns(left, 2560), [])
        wide = [{"bbox": [100 + i * 20, 0, 0, 0], "text": "word"} for i in range(100)]
        self.assertEqual(page_columns(wide, 2560), [])

    def test_a_book_s_own_header_pattern(self):
        pat = r"(?P<section>.+?)\s+\|\s+p\.\s*(?P<book_page>\d+)\s+\|\s+ang\s+(?P<ang_from>\d+)(?:-(?P<ang_to>\d+))?"
        h = header_hints("Sukhmani Sahib | p. 42 | ang 262-263", pat)
        self.assertEqual((h["section"], h["book_page"], h["ang_from"], h["ang_to"]), ("Sukhmani Sahib", 42, 262, 263))
        self.assertEqual(header_hints("Sukhmani Sahib | p. 43 | ang 263", pat)["ang_to"], 263)
        self.assertEqual(header_hints("no header here", pat)["ang_from"], None)
        lines = [line(20, 50, "Sukhmani Sahib | p. 42 | ang 262-263", zone="header")]
        self.assertEqual(header_of(lines, pat)["book_page"], 42)

    def test_misread_angs_are_put_right_by_their_neighbours(self):
        from lib.ocr_zones import smooth_hints
        # a book that advances one ang every five pages, with the Santhya's own
        # misreads put in: 1 read as 7, 3 as 92, 7 as 491, one header missing,
        # and one page number read as 900
        truth = {p: 1 + (p - 1) // 5 for p in range(1, 61)}
        read = {**truth, 4: 7, 12: 92, 35: 491, 8: None}
        raw = {p: {"ang_from": a, "ang_to": a, "book_page": (p + 10 if p != 6 else 900), "section": None}
               for p, a in read.items()}
        out = smooth_hints(raw)
        self.assertEqual({p: out[p]["ang_from"] for p in out}, truth)
        self.assertEqual({p for p in out if out[p]["ang_source"] == "smoothed"}, {4, 8, 12, 35})
        self.assertTrue(all(out[p]["ang_to"] >= out[p]["ang_from"] for p in out))
        # the book's page number too: one offset for the whole book
        self.assertEqual(out[6]["book_page"], 16)
        # idempotent, sources included
        self.assertEqual(smooth_hints(out), out)
        # a book that gives no headers at all says so and changes nothing
        none = smooth_hints({1: {"ang_from": None, "ang_to": None, "book_page": None, "section": None}})
        self.assertEqual((none[1]["ang_from"], none[1]["ang_source"]), (None, None))

    def test_a_hairline_rule_is_found_and_a_bar_or_a_blank_page_is_not(self):
        import numpy as np
        from lib.ocr_zones import find_vertical_rule
        page = np.full((1400, 1000), 255, dtype=np.uint8)
        self.assertIsNone(find_vertical_rule(page))
        ruled = page.copy()
        ruled[150:1250, 399:402] = 0                                  # 3 px, the height of the body
        self.assertEqual(find_vertical_rule(ruled), 400)
        # the darkest candidate wins: a short table line beside a full rule
        both = ruled.copy()
        both[600:700, 600:602] = 0
        self.assertEqual(find_vertical_rule(both), 400)
        barred = page.copy()
        barred[150:1250, 380:440] = 0                                 # 6% of the width: a bar, not a hairline
        self.assertIsNone(find_vertical_rule(barred))
        crowded = ruled.copy()
        crowded[150:1250, 404:406] = 0                                # ink right beside it: not the white a gutter has
        self.assertIsNone(find_vertical_rule(crowded))
        # a rule beside one band only, its rows reported; text elsewhere on the
        # page does not hide it (the Santhya's arth runs full width above it)
        from lib.ocr_zones import vertical_rule
        partial = page.copy()
        partial[700:1100, 399:402] = 0
        for y in range(200, 640, 40):
            partial[y:y + 24, 120:880] = 0                            # full-width "lines" above the rule
        got = vertical_rule(partial)
        self.assertEqual((got["x"], got["y0"], got["y1"]), (400, 700, 1100))
        # a rule set tight against the text it divides: the ink beside it
        # over the rule's own rows decides, not the text above
        tight = partial.copy()
        tight[700:1100, 120:392] = 0                                  # a black block 7 px from the rule
        self.assertEqual(vertical_rule(tight)["x"], 400)
        tight[700:1100, 120:398] = 0                                  # 1 px: no gutter
        self.assertIsNone(vertical_rule(tight))

    def test_the_rule_read_as_a_bar_belongs_to_neither_half(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")
        words = [{"text": t, "bbox": [x, 960, x + (8 if t == "|" else 60), 1039]} for t, x in
                 (("ਗਾਵੈ", 272), ("ਕੋ", 379), ("ਤਾਣੁ", 439), ("|", 738), ("ਕੋਈ", 780), ("ਜਿਸ", 859))]
        line = {"zone": "body", "bbox": [272, 960, 919, 1039], "text": " ".join(w["text"] for w in words), "words": words}
        out = m.split_at_gutter([line], 742)
        self.assertEqual([ln["text"] for ln in out], ["ਗਾਵੈ ਕੋ ਤਾਣੁ", "ਕੋਈ ਜਿਸ"])
        self.assertEqual(out[1]["bbox"][0], 780)                            # the piece starts at its first word, not at the bar

    def test_a_line_is_split_at_the_gutter_only_where_the_rule_runs(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")
        def across(y):
            return {"n": 1, "zone": "body", "bbox": [100, y, 2300, y + 40], "text": "ਖੱਬੇ ਸੱਜੇ",
                    "words": [{"bbox": [100, y, 900, y + 40], "text": "ਖੱਬੇ"}, {"bbox": [1300, y, 2300, y + 40], "text": "ਸੱਜੇ"}]}
        out = m.split_at_gutter([across(300), across(1500)], 1200, span=(1000, 2000))
        self.assertEqual([ln["text"] for ln in out], ["ਖੱਬੇ ਸੱਜੇ", "ਖੱਬੇ", "ਸੱਜੇ"])
        self.assertEqual([bool(ln.get("split")) for ln in out], [False, True, True])
        self.assertEqual(len(m.split_at_gutter([across(300), across(1500)], 1200)), 4)

    def test_a_rule_is_a_gutter_even_where_the_engine_read_across_it(self):
        import numpy as np
        from lib.ocr_zones import page_columns_with_source
        blank = np.full((3300, 2560), 255, dtype=np.uint8)
        # every line crosses x=1200: Tesseract merged the columns, as it does
        across = [{"bbox": [100, 100 + 50 * i, 2300, 140 + 50 * i], "text": "ਸ਼ਬਦ ਸ਼ਬਦ ਸ਼ਬਦ", "zone": "body",
                   "words": [{"bbox": [100, 100 + 50 * i, 300, 140 + 50 * i], "text": "ਸ਼ਬਦ"},
                             {"bbox": [1300, 100 + 50 * i, 1500, 140 + 50 * i], "text": "ਸ਼ਬਦ"},
                             {"bbox": [1900, 100 + 50 * i, 2100, 140 + 50 * i], "text": "ਸ਼ਬਦ"}]} for i in range(20)]
        words = [w for ln in across for w in ln["words"]]
        # the crossing test refuses the gutter the ink and the word starts suggest...
        self.assertEqual(page_columns_with_source(words, 2560, blank, across), ([], None))
        # ...and a rule overrules it: crossing lines are what split_at_gutter cuts
        self.assertEqual(page_columns_with_source(words, 2560, blank, across, rule_x=1200),
                         ([(0, 1200), (1200, 2560)], "rule"))
        # a rule with nothing to its left is a border, not a gutter
        self.assertEqual(page_columns_with_source(words, 2560, blank, across, rule_x=50), ([], None))
        self.assertEqual(page_columns(words, 2560, blank, across, rule_x=1200), [(0, 1200), (1200, 2560)])


CORPUS = [
    (1, 1, 22, "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ ॥"),
    (2, 1, 22, "ਪ੍ਰਭ ਕਿਰਪਾ ਤੇ ਪ੍ਰਾਣੀ ਛੁਟੈ ॥"),
    (3, 2, 23, "ਭੈ ਕਾਹੂ ਕਉ ਦੇਤ ਨਹਿ ਨਹਿ ਭੈ ਮਾਨਤ ਆਨ ॥"),
    (4, 2, 23, "ਕਹੁ ਨਾਨਕ ਸੁਨਿ ਰੇ ਮਨਾ ਗਿਆਨੀ ਤਾਹਿ ਬਖਾਨਿ ॥੧੬॥"),
    (5, 3, 800, "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ ॥"),              # SGGS repeats lines verbatim
    (6, 4, 900, "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੇ ਸਦਾ ॥"),          # a near-neighbour with different text
    (7, 5, 22, "ਜਿਨ ਨਿਰਭਉ ਜਿਨ ਹਰਿ ਨਿਰਭਉ ਧਿਆਇਆ ਜੀ ਤਿਨ ਕਾ ਭਉ ਸਭੁ ਗਵਾਸੀ ॥"),
]


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.idx = CorpusIndex.from_rows(CORPUS)

    def test_exact_in_window(self):
        m = match_text("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ", self.idx, {"ang_from": 22, "ang_to": 22})
        self.assertEqual((m["line_id"], m["method"]), (1, "ang-window"))

    def test_one_confusion_still_matches(self):
        m = match_text("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਬਉ ਮਿਟੈ ॥", self.idx, {"ang_from": 22, "ang_to": 22})
        self.assertEqual(m["line_id"], 1)
        self.assertLess(m["score"], 1.0)

    def test_repeated_identical_text_is_not_ambiguous(self):
        m = match_text("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ", self.idx, {"ang_from": 800, "ang_to": 800})
        self.assertEqual(m["line_id"], 5)

    def test_close_runner_up_with_different_text_is_refused(self):
        # inside a window holding both 5 and 6, a reading half-way between them matches nothing
        idx = CorpusIndex.from_rows([(5, 3, 800, "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ ਸਦਾ ॥"),
                                     (6, 4, 800, "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੇ ਸਦਾ ॥")])
        self.assertIsNone(match_text("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟ ਸਦਾ", idx, {"ang_from": 800, "ang_to": 800}))

    def test_quote_with_its_citation_matches_on_a_span(self):
        m = match_text("ਭੈ ਕਾਹੂ ਕਉ ਦੇਤ ਨਹਿ ਨਹਿ ਭੈ ਮਾਨਤ ਆਨ ॥ (ਸਲੋਕ ਮ:੯) ਸਲੋਕ ਮਹਲਾ ਨੌਵਾਂ ਵਿਚ",
                       self.idx, {"ang_from": 23, "ang_to": 23})
        self.assertEqual(m["line_id"], 3)
        self.assertIsNotNone(m["span"])
        self.assertEqual(m["span"][0], 0)

    def test_short_key_is_never_matched(self):
        self.assertIsNone(match_text("ਭੈ ਕਾਹੂ", self.idx, {"ang_from": 23, "ang_to": 23}))

    def test_global_fallback_is_stricter(self):
        exact = match_text("ਕਹੁ ਨਾਨਕ ਸੁਨਿ ਰੇ ਮਨਾ ਗਿਆਨੀ ਤਾਹਿ ਬਖਾਨਿ", self.idx, None)
        self.assertEqual((exact["line_id"], exact["method"]), (4, "global"))
        noisy = match_text("ਕਹੁ ਨਾਨਕ ਸੁਨਿ ਰੇ ਮਨਾ ਗਿਆਨੀ ਤਾਹ ਬਖਾਨ", self.idx, None)
        windowed = match_text("ਕਹੁ ਨਾਨਕ ਸੁਨਿ ਰੇ ਮਨਾ ਗਿਆਨੀ ਤਾਹ ਬਖਾਨ", self.idx, {"ang_from": 23, "ang_to": 23})
        self.assertIsNotNone(windowed)
        from lib.ocr_match import GLOBAL_ACCEPT
        self.assertTrue(noisy is None or noisy["score"] >= GLOBAL_ACCEPT)

    def test_window_slack_covers_a_page_break(self):
        m = match_text("ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ", self.idx, {"ang_from": 23, "ang_to": 23})
        self.assertEqual(m["line_id"], 1)


class RouteTests(unittest.TestCase):
    def test_reasons(self):
        self.assertIsNone(needs_arbiter({"kind": "gurbani", "agreement": 0.1}))
        self.assertEqual(needs_arbiter({"kind": "commentary", "agreement": 0.5}), "low-agreement")
        self.assertEqual(needs_arbiter({"kind": "commentary", "agreement": 0.95, "oov": 0.5, "words": 6}), "oov")
        self.assertEqual(needs_arbiter({"kind": "gurbani-unmatched"}), "unmatched-bold")
        self.assertIsNone(needs_arbiter({"kind": "commentary", "agreement": 0.95, "oov": 0.0, "conf": 0.9}))

    def test_page_routing(self):
        lines = [{"zone": "body", "kind": "commentary", "agreement": 0.5}] * 3 + \
                [{"zone": "body", "kind": "commentary", "agreement": 0.99}] * 20
        self.assertEqual(page_needs_arbiter(lines)["lines"], 3)
        self.assertIsNone(page_needs_arbiter(lines[3:]))

    def test_budget_cap_and_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = os.path.join(d, "costs.jsonl")
            b = Budget(ledger, cap_usd=0.01)
            self.assertTrue(b.allows(0.0015))
            b.charge("vision", "book", 1, 0.0015)
            self.assertAlmostEqual(b.spent_this_month(), 0.0015)
            self.assertTrue(b.allows(0.008))
            self.assertFalse(b.allows(0.009))
            rows = [json.loads(x) for x in open(ledger, encoding="utf-8")]
            self.assertEqual((rows[0]["engine"], rows[0]["pages"]), ("vision", 1))


class EngineParsingTests(unittest.TestCase):
    def test_tesseract_words_group_into_lines(self):
        d = {"text": ["", "ਨਿਰਭਉ", "ਜਪੈ", "Page", "41"], "conf": [-1, 91.0, 80.0, 95.0, 90.0],
             "block_num": [1, 1, 1, 2, 2], "par_num": [1, 1, 1, 1, 1], "line_num": [0, 1, 1, 1, 1],
             "left": [0, 100, 300, 1000, 1200], "top": [0, 200, 202, 3200, 3200],
             "width": [0, 150, 120, 100, 50], "height": [0, 40, 38, 30, 30]}
        lines = TesseractEngine.lines_from_data(d, "en")
        self.assertEqual([ln["text"] for ln in lines], ["ਨਿਰਭਉ ਜਪੈ", "Page 41"])
        self.assertEqual(lines[0]["bbox"], [100, 200, 420, 240])
        self.assertAlmostEqual(lines[0]["conf"], 0.855)
        self.assertEqual([ln["n"] for ln in lines], [1, 2])

    def test_a_region_is_read_in_page_coordinates(self):
        import numpy as np
        from lib.ocr_engines import tess_lang_of
        d = {"text": ["ਨਿਰਭਉ", "ਜਪੈ"], "conf": [91.0, 80.0], "block_num": [1, 1], "par_num": [1, 1],
             "line_num": [1, 1], "left": [10, 210], "top": [5, 7], "width": [150, 120], "height": [40, 38]}
        seen = {}

        class FakePT:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(im, lang, config, output_type):
                seen.update({"size": im.size, "lang": lang, "config": config})
                return d
        eng = TesseractEngine.__new__(TesseractEngine)
        eng.pt, eng.psm, eng.oem, eng.tess_lang = FakePT, 3, 1, "pan"
        page = np.full((1400, 1000), 255, dtype=np.uint8)
        lines = eng.recognise_region(page, [100, 200, 600, 260], "pa")
        self.assertEqual(seen, {"size": (500, 60), "lang": "pan", "config": "--oem 1 --psm 7"})
        self.assertEqual(lines[0]["bbox"], [110, 205, 430, 245])
        self.assertEqual(lines[0]["words"][1]["bbox"], [310, 207, 430, 245])
        self.assertEqual(lines[0]["psm"], 7)
        self.assertEqual([tess_lang_of(k) for k in ("tesseract", "tesseract-pan", "tesseract-gurmukhi", "tesseract-hin")],
                         [None, "pan", "script/Gurmukhi", "hin"])

    def test_vision_lines_break_on_eol(self):
        def sym(t, brk=0):
            return SimpleNamespace(text=t, confidence=0.9,
                                   property=SimpleNamespace(detected_break=SimpleNamespace(type_=brk)))

        def word(t, x, y, brk=0):
            syms = [sym(c) for c in t[:-1]] + [sym(t[-1], brk)]
            verts = [SimpleNamespace(x=x, y=y), SimpleNamespace(x=x + 50, y=y), SimpleNamespace(x=x + 50, y=y + 30),
                     SimpleNamespace(x=x, y=y + 30)]
            return SimpleNamespace(symbols=syms, bounding_box=SimpleNamespace(vertices=verts), confidence=0.9)

        para = SimpleNamespace(words=[word("ab", 0, 0, 1), word("cd", 60, 0, 5), word("ef", 0, 40, 5)])
        resp = SimpleNamespace(full_text_annotation=SimpleNamespace(
            pages=[SimpleNamespace(blocks=[SimpleNamespace(paragraphs=[para])])]))
        engine = VisionEngine.__new__(VisionEngine)
        lines = engine._lines(resp)
        self.assertEqual([ln["text"] for ln in lines], ["ab cd", "ef"])
        self.assertEqual(lines[0]["bbox"], [0, 0, 110, 30])

    def test_split_block_and_numbering(self):
        ls = split_block("one\ntwo\n", [0, 0, 100, 40], 0.5, "footnote")
        self.assertEqual([(l["text"], l["bbox"][1], l["bbox"][3]) for l in ls], [("one", 0, 20), ("two", 20, 40)])
        ordered = number([{"bbox": [0, 50, 1, 60], "text": "b"}, {"bbox": [0, 10, 1, 20], "text": "a"}])
        self.assertEqual([l["text"] for l in ordered], ["a", "b"])

    def test_smart_resize_multiple_of_28(self):
        w, h = smart_resize(2560, 3300)
        self.assertEqual((w % 28, h % 28), (0, 0))
        self.assertLessEqual(w * h, 11289600)


class PagesTests(unittest.TestCase):
    def test_parse_pages(self):
        self.assertEqual(parse_pages("1-3,7", 10), [0, 1, 2, 6])
        self.assertEqual(parse_pages(None, 3), [0, 1, 2])
        self.assertEqual(parse_pages("9-20", 10), [8, 9])

    def test_render_probe_and_deskew_on_a_drawn_pdf(self):
        try:
            import pymupdf
            import numpy as np
            from lib.ocr_pages import deskew, render, crop_border
            from lib.ocr_probe import probe_pdf
        except ImportError:
            self.skipTest("pymupdf/cv2 not installed")
        with tempfile.TemporaryDirectory() as d:
            text_pdf = os.path.join(d, "t.pdf")
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 100), "\n".join(["The quick brown fox jumps over the lazy dog."] * 10), fontsize=11)
            doc.save(text_pdf)
            self.assertEqual(probe_pdf(text_pdf)["shape"], "text")
            img = render(page, dpi=100)
            self.assertEqual(img.dtype, np.uint8)
            rot, angle = deskew(img)
            self.assertLess(abs(angle), 0.3)
            framed = np.full((400, 300), 255, np.uint8)
            framed[:, :20] = 0
            framed[:15, :] = 0
            cut, box = crop_border(framed)
            self.assertEqual((box[0], box[2]), (15, 20))
            doc.close()


class ManifestTests(unittest.TestCase):
    def test_manifest_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.pdf", "b.pdf"):
                open(os.path.join(d, name), "wb").close()
            json.dump({"author": "X", "language": "pa", "reader": "ocr", "licence": "copyright",
                       "works": [{"file": "a.pdf", "work": "w", "part": 2, "title": "W, Part 2", "book": "w-2"},
                                 {"file": "b.pdf", "title": "B", "licence": "public-domain", "reader": "legacy-font"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            m = load_manifest(d)
            self.assertEqual([os.path.basename(p) for p in list_sources(d, m)], ["a.pdf", "b.pdf"])
            a = parse_source(os.path.join(d, "a.pdf"), m)
            self.assertEqual((a["work"], a["part"], a["book"], a["quote_policy"], a["language"]),
                             ("w", 2, "w-2", "summarise", "pa"))
            self.assertEqual(a["work_title"], "W")
            b = parse_source(os.path.join(d, "b.pdf"), m)
            self.assertEqual((b["work"], b["quote_policy"], b["reader"], b["author"]), ("b", "verbatim", "legacy-font", "X"))
            # what a work is, how far it goes and how its pages are read: the defaults
            self.assertEqual((a["kind"], a["layout"], a["translate"], a["angs"], a["header_pattern"]),
                             ("essay", "auto", True, None, None))

    def test_kind_translate_layout_and_angs_come_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.pdf", "b.pdf", "c.pdf"):
                open(os.path.join(d, name), "wb").close()
            json.dump({"author": "X", "language": "pa", "reader": "ocr", "kind": "commentary", "translate": False,
                       "works": [{"file": "a.pdf", "title": "A", "angs": [1, 53], "layout": "paired-columns",
                                  "header_pattern": "(?P<section>.+?) \\((?P<book_page>\\d+)\\)"},
                                 {"file": "b.pdf", "title": "B", "kind": "word-meaning", "translate": True},
                                 {"file": "c.pdf", "title": "C", "language": "en"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            m = load_manifest(d)
            a, b, c = (parse_source(os.path.join(d, n), m) for n in ("a.pdf", "b.pdf", "c.pdf"))
            self.assertEqual((a["kind"], a["translate"], a["layout"], a["angs"]), ("commentary", False, "paired-columns", [1, 53]))
            self.assertIn("book_page", a["header_pattern"])
            self.assertEqual((b["kind"], b["translate"], b["layout"]), ("word-meaning", True, "auto"))
            self.assertEqual((c["language"], c["translate"]), ("en", False))            # English: nothing to translate
            json.dump({"works": [{"file": "a.pdf", "title": "A", "kind": "novel"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            with self.assertRaises(ValueError):
                parse_source(os.path.join(d, "a.pdf"), load_manifest(d))

    def test_a_book_may_join_another_corpus_and_carry_an_english_title(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.pdf"), "wb").close()
            json.dump({"language": "pa", "reader": "ocr", "corpus": "barusahib",
                       "works": [{"file": "a.pdf", "title": "ਜੀਵਨ ਕਥਾ", "title_en": "The Life"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            a = parse_source(os.path.join(d, "a.pdf"), load_manifest(d))
            self.assertEqual((a["corpus"], a["title_en"], a["work_title"]), ("barusahib", "The Life", "ਜੀਵਨ ਕਥਾ"))

    def test_a_work_in_several_volumes_covers_all_their_angs(self):
        from importlib import import_module
        work_angs = import_module("12_ingest_writings").work_angs
        self.assertEqual(work_angs([{"angs": [1, 53]}, {"angs": [52, 150]}, {"angs": None}, {"angs": [489, 607]}]),
                         [1, 607])
        self.assertIsNone(work_angs([{}, {"angs": None}]))

    def test_no_manifest_is_todays_behaviour(self):
        meta = parse_source("C:/x/1492003384127-Dhooja-Bhau-Part-2.pdf", None)
        self.assertEqual((meta["essay"], meta["part"], meta["work"], meta["reader"]), (127, 2, "dhooja-bhau", "pdf-text"))
        self.assertEqual((meta["kind"], meta["layout"], meta["translate"]), ("essay", "auto", False))
        
    def test_a_notation_book_carries_its_kind_and_style(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.pdf", "b.pdf", "c.pdf"):
                open(os.path.join(d, name), "wb").close()
            json.dump({"author": "X", "language": "pa", "reader": "ocr", "kind": "notation",
                       "style": {"table": "bars", "shabad_position": "after"},
                       "works": [{"file": "a.pdf", "title": "A", "book": "a"},
                                 {"file": "b.pdf", "title": "B", "book": "b", "style": {"table": "ruled", "labels": True}},
                                 {"file": "c.pdf", "title": "C", "book": "c", "kind": "essay"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            m = load_manifest(d)
            a, b, c = (parse_source(os.path.join(d, n), m) for n in ("a.pdf", "b.pdf", "c.pdf"))
            self.assertEqual(a["kind"], "notation")
            # the folder's style under the defaults; a work's keys over the folder's
            self.assertEqual((a["style"]["table"], a["style"]["shabad_position"], a["style"]["swar_row"]), ("bars", "after", "above"))
            self.assertEqual((b["style"]["table"], b["style"]["labels"], b["style"]["shabad_position"]), ("ruled", True, "after"))
            # a writings work in a notation folder has no style at all
            self.assertEqual((c["kind"], c.get("style")), ("essay", None))
            json.dump({"author": "X", "kind": "notation", "style": {"table": "wavy"},
                       "works": [{"file": "a.pdf", "title": "A"}]}, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            with self.assertRaises(ValueError):
                parse_source(os.path.join(d, "a.pdf"), load_manifest(d))


class DriverTests(unittest.TestCase):
    """27_ingest_book.py plans a run from the manifest without running anything."""

    def _plan(self, manifest: dict, **kw):
        from importlib import import_module
        drv = import_module("27_ingest_book")
        with tempfile.TemporaryDirectory() as d:
            for w in manifest["works"]:
                open(os.path.join(d, w["file"]), "wb").close()
            json.dump(manifest, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            metas = drv.books_in(d, kw.pop("book", None))
            kw.setdefault("ocr_dir", os.path.join(d, "ocr"))
            kw.setdefault("corpus_db", os.path.join(d, "none.sqlite"))
            return drv.plan(d, metas, **kw)

    def test_a_punjabi_book_runs_both_tesseract_variants_and_is_translated(self):
        steps = self._plan({"author": "A", "language": "pa", "reader": "ocr", "licence": "public-domain",
                            "works": [{"file": "s.pdf", "work": "santhya", "book": "santhya-1", "title": "S"}]})
        names = [s for s, _ in steps]
        self.assertEqual(names, ["pages", "ocr", "ocr", "eval", "merge", "ingest", "cite", "embed", "db",
                                 "translate", "embed-en", "db-en"])
        ocr = [a for s, a in steps if s == "ocr"]
        self.assertIn("pan", ocr[0])
        self.assertIn("script/Gurmukhi", ocr[1])
        self.assertTrue(all("--lang" in a and a[a.index("--lang") + 1] == "pa" for a in ocr))
        # without ground truth the eval is a message, not a command; likewise the citations without a corpus
        self.assertTrue(isinstance(dict(steps)["eval"], str) and "same weight" in dict(steps)["eval"])
        self.assertTrue(isinstance(dict(steps)["cite"], str))
        tr = [a for s, a in steps if s == "translate"][0]
        self.assertEqual(tr[tr.index("--src-lang") + 1], "pa")
        self.assertIn("--translations", [a for s, a in steps if s == "embed-en"][0])
        # coverage is on by default since it was measured; --out-name names a merge only when asked
        merge = dict(steps)["merge"]
        self.assertIn("--coverage", merge)
        self.assertNotIn("--out-name", merge)

    def test_coverage_comes_from_the_manifest_or_the_flag(self):
        book = {"author": "A", "language": "pa", "reader": "ocr", "licence": "public-domain",
                "works": [{"file": "s.pdf", "work": "santhya", "book": "santhya-1", "title": "S", "coverage": True}]}
        self.assertIn("--coverage", dict(self._plan(book))["merge"])
        self.assertNotIn("--coverage", dict(self._plan(book, coverage=False))["merge"])
        del book["works"][0]["coverage"]
        self.assertIn("--coverage", dict(self._plan(book))["merge"])            # the measured default: on
        book["works"][0]["coverage"] = False
        self.assertNotIn("--coverage", dict(self._plan(book))["merge"])         # a book may say no
        del book["works"][0]["coverage"]
        merge = dict(self._plan(book, coverage=True, out_name="merged-cov"))["merge"]
        self.assertIn("--coverage", merge)
        self.assertEqual(merge[merge.index("--out-name") + 1], "merged-cov")

    def test_correct_agreed_comes_from_the_manifest_or_the_flag(self):
        book = {"language": "pa", "reader": "ocr",
                "works": [{"file": "s.pdf", "work": "jiwan", "book": "jiwan-1", "title": "J"}]}
        self.assertNotIn("--correct-agreed", dict(self._plan(book))["merge"])       # off unless asked
        book["works"][0]["correct_agreed"] = True
        self.assertIn("--correct-agreed", dict(self._plan(book))["merge"])
        self.assertNotIn("--correct-agreed", dict(self._plan(book, correct_agreed=False))["merge"])

    def test_the_translation_takes_the_books_glossary_prompt_and_arbiter(self):
        book = {"language": "pa", "reader": "ocr", "glossary": "glossary.json", "prompt": "rules", "arbiter": "self",
                "works": [{"file": "s.pdf", "work": "jiwan", "book": "jiwan-1", "title": "J", "translate": True}]}
        argv = dict(self._plan(book))["translate"]
        self.assertEqual(argv[argv.index("--prompt") + 1], "rules")
        self.assertEqual(argv[argv.index("--arbiter") + 1], "self")
        self.assertTrue(argv[argv.index("--glossary") + 1].endswith("glossary.json"))
        del book["glossary"], book["prompt"], book["arbiter"]
        argv = dict(self._plan(book))["translate"]
        self.assertNotIn("--glossary", argv)
        self.assertNotIn("--prompt", argv)

    def test_a_book_that_joins_another_corpus_names_its_folder_at_every_step(self):
        from lib.paths import ROOT
        book = {"language": "pa", "reader": "ocr", "corpus": "barusahib", "translate": True,
                "works": [{"file": "a.pdf", "work": "jiwan", "book": "jiwan-1", "part": 1, "title": "J"},
                          {"file": "b.pdf", "work": "jiwan", "book": "jiwan-2", "part": 2, "title": "J"}]}
        folder = os.path.join(ROOT, "data", "barusahib")
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "corpus.sqlite")
            with open(db, "wb") as fh:
                fh.write(b"a scripture database")              # has_rows asks only that it is there
            steps = self._plan(book, corpus_db=db)
        argv = {s: a for s, a in reversed(steps)}                  # the first of each step
        ingest = argv["ingest"]
        self.assertEqual(ingest[ingest.index("--out") + 1], folder)
        # one work in two volumes: its citations resolved once, and only its own
        cites = [a for s, a in steps if s == "cite"]
        self.assertEqual(len(cites), 1)
        self.assertEqual((cites[0][cites[0].index("--src") + 1], cites[0][cites[0].index("--work") + 1]),
                         (folder, "jiwan"))
        self.assertEqual(argv["embed"][argv["embed"].index("--corpus") + 1], "barusahib-pa")
        self.assertEqual(argv["db"][argv["db"].index("--units") + 1], os.path.join(folder, "units-pa.jsonl"))
        self.assertEqual(argv["translate"][argv["translate"].index("--src") + 1], folder)
        self.assertEqual(argv["embed-en"][argv["embed-en"].index("--corpus") + 1], "barusahib-en")
        self.assertEqual(argv["db-en"][argv["db-en"].index("--units") + 1], os.path.join(folder, "units.jsonl"))
        # without the key, today's folder and corpora
        del book["corpus"]
        argv = dict(self._plan(book))
        self.assertNotIn("--out", argv["ingest"])
        self.assertNotIn("--corpus", argv["embed-en"])

    def test_a_notation_book_runs_the_notation_steps_and_none_of_the_prose_ones(self):
        manifest = {"author": "Prin. Dyal Singh", "language": "pa", "reader": "ocr", "kind": "notation",
                    "style": {"table": "bars"},
                    "works": [{"file": "gss1.pdf", "work": "gurmat-sangeet-sagar", "part": 1,
                               "book": "gurmat-sangeet-sagar-1", "title": "Gurmat Sangeet Sagar"}]}
        steps = self._plan(manifest)
        names = [s for s, _ in steps]
        self.assertEqual(names, ["pages", "ocr", "ocr", "ocr", "merge", "notation", "notation-eval", "notation-db"])
        ocr = [a for s, a in steps if s == "ocr"]
        self.assertIn("pan", ocr[0])
        self.assertEqual(ocr[1][ocr[1].index("--psm") + 1], "4")          # the psm 4 pass keeps sparse grid rows
        self.assertIn("script/Gurmukhi", ocr[2])
        self.assertTrue(isinstance(dict(steps)["notation-eval"], str) and "--gt" in dict(steps)["notation-eval"])
        self.assertTrue(dict(steps)["notation-db"][0].endswith("32_build_notations_db.py"))
        # --gt: the mid-book window (the PDF is empty here, so no window) and a STOP after the review page
        gt = self._plan(manifest, gt=True)
        self.assertEqual([s for s, _ in gt], ["pages", "ocr", "ocr", "ocr", "merge", "notation", "notation-gt", "notation-gt"])
        self.assertIn("--mid", [a for s, a in gt if s == "notation-gt"][0])
        self.assertTrue(isinstance(gt[-1][1], str) and gt[-1][1].startswith("STOP"))
        # a writings book (an essay) in the same folder still takes the prose route
        mixed = dict(manifest, kind="essay")
        mixed["works"] = manifest["works"] + [{"file": "p.pdf", "work": "prose", "book": "prose-1", "title": "P", "kind": "essay"}]
        mixed["works"][0] = dict(mixed["works"][0], kind="notation")
        both = self._plan(mixed)
        self.assertIn("ingest", [s for s, _ in both])
        self.assertIn("notation", [s for s, _ in both])

    def test_the_review_window_sits_at_the_middle_of_the_book(self):
        from lib.notation import mid_window
        self.assertEqual(mid_window(373, 7), (168, 174))
        self.assertEqual(mid_window(10, 5), (4, 8))
        self.assertEqual(mid_window(3, 5), (1, 3))

    def test_a_window_is_rendered_one_page_wider_on_each_side_than_it_is_read(self):
        from importlib import import_module
        drv = import_module("27_ingest_book")
        self.assertEqual(drv.widen_pages("165-176,200"), "164-177,199-201")
        self.assertEqual(drv.widen_pages("1-4"), "1-5")
        steps = self._plan({"author": "A", "language": "pa", "reader": "ocr", "kind": "notation",
                            "works": [{"file": "n.pdf", "work": "n", "book": "n", "title": "N"}]}, pages="10-13")
        argv = dict(steps)
        self.assertEqual(argv["pages"][-1], "1-21,9-16")    # rendered: the front pages, then a page before, the lookahead and a page after
        self.assertEqual(argv["merge"][-1], "1-20,10-15")   # read: the front pages (the index), the window and two pages past it;
        # the last twenty pages too when the book is long enough to have a back index (an empty pdf has no pages here)

    def test_an_english_book_is_not_translated_and_a_hindi_one_is(self):
        en = self._plan({"author": "A", "language": "en", "reader": "ocr",
                         "works": [{"file": "t.pdf", "work": "ten", "book": "ten", "title": "T"}]})
        self.assertEqual([s for s, _ in en], ["pages", "ocr", "eval", "merge", "ingest", "cite", "embed", "db"])
        hi = self._plan({"author": "A", "language": "hi", "reader": "ocr",
                         "works": [{"file": "h.pdf", "work": "h", "book": "h", "title": "H"}]},
                        gpu_engines=["dotsocr"])
        ocr = [a for s, a in hi if s == "ocr"]
        self.assertEqual(len(ocr), 2)
        self.assertIn("dotsocr", ocr[1])
        self.assertIn("translate", [s for s, _ in hi])

    def test_ground_truth_mode_stops_before_the_merge(self):
        steps = self._plan({"author": "A", "language": "pa", "reader": "ocr",
                            "works": [{"file": "s.pdf", "work": "s", "book": "s", "title": "S"}]}, gt=True)
        self.assertEqual([s for s, _ in steps], ["pages", "ocr", "ocr", "gt", "gt"])
        self.assertTrue(str(steps[-1][1]).startswith("STOP"))

    def test_no_translate_leaves_a_message(self):
        steps = self._plan({"author": "A", "language": "pa", "reader": "ocr",
                            "works": [{"file": "s.pdf", "work": "s", "book": "s", "title": "S"}]}, translate=False)
        self.assertTrue(isinstance(dict(steps)["translate"], str))
        self.assertIn("--no-translate", dict(steps)["translate"])
        self.assertNotIn("embed-en", [s for s, _ in steps])

    def test_each_work_says_whether_it_is_translated(self):
        # a commentary kept in Punjabi for display beside the verse, and a
        # translation meant for English search, in one folder: only the second
        # goes through 26, and the English corpus is still built
        manifest = {"author": "A", "language": "pa", "reader": "ocr",
                    "works": [{"file": "s.pdf", "work": "santhya", "book": "s-1", "title": "S", "translate": False},
                              {"file": "t.pdf", "work": "teeka", "book": "t-1", "title": "T", "kind": "translation"}]}
        steps = self._plan(manifest)
        tr = [a for s, a in steps if s == "translate"]
        cmds = [a for a in tr if not isinstance(a, str)]
        self.assertEqual([a[a.index("--work") + 1] for a in cmds], ["teeka"])
        self.assertTrue(any(isinstance(a, str) and "santhya" in a for a in tr))
        self.assertIn("embed-en", [s for s, _ in steps])
        # every work off: the English side is skipped, and the message says which key turns it on
        manifest["works"][1]["translate"] = False
        steps = self._plan(manifest)
        msg = dict(steps)["translate"]
        self.assertTrue(isinstance(msg, str) and "translate: false" in msg and "--translate" in msg)
        self.assertNotIn("embed-en", [s for s, _ in steps])
        # --translate overrides the manifest for all of them
        steps = self._plan(manifest, translate=True)
        self.assertEqual(sorted(a[a.index("--work") + 1] for s, a in steps if s == "translate"),
                         ["santhya", "teeka"])


class BenchTests(unittest.TestCase):
    """29_bench_books.py: the sample, the scorecard and its table."""

    def test_the_sample_is_the_first_pages_and_a_spread_over_the_rest(self):
        from importlib import import_module
        b = import_module("29_bench_books")
        self.assertEqual(b.sample_pages(15), "1-15")                       # a short work: all of it
        spec = b.sample_pages(530)
        self.assertTrue(spec.startswith("1-12,"))
        picks = [int(x) for x in spec.split(",")[1:]]
        self.assertEqual(len(picks), 8)
        self.assertTrue(all(12 < p <= 530 for p in picks) and picks == sorted(picks))
        # spread: no two picks closer than half the even step
        step = (530 - 12) / 8
        self.assertTrue(all(b - a >= step / 2 for a, b in zip(picks, picks[1:])))

    def _book(self, d: str) -> str:
        book_dir = os.path.join(d, "b")
        os.makedirs(os.path.join(book_dir, "merged"))
        with open(os.path.join(book_dir, "pages.json"), "w", encoding="utf-8") as fh:
            json.dump({"book": "b", "language": "pa", "probe": {"shape": "image"}, "pages": [{"page": 1}]}, fh)
        h = 40
        line = lambda n, y, text, kind, **kw: {"n": n, "zone": "body", "bbox": [100, y, 1000, y + h], "text": text,
                                                "kind": kind, "col": 0, **kw}
        lines = [line(1, 100, "ਸਿਰਲੇਖ", "heading", bold=True),
                 line(2, 200, "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥", "gurbani", bold=True,
                      matches=[{"line_id": 3, "shabad_id": 1, "ang": 1, "score": 0.98, "source": "G"}]),
                 line(3, 260, "ਇਸ ਤੁਕ ਦਾ ਅਰਥ ਇਹ ਹੈ ਕਿ ਸੋਚਣ ਨਾਲ ਕੁਝ ਨਹੀਂ ਬਣਦਾ।", "commentary", route="oov"),
                 line(4, 310, "ਹੋਰ ਵਿਆਖਿਆ ਇੱਥੇ ਹੈ।", "commentary", route="oov", recovered={"region": 0}),
                 {"n": 5, "zone": "header", "bbox": [100, 20, 600, 50], "text": "ਅੰਗ ੧-੨", "dropped": True}]
        meta = {"page": 1, "page_w": 1200, "page_h": 1600, "body_h": h, "columns": None, "columns_source": None,
                "hints": {"ang_from": 1, "ang_to": 2, "book_page": 7, "section": None, "ang_source": "header"},
                "coverage": {"on": True, "regions": [{"why": "recovered"}], "recovered": 1, "rejected": {"short": 2},
                             "uncovered_after": 0}}
        with open(os.path.join(book_dir, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": meta}) + "\n")
            for ln in lines:
                fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
        return book_dir

    def test_a_scorecard_counts_what_the_merge_and_the_reader_say(self):
        from importlib import import_module
        b = import_module("29_bench_books")
        with tempfile.TemporaryDirectory() as d:
            row = b.scorecard(self._book(d), {"book": "b", "language": "pa", "kind": "commentary", "angs": [1, 53]},
                              corpus=os.path.join(d, "none.sqlite"))
        self.assertEqual(row["pages"], 1)
        self.assertEqual(row["columns"], {"none": 1})
        self.assertEqual((row["lines"]["gurbani"], row["lines"]["heading"], row["lines"]["commentary"]), (1, 1, 2))
        self.assertEqual((row["matched"], row["recovered"]), (1, 1))
        self.assertEqual(row["routes"], [("oov", 2)])
        self.assertEqual(row["coverage"]["recovered"], 1)
        self.assertEqual(row["layout"], {"single": 1})
        self.assertEqual(row["ang_source"], {"header": 1})
        self.assertEqual(row["angs"], {"found": [1, 2], "manifest": [1, 53]})
        # a commentary: the prose after the verse explains it
        self.assertEqual(row["paragraphs"]["quotes"], 1)
        self.assertGreaterEqual(row["paragraphs"]["explains"], 1)
        self.assertNotIn("accuracy", row)                                  # no ground truth: no number invented
        text = b.markdown([row, {"book": "x", "reader": "pdf-text", "error": "not a scanned book"}])
        self.assertIn("| b | ocr | 1 |", text)
        self.assertIn("no gt", text)
        self.assertIn("| x | pdf-text | - |", text)


class VoterTests(unittest.TestCase):
    """22_ocr_merge.py: which engines vote, and with what weight."""

    def test_an_engine_measured_below_the_bar_does_not_vote_but_the_pivot_always_does(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")
        acc = {"tesseract-pan": 0.958, "surya": 0.921, "dotsocr": 0.883, "indicocr": 0.831}
        keep, drop = m.choose_voters(["dotsocr", "indicocr", "surya", "tesseract-pan", "pdftext"], acc, "tesseract-pan")
        self.assertEqual(keep, ["surya", "tesseract-pan", "pdftext"])        # pdftext unmeasured: keeps voting
        self.assertEqual(drop, ["dotsocr", "indicocr"])
        keep, drop = m.choose_voters(["indicocr", "surya"], acc, "indicocr")
        self.assertEqual((keep, drop), (["indicocr", "surya"], []))
        # a Tesseract variant beside a Tesseract pivot votes below the bar (Santhya vol. 4)
        keep, drop = m.choose_voters(["tesseract-gurmukhi", "tesseract-pan", "dotsocr"],
                                     {"tesseract-pan": 0.786, "tesseract-gurmukhi": 0.783, "dotsocr": 0.80},
                                     "tesseract-pan")
        self.assertEqual((keep, drop), (["tesseract-gurmukhi", "tesseract-pan"], ["dotsocr"]))
        w = m.auto_weights(acc, ["surya"])
        self.assertEqual(w["surya"], round(0.921 ** 4, 3))
        self.assertEqual(m.auto_weights({}, ["x"])["x"], 0.5)


class GlossaryTests(unittest.TestCase):
    """lib/mt_glossary.py: a book's terms, named in the prompt and checked in the English."""

    def glossary(self):
        from lib import mt_glossary as g
        doc = {"rule": "Keep Sikh terms transliterated.",
               "terms": [{"pa": ["ਕੜਾਹ ਪ੍ਰਸ਼ਾਦ", "ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ"], "en": "Karah Prasad", "accept": ["Karah Parshad"]},
                         {"pa": ["ਭੋਗ"], "en": "bhog", "accept": ["concluded"]},
                         {"pa": ["ਸੰਗਤ"], "en": "Sangat", "accept": ["congregation"]},
                         {"pa": ["ਮਹਾਰਾਜ"], "en": "Maharaj", "whole": True},
                         {"pa": ["ਦਮੜਾ"], "en": "damra", "accept": ["coin"]}]}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "glossary.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)
            return g, g.load(path)

    def test_terms_are_found_with_their_inflections_and_checked_in_the_english(self):
        g, G = self.glossary()
        src = "ਜਦ ਆਸਾ ਦੀ ਵਾਰ ਦਾ ਭੋਗ ਪਿਆ, ਸੰਗਤਾਂ ਨੇ ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ ਛਕਿਆ। ੧੦੧ ਦਮੜਾ।"
        self.assertEqual([t["en"] for t in g.present(G, src)], ["Karah Prasad", "bhog", "Sangat", "damra"])
        good = "When the Asa di Var concluded, the congregations partook of Karah Parshad. 101 damras."
        self.assertEqual(g.missing(G, src, good), [])
        bad = "When Asa Ji's time of death was near, the people ate Karah-Pushad. 101 drums."
        self.assertEqual([t["en"] for t in g.missing(G, src, bad)], ["Karah Prasad", "bhog", "Sangat", "damra"])
        self.assertEqual(g.present(G, "ਮਹਾਰਾਜਾ ਰਣਜੀਤ ਸਿੰਘ"), [])                # the king is not the saint's honorific
        from lib import mt_glossary
        idiom = {"terms": [{"pa": ["ਸੇਵਾ"], "en": "Sewa", "accept": [], "whole": False,
                            "unless": [normalise("ਦੀ ਸੇਵਾ ਵਿੱਚ ਬੇਨਤੀ")]}]}
        self.assertEqual(mt_glossary.present(idiom, "ਸੰਤ ਜੀ ਦੀ ਸੇਵਾ ਵਿੱਚ ਬੇਨਤੀ ਕੀਤੀ"), [])   # "humbly submitted to"
        self.assertEqual(len(mt_glossary.present(idiom, "ਲੰਗਰ ਦੀ ਸੇਵਾ ਕੀਤੀ")), 1)
        self.assertEqual(g.prompt_lines(g.present(G, src)[:1], src), ["ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ = Karah Prasad"])

    def test_the_models_glosses_go_and_the_authors_brackets_stay(self):
        g, G = self.glossary()
        src = "ਭੋਗ ਪਿਆ। ਸਾਰੀ ਸੰਗਤ (ਸਭ ਪ੍ਰੇਮੀ) ਆਈ।"
        en = "The *bhog* (offering) was held. The whole Sangat (devotee congregation) (all the devotees) came."
        out, n = g.strip_glosses(en, src, g.present(G, src))
        self.assertEqual(out, "The bhog was held. The whole Sangat (all the devotees) came.")
        self.assertEqual(n, 3)

    def test_the_rules_prompt_names_only_the_terms_the_paragraph_uses(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        g, G = self.glossary()
        p = tw.rules_prompt(G, "ਭੋਗ ਪਿਆ ਤੇ ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ ਵਰਤਿਆ।")
        self.assertIn("Keep Sikh terms transliterated.", p)
        self.assertIn("ਭੋਗ = bhog", p)
        self.assertIn("ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ = Karah Prasad", p)
        self.assertNotIn("Sangat", p)
        self.assertTrue(p.startswith("Translate the text below to English."))
        one = tw.rules_prompt(G, "ਭੋਗ ਪਿਆ ਤੇ ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ ਵਰਤਿਆ।", pairs=False)     # sarvam: one line, no Gurmukhi
        self.assertEqual(one, "Translate the text below to English. Keep these Sikh terms as they are: "
                              "Karah Prasad, bhog. Do not add explanations or brackets.")

    def test_the_arbiter_retranslates_what_was_refused_and_refuses_what_it_cannot_fix(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        g, G = self.glossary()

        class Arbiter:
            name, instruct = "fake", None
            def model_desc(self):
                return "fake"
            def translate_many(self, texts, tgt):
                assert self.instruct is not None                        # always the rules prompt
                return ["When the *bhog* (offering) was held." if "ਭੋਗ" in t else "Something else." for t in texts]

        recs = [{"unit_id": "a", "text": "ਜਦ ਭੋਗ ਪਿਆ।"}, {"unit_id": "b", "text": "ਸੰਗਤ ਆਈ।"}]
        out = tw.arbitrate(Arbiter(), recs, "pa", G, "English")
        self.assertEqual((out[0][1], out[0][2]), ("When the bhog was held.", None))    # stripped, then accepted
        self.assertEqual(out[1][2], "term: Sangat")                                   # refused twice

    def test_a_model_without_an_instruction_refuses_the_rules_prompt(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        g, G = self.glossary()
        with self.assertRaises(SystemExit):
            tw.set_prompt(SimpleNamespace(name="indictrans2"), "rules", G)
        tw.set_prompt(SimpleNamespace(name="indictrans2"), "stock", G)           # stock is always fine

    def test_terms_are_written_into_the_punjabi_for_a_translator_that_takes_no_instruction(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        g, G = self.glossary()
        text, n = g.inline_terms(G, "ਜਦ ਭੋਗ ਪਿਆ, ਸੰਗਤਾਂ ਨੇ ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ ਛਕਿਆ।")
        self.assertEqual((text, n), ("ਜਦ bhog ਪਿਆ, Sangats ਨੇ Karah Prasad ਛਕਿਆ।", 3))
        self.assertEqual(tw.prepare("terms", G, "ਭੋਗ ਪਿਆ"), "bhog ਪਿਆ")
        self.assertEqual(tw.prepare("stock", G, "ਭੋਗ ਪਿਆ"), "ਭੋਗ ਪਿਆ")
        self.assertEqual(tw.arbiter_prompt(SimpleNamespace(), "stock"), "rules")     # an engine that follows instructions

    def test_the_check_names_a_missing_term(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        g, G = self.glossary()
        src = "ਜਦ ਆਸਾ ਦੀ ਵਾਰ ਦਾ ਭੋਗ ਪਿਆ।"
        self.assertEqual(tw.check("When Asa Ji's time of death was near.", src, "pa", G), "term: bhog")
        self.assertIsNone(tw.check("When the Asa di Var concluded.", src, "pa", G))
        self.assertIsNone(tw.check("When Asa Ji's time of death was near.", src, "pa"))   # no glossary: as before


class TranslationTests(unittest.TestCase):
    """26_translate_writings.py's checks and sentence handling, 28_translate_bench.py's sampling and scoring."""


    def test_a_bench_row_names_its_prompt_stripping_and_arbiter(self):
        from importlib import import_module
        tb = import_module("28_translate_bench")
        self.assertEqual(tb.parse_label("sarvam"), ("sarvam", "stock", True, False))
        self.assertEqual(tb.parse_label("sarvam:rules"), ("sarvam", "rules", True, False))
        self.assertEqual(tb.parse_label("sarvam:rules:raw"), ("sarvam", "rules", False, False))
        self.assertEqual(tb.parse_label("sarvam:rules+self"), ("sarvam", "rules", True, True))
        self.assertEqual(tb.parse_label("sarvam:terms+self"), ("sarvam", "terms", True, True))
        with self.assertRaises(SystemExit):
            tb.parse_label("sarvam:fancy")
    def test_a_batch_the_card_cannot_hold_is_retried_one_paragraph_at_a_time(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")

        class Tight:
            calls = []

            def translate_many(self, texts, tgt):
                self.calls.append(len(texts))
                if len(texts) > 1 or "ਲੰਮਾ" in texts[0]:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 50.00 MiB")
                return ["en:" + t for t in texts]
        out = tw.translate_batch(Tight(), ["ਇਕ", "ਦੋ", "ਬਹੁਤ ਲੰਮਾ ਪੈਰਾ"], "English")
        self.assertEqual(out, ["en:ਇਕ", "en:ਦੋ", None])                  # only the one that fails alone is lost
        self.assertEqual(Tight.calls, [3, 1, 1, 1])

        class Broken:
            def translate_many(self, texts, tgt):
                raise RuntimeError("something else")
        with self.assertRaises(RuntimeError):                            # any other error is still an error
            tw.translate_batch(Broken(), ["ਇਕ", "ਦੋ"], "English")

    def test_weights_under_vendor_models_are_used_before_the_hub_is_asked(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        with tempfile.TemporaryDirectory() as d:
            real = tw.ROOT
            tw.ROOT = d
            try:
                self.assertEqual(tw.local_weights("sarvamai/sarvam-translate"), "sarvamai/sarvam-translate")
                os.makedirs(os.path.join(d, "vendor", "models", "sarvam-translate"))
                self.assertEqual(tw.local_weights("sarvamai/sarvam-translate"),
                                 os.path.join(d, "vendor", "models", "sarvam-translate"))
            finally:
                tw.ROOT = real

    def test_sentences_split_on_dandas_and_stops_and_join_back(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        para = "ਪਹਿਲਾ ਵਾਕ ਹੈ। ਦੂਜਾ ਵਾਕ ॥ Third one. Fourth?  ਪੰਜਵਾਂ"
        sents = tw.split_sentences(para)
        self.assertEqual(sents, ["ਪਹਿਲਾ ਵਾਕ ਹੈ।", "ਦੂਜਾ ਵਾਕ ॥", "Third one.", "Fourth?", "ਪੰਜਵਾਂ"])
        self.assertEqual(" ".join(sents), " ".join(para.split()))
        self.assertEqual(tw.split_sentences("  "), [])

    def test_a_reasoning_models_think_block_is_not_a_translation(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        self.assertEqual(tw.strip_think("<think>\nlet me see\n</think>\n\nThe Guru said."), "The Guru said.")
        self.assertEqual(tw.strip_think("plain"), "plain")
        self.assertEqual(tw.strip_think(None), "")

    def test_the_source_script_check_follows_the_language(self):
        from importlib import import_module
        tw = import_module("26_translate_writings")
        self.assertEqual(tw.check("The guru said ਸਤਿ", "x", "pa"), "Punjabi in the English")
        self.assertEqual(tw.check("The guru said सत", "x", "hi"), "Hindi in the English")
        self.assertIsNone(tw.check("The guru said सत", "x", "pa"))
        self.assertEqual(tw.check("", "x", "hi"), "empty")

    def test_bench_sample_is_stratified_and_stable(self):
        from importlib import import_module
        tb = import_module("28_translate_bench")
        recs = [{"unit_id": "u%03d" % i, "text": "x" * (10 + i * 7)} for i in range(60)]
        a = tb.sample_records(recs, 9, seed=1)
        b = tb.sample_records(recs, 9, seed=1)
        self.assertEqual([r["unit_id"] for r in a], [r["unit_id"] for r in b])
        lens = sorted(len(r["text"]) for r in a)
        self.assertEqual(len(a), 9)
        self.assertTrue(lens[0] < 150 and lens[-1] > 300)       # a short one and a long one both drawn
        self.assertEqual(len(tb.sample_records(recs, 0, 0)), 60)

    def test_bench_scoring_rewards_the_closer_translation(self):
        from importlib import import_module
        tb = import_module("28_translate_bench")
        refs = ["The Guru went to the river at dawn.", "He sang the hymn twice."]
        close = ["The Guru went to the river at dawn.", "He sang the hymn two times."]
        far = ["A cat sat.", ""]
        srcs = ["ਗੁਰੂ ਸਵੇਰੇ ਦਰਿਆ ਗਏ।", "ਉਸ ਨੇ ਦੋ ਵਾਰ ਸ਼ਬਦ ਗਾਇਆ।"]
        sc, sf = tb.score(close, refs, srcs), tb.score(far, refs, srcs)
        self.assertGreater(sc["chrf"], sf["chrf"])
        self.assertEqual((sc["scored"], sf["scored"]), (2, 1))
        self.assertEqual(len(sc["per_pair"]), 2)
        self.assertGreater(sc["ratio"], 1.0)


class LegacyFontTests(unittest.TestCase):
    """
    Text typed in a legacy Gurmukhi font, converted without OCR (lib/legacy_font.py).

    The cases are the examples anvaad-js 1.5.1 tests itself with, and each
    expected value is what that library returned for it: the port is held to
    the library, not to a second opinion about Gurmukhi.
    """

    LIBRARY = [
        ("0123456789!?()'‘’:/",
         "੦੧੨੩੪੫੬੭੮੯!?()'‘’:/"),
        ("hy myry gurU dy ipAwry is`K! mYƒ Aw ky iml, mYƒ Aw ky iml [rhwau[",
         "ਹੇ ਮੇਰੇ ਗੁਰੂ ਦੇ ਪਿਆਰੇ ਸਿੱਖ! ਮੈਨੂੰ ਆ ਕੇ ਮਿਲ, ਮੈਨੂੰ ਆ ਕੇ ਮਿਲ ।ਰਹਾਉ।"),
        ("rzw b^So rwizk irhwko rhIm ]1]",
         "ਰਜ਼ਾ ਬਖ਼ਸ਼ੋ ਰਾਜ਼ਿਕ ਰਿਹਾਕੋ ਰਹੀਮ ॥੧॥"),
        ("rjæw bKæsæo rwijæk irhwko rhIm ]1]",
         "ਰਜ਼ਾ ਬਖ਼ਸ਼ੋ ਰਾਜ਼ਿਕ ਰਿਹਾਕੋ ਰਹੀਮ ॥੧॥"),
        ("rwm jpau jIA AYsy AYsy ] DR¨ pRihlwd jipE hir jYsy ]1]",
         "ਰਾਮ ਜਪਉ ਜੀਅ ਐਸੇ ਐਸੇ ॥ ਧ੍ਰੂ ਪ੍ਰਹਿਲਾਦ ਜਪਿਓ ਹਰਿ ਜੈਸੇ ॥੧॥"),
        ("ijgw CqR jVwv kælZI cOr mukqw lwlrI",
         "ਜਿਗਾ ਛਤ੍ਰ ਜੜਾਵ ਕ਼ਲਗ਼ੀ ਚੌਰ ਮੁਕਤਾ ਲਾਲਰੀ"),
        ("ikRpws kæwkæm AqlsI bhu mol cIrn cusq nO ]",
         "ਕ੍ਰਿਪਾਸ ਕ਼ਾਕ਼ਮ ਅਤਲਸੀ ਬਹੁ ਮੋਲ ਚੀਰਨ ਚੁਸਤ ਨੌ ॥"),
        ("su`D ispwh durMq dubwh su swj snwh durjwn dlYNgy ]",
         "ਸੁੱਧ ਸਿਪਾਹ ਦੁਰੰਤ ਦੁਬਾਹ ਸੁ ਸਾਜ ਸਨਾਹ ਦੁਰਜਾਨ ਦਲੈਂਗੇ ॥"),
        ("rwgu gauVI pUrbI₁ mhlw 5",
         "ਰਾਗੁ ਗਉੜੀ ਪੂਰਬੀ₁ ਮਹਲਾ ੫"),
        ("gauVI kbIr jI iqpdy₁₅ ]",
         "ਗਉੜੀ ਕਬੀਰ ਜੀ ਤਿਪਦੇ₁₅ ॥"),
        ("<> siqgur pRswid ]",
         "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥"),
        ("1Eå siqgur pRswid ]",
         "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥"),
        ("¡ siqgur pRswid ]",
         "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥"),
        ("ÅÆ siqgur pRswid ]",
         "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥"),
        ("hUM",
         "ਹੂੰ"),
        ("El@w",
         "ਓਲੑਾ"),
        ("aulwm@y",
         "ਉਲਾਮੑੇ"),
        ("suohwgix",
         "ਸੋੁਹਾਗਣਿ"),
        ("clwey",
         "ਚਲਾਏ"),
        ("AOr",
         "ਔਰ"),
        ("hoveI",
         "ਹੋਵਈ"),
        ("ny Awein guil gulSin &¤ro zyb",
         "ਨੇ ਆੲਨਿ ਗੁਲਿ ਗੁਲਸ਼ਨਿ ਫ਼ੱਰੋ ਜ਼ੇਬ"),
        ("lweIAW",
         "ਲਾਈਆਂ"),
        ("vfw swihbu aUcw Qwau ]",
         "ਵਡਾ ਸਾਹਿਬੁ ਊਚਾ ਥਾਉ ॥"),
        ("pRB kau sd bil jweI jIa ]2]",
         "ਪ੍ਰਭ ਕਉ ਸਦ ਬਲਿ ਜਾਈ ਜੀੳ ॥੨॥"),
        ("AMimRq vylw scu nwau vifAweI vIcwru ]",
         "ਅੰਮ੍ਰਿਤ ਵੇਲਾ ਸਚੁ ਨਾਉ ਵਡਿਆਈ ਵੀਚਾਰੁ ॥"),
        ("sMiDAw pRwq ies˜wnu krwhI ]",
         "ਸੰਧਿਆ ਪ੍ਰਾਤ ਇਸ੍ਨਾਨੁ ਕਰਾਹੀ ॥"),
        ("BwTI ggnu isMi|Aw Aru cuMi|Aw knk kls ieku pwieAw ]",
         "ਭਾਠੀ ਗਗਨੁ ਸਿੰਙਿਆ ਅਰੁ ਚੁੰਙਿਆ ਕਨਕ ਕਲਸ ਇਕੁ ਪਾਇਆ ॥"),
        ("iqn ky nwm Anyk Anµq ]",
         "ਤਿਨ ਕੇ ਨਾਮ ਅਨੇਕ ਅਨੰਤ ॥"),
        ("6 : sMgq dw Asr-nwn`qÍ",
         "੬ : ਸੰਗਤ ਦਾ ਅਸਰ-ਨਾਨੱਤ੍ਵ"),
        ("iPir puCix isD nwnkw! mwq lok ivic ikAw vrqwrw?",
         "ਫਿਰਿ ਪੁਛਣਿ ਸਿਧ ਨਾਨਕਾ! ਮਾਤ ਲੋਕ ਵਿਚਿ ਕਿਆ ਵਰਤਾਰਾ?"),
        ("5 : jpujI AMqly slok ‘pvx gurU’ dw ArQ",
         "੫ : ਜਪੁਜੀ ਅੰਤਲੇ ਸਲੋਕ ‘ਪਵਣ ਗੁਰੂ’ ਦਾ ਅਰਥ"),
        ("hr do Awlm kImiq X¤k qwir mUie Xwir mw ] 2 ] 1 ]",
         "ਹਰ ਦੋ ਆਲਮ ਕੀਮਤਿ ਯੱਕ ਤਾਰਿ ਮੂਇ ਯਾਰਿ ਮਾ ॥ ੨ ॥ ੧ ॥"),
        ("vwihgurU jI kI &qh ]",
         "ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਹ ॥"),
        ("ik®s˜w qy jwnaU hir hir nwcMqI nwcnw ]1]",
         "ਕ੍ਰਿਸ੍ਨਾ ਤੇ ਜਾਨਊ ਹਰਿ ਹਰਿ ਨਾਚੰਤੀ ਨਾਚਨਾ ॥੧॥"),
        ("ibKu kw kIVw ibKu isau lwgw ibs†w mwih smweI ]",
         "ਬਿਖੁ ਕਾ ਕੀੜਾ ਬਿਖੁ ਸਿਉ ਲਾਗਾ ਬਿਸ੍ਟਾ ਮਾਹਿ ਸਮਾਈ ॥"),
        ("Asçrj rUpM rhMq jnmM ]",
         "ਅਸ੍ਚਰਜ ਰੂਪੰ ਰਹੰਤ ਜਨਮੰ ॥"),
        ("duKu prhir suKu Gir lY jwie ]",
         "ਦੁਖੁ ਪਰਹਰਿ ਸੁਖੁ ਘਰਿ ਲੈ ਜਾਇ ॥"),
        ("kwrHw quJY n ibAwpeI nwnk imtY aupwiD ]1]",
         "ਕਾਰ੍ਹਾ ਤੁਝੈ ਨ ਬਿਆਪਈ ਨਾਨਕ ਮਿਟੈ ਉਪਾਧਿ ॥੧॥"),
        ("sB lwlc iqAwg dey igRh ky iek sÎwm ky pÎwr kI hY su BuKI ]636]",
         "ਸਭ ਲਾਲਚ ਤਿਆਗ ਦਏ ਗ੍ਰਿਹ ਕੇ ਇਕ ਸ੍ਯਾਮ ਕੇ ਪ੍ਯਾਰ ਕੀ ਹੈ ਸੁ ਭੁਖੀ ॥੬੩੬॥"),
        ("AnfMf bwFÎ ]7]93]",
         "ਅਨਡੰਡ ਬਾਢ੍ਯ ॥੭॥੯੩॥"),
        ("hM BI vM\\w fumxI rovw JIxI bwix ]2]",
         "ਹੰ ਭੀ ਵੰਞਾ ਡੁਮਣੀ ਰੋਵਾ ਝੀਣੀ ਬਾਣਿ ॥੨॥"),
        ("mhW du`K pwvY iqs ko kwL(35)",
         "ਮਹਾਂ ਦੁੱਖ ਪਾਵੈ ਤਿਸ ਕੋ ਕਾਲ਼(੩੫)"),
        ("ikDO sMKnI icqRnI pdmnI hY ]23]191]",
         "ਕਿਧੌ ਸੰਖਨੀ ਚਿਤ੍ਰਨੀ ਪਦਮਨੀ ਹੈ ॥੨੩॥੧੯੧॥"),
        ("kvnu jogu kaunu g´wnu D´wnu kvn ibiD ausœiq krIAY ]",
         "ਕਵਨੁ ਜੋਗੁ ਕਉਨੁ ਗੵਾਨੁ ਧੵਾਨੁ ਕਵਨ ਬਿਧਿ ਉਸ੍ਤਤਿ ਕਰੀਐ ॥"),
        ("kW@n",
         "ਕੑਾਂਨ"),
        ("s`uD ispwh durMq dubwh su swj snwh durjwn dlYNgy ]",
         "ਸੁੱਧ ਸਿਪਾਹ ਦੁਰੰਤ ਦੁਬਾਹ ਸੁ ਸਾਜ ਸਨਾਹ ਦੁਰਜਾਨ ਦਲੈਂਗੇ ॥"),
        ("sq`Rün ko pl mo bD kIE ]386]",
         "ਸਤ੍ਰੁੱਨ ਕੋ ਪਲ ਮੋ ਬਧ ਕੀਓ ॥੩੮੬॥"),
        ("slok mÚ 3 ]",
         "ਸਲੋਕ ਮਃ ੩ ॥"),
        ("hir Awpy kwn@ü aupwiedw myry goivdw hir Awpy gopI KojI jIau ]",
         "ਹਰਿ ਆਪੇ ਕਾਨੑੁ ਉਪਾਇਦਾ ਮੇਰੇ ਗੋਵਿਦਾ ਹਰਿ ਆਪੇ ਗੋਪੀ ਖੋਜੀ ਜੀਉ ॥"),
        ("Xky dwnh muMgo idZr nu^d nIm ]45]",
         "ਯਕੇ ਦਾਨਹ ਮੁੰਗੋ ਦਿਗ਼ਰ ਨੁਖ਼ਦ ਨੀਮ ॥੪੫॥"),
    ]

    def test_it_returns_what_the_library_it_was_ported_from_returns(self):
        from lib.legacy_font import to_unicode
        for typed, expected in self.LIBRARY:
            with self.subTest(typed=typed):
                self.assertEqual(to_unicode(typed), expected)

    def test_the_sihari_is_typed_before_its_consonant_and_written_after(self):
        from lib.legacy_font import to_unicode
        self.assertEqual(to_unicode("ik"), "ਕਿ")
        self.assertEqual(to_unicode("ikRpw"), "ਕ੍ਰਿਪਾ")           # past the half letter too
        self.assertEqual(to_unicode("ie"), "ਇ")                  # and on the vowel carrier it is the vowel
        self.assertEqual(to_unicode("i"), "ਿ")                   # with nothing after it, it is itself

    def test_a_mark_typed_in_the_wrong_order_is_put_right(self):
        from lib.legacy_font import to_unicode
        self.assertEqual(to_unicode("hUM"), to_unicode("hMU"))
        self.assertEqual(to_unicode("dlYNgy"), to_unicode("dlNYgy"))

    def test_what_is_already_unicode_or_not_a_key_passes_through(self):
        from lib.legacy_font import to_unicode
        self.assertEqual(to_unicode("ਸਤਿ ਨਾਮੁ"), "ਸਤਿ ਨਾਮੁ")
        self.assertEqual(to_unicode(""), "")
        self.assertEqual(to_unicode("(1-23-1)"), "(੧-੨੩-੧)")
        # where the library writes the word "undefined", the character is kept
        self.assertEqual(to_unicode("i, k"), ",ਿ ਕ")
        self.assertNotIn("undefined", to_unicode("i(k) i. i-"))

    def test_the_table_has_one_entry_per_key_and_every_half_letter_in_it(self):
        from lib.legacy_font import CORRECTIONS, HALF, MAPPING
        self.assertEqual(len(MAPPING), 97)
        self.assertTrue(all(len(k) == 1 for k in MAPPING))
        self.assertEqual(HALF - set(MAPPING), set())
        self.assertTrue(all(len(c) == 2 for c in CORRECTIONS))

    def test_a_pdf_in_a_legacy_font_is_read_line_for_line(self):
        # PyMuPDF stood in for: four lines of a pauri in GurbaniAkhar with the
        # id the book prints after each, a gap, and a heading in a Latin font
        from lib import writings_legacy

        def ln(y, text, font):
            return {"bbox": (72.0, y, 400.0, y + 12.0), "spans": [{"text": text, "font": font}]}
        page = SimpleNamespace(get_text=lambda kind: {"blocks": [{"lines": [
            ln(100.0, "suxI pukwr dwqwr pRB (1-23-1)", "GurbaniAkharHeavy"),
            ln(114.0, "gur nwnk jg mwih pTwieAw (1-23-2)", "GurbaniAkharHeavy"),
            ln(128.0, "crn Doie rhrwis kir (1-23-3)", "GurbaniAkharHeavy"),
            ln(142.0, "crxwimRqu isKW pIlwieAw (1-23-4)", "GurbaniAkharHeavy"),
            ln(200.0, "Vaar 1", "TimesNewRoman"),
        ]}]})

        class Doc(list):
            def close(self):
                pass
        real = sys.modules.get("pymupdf")
        sys.modules["pymupdf"] = SimpleNamespace(open=lambda path: Doc([page]))
        try:
            out = writings_legacy.read_legacy_pdf("vaaran.pdf")
        finally:
            if real is None:
                del sys.modules["pymupdf"]
            else:
                sys.modules["pymupdf"] = real
        (only,) = out["pages"]
        self.assertEqual(only["marker"], "1.23")
        self.assertEqual([p["text"] for p in only["paragraphs"]],
                         ["ਸੁਣੀ ਪੁਕਾਰ ਦਾਤਾਰ ਪ੍ਰਭ ਗੁਰ ਨਾਨਕ ਜਗ ਮਾਹਿ ਪਠਾਇਆ "
                          "ਚਰਨ ਧੋਇ ਰਹਰਾਸਿ ਕਰਿ ਚਰਣਾਮ੍ਰਿਤੁ ਸਿਖਾਂ ਪੀਲਾਇਆ", "Vaar 1"])


class MergeStageTests(unittest.TestCase):
    """The M1 stages: multi-span matching, block segmentation, voting, the lexicon gate, correction."""

    def test_two_verse_lines_in_one_reading_both_match(self):
        from lib.ocr_match import match_text_all
        idx = CorpusIndex.from_rows([(1, 1, 284, "ਨਾਮ ਕੇ ਧਾਰੇ ਸਗਲੇ ਜੰਤ ॥"), (2, 1, 284, "ਨਾਮ ਕੇ ਧਾਰੇ ਖੰਡ ਬ੍ਰਹਮੰਡ ॥"),
                                     (3, 1, 284, "ਨਾਮ ਕੇ ਧਾਰੇ ਸਗਲ ਆਕਾਰ ॥")])
        found = match_text_all("ਨਾਮ ਕੇ ਧਾਰੇ ਸਗਲੇ ਜੰਤ॥ਨਾਮ ਕੇ ਧਾਰੇ ਖੰਡ ਬ੍ਰਹਮੰਡ॥", idx, None)
        self.assertEqual([m["line_id"] for m in found], [1, 2])

    def test_a_contained_half_line_does_not_make_the_whole_ambiguous(self):
        from lib.ocr_match import match_text_all
        idx = CorpusIndex.from_rows([(1, 1, 943, "ਘਟਿ ਘਟਿ ਸੁੰਨ ਕਾ ਜਾਣੈ ਭੇਉ ॥ ਆਦਿ ਪੁਰਖੁ ਨਿਰੰਜਨ ਦੇਉ ॥"),
                                     (2, 2, 1129, "ਆਦਿ ਪੁਰਖੁ ਨਿਰੰਜਨ ਦੇਉ ॥੬॥੫॥")])
        found = match_text_all("ਘਟਿ ਘਟਿ ਸੁੰਨ ਕਾ ਜਾਣੈ ਭੇਉ॥ ਆਦਿ ਪੁਰਖੁ ਨਿਰੰਜਨ ਦੇਉ॥ (ਰਾਮਕਲੀ ਮ: ੧)", idx, None)
        self.assertEqual([m["line_id"] for m in found], [1])

    def test_key_positions_and_splice_keep_the_citation(self):
        from lib.ocr_text import key_positions, splice
        t = "ਸਗਲ ਸਮਗ੍ਰੀ ਡਰਹਿ ਬਿਆਪੀ॥ (ਮਾਰੂ ਮ:੫-੧)"
        key, spans = key_positions(t)
        self.assertEqual(key, match_key(t))
        end = spans[len(match_key("ਸਗਲ ਸਮਗ੍ਰੀ ਡਰਹਿ ਬਿਆਪੀ")) - 1][1]
        out = splice(t, [(spans[0][0], end, "<CORPUS>")])
        self.assertTrue(out.startswith("<CORPUS>") and out.endswith("(ਮਾਰੂ ਮ:੫-੧)"))

    def test_segment_block_cuts_at_pivot_lines(self):
        from lib.ocr_vote import segment_block
        piv = [{"bbox": [0, 0, 100, 10], "text": "He came like a song of Heaven, and began singing as"},
               {"bbox": [0, 10, 100, 20], "text": "he felt the touch of the breeze and saw the blue expanse of"},
               {"bbox": [0, 20, 100, 30], "text": "sky."}]
        blk = {"bbox": [0, 0, 100, 30], "text": "He came like a song of Heaven and began singing as he felt "
                                                "the touch of the breeze and saw the blue expanse of sky."}
        pieces = segment_block(blk, piv)
        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[0]["text"].startswith("He came") and pieces[2]["text"].startswith("sky"))
        self.assertEqual(pieces[1]["bbox"], [0, 10, 100, 20])

    def test_vote_majority_and_agreement(self):
        from lib.ocr_vote import vote
        v = vote([{"engine": "a", "text": "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਬਉ ਮਿਟੈ", "conf": 0.9},
                  {"engine": "b", "text": "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ", "conf": 0.9},
                  {"engine": "c", "text": "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ", "conf": 0.9}])
        self.assertEqual(v["text"], "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ")
        self.assertLess(v["agreement"], 1.0)
        alone = vote([{"engine": "a", "text": "ਨਿਰਭਉ", "conf": 0.5}])
        self.assertEqual((alone["text"], alone["agreement"], alone["n"]), ("ਨਿਰਭਉ", None, 1))

    def test_lexicon_gate_and_corrector(self):
        from lib.ocr_correct import Corrector
        from lib.ocr_lexicon import Lexicon
        lex = Lexicon.from_words(gurbani=["ਨਾਮੁ", "ਜਪੋ"], modern=["ਭਾਰੀ", "ਪੰਜ", "ਕਿਤਾਬ"])
        self.assertTrue(lex.knows("ਭਾਰੀ") and lex.knows("ਨਾਮੁ,") and lex.knows("੧੨੩"))
        self.assertFalse(lex.knows("ਬਾਰੀ"))
        rate, unknown = lex.oov("ਬਾਰੀ ਕਿਤਾਬ ਪੰਜ")
        self.assertEqual((round(rate, 2), unknown), (0.33, ["ਬਾਰੀ"]))
        c = Corrector(lex)
        self.assertEqual(c.correct_word("ਬਾਰੀ"), ("ਭਾਰੀ", "confusion"))       # ਬ/ਭ, one seeded confusion
        self.assertEqual(c.correct_word("ਭਾਰੀ"), ("ਭਾਰੀ", None))             # known: never touched
        self.assertEqual(c.correct_word("ਪੰਜਵੇਂ")[1], None)                   # a dropped syllable is no confusion
        text, changes = c.correct_line("ਬਾਰੀ ਕਿਤਾਬ", only={"ਕਿਤਾਬ"})
        self.assertEqual((text, changes), ("ਬਾਰੀ ਕਿਤਾਬ", []))               # only disputed words are candidates

    def test_bold_split_needs_two_modes(self):
        from lib.ocr_layout import bold_split
        self.assertIsNone(bold_split([4.1, 4.15, 4.2, 4.12, 4.18]))
        split = bold_split([4.1, 4.15, 4.2, 4.12, 4.6, 4.65, 4.58])
        self.assertTrue(split and 4.2 < split < 4.58)

    def test_two_faint_lines_do_not_make_the_rest_of_the_page_bold(self):
        # Ten Masters p.88: widths 4.13 and 4.81 below, 34 lines of 4.81-5.17
        # above -- one weight with two thin outliers, not a bold page
        from lib.ocr_layout import bold_split
        page = [4.13, 4.81] + [4.81 + 0.01 * i for i in range(34)]
        self.assertIsNone(bold_split(page))

    def test_a_long_bold_line_the_merge_called_prose_is_prose_to_the_reader(self):
        # Jiwan Sant Attar Singh p.154 (a grey 120 dpi scan): 15 of 31 lines
        # read "bold" from stroke width alone; the merge classified them as
        # commentary all the same (long, no verse mark), and the reader must
        # not turn each into a quotation and a paragraph break
        from lib.ocr_layout import as_runs, page_paragraphs
        lines = [{"n": i + 1, "zone": "body", "kind": "commentary", "bold": i % 2 == 1,
                  "text": "ਇਹ ਸਤਰ ਨੰਬਰ %d ਹੈ ਜੋ ਪੈਰੇ ਦਾ ਹਿੱਸਾ ਹੈ ਤੇ ਅੱਗੇ ਚਲਦੀ ਹੈ" % i,
                  "bbox": [360, 400 + 120 * i, 3200, 508 + 120 * i]} for i in range(6)]
        self.assertFalse(any(r["italic"] for r in as_runs(lines, 4750)))
        self.assertEqual(len(page_paragraphs(lines, 4750, None, layout="columns")), 1)
        # what the merge did call a verse or a heading is still set apart
        verse = dict(lines[0], kind="gurbani-unmatched", bold=True, text="ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ॥")
        title = dict(lines[0], kind="heading", bold=True, text="ਸਿਰਲੇਖ", bbox=[360, 100, 900, 208])
        self.assertEqual([r["italic"] for r in as_runs([title, verse, lines[2]], 4750)], [True, True, False])

    def test_line_boxes_that_vary_with_their_letters_are_one_paragraph(self):
        # OCR boxes of one paragraph's lines are 100-113 px tall as ascenders
        # and descenders come and go; the size rule written for a PDF's exact
        # font sizes cut a paragraph at every line (226 of 263 body paragraphs
        # of the Sant Attar Singh sample were one line)
        from lib.ocr_layout import page_paragraphs
        heights = [100, 113, 104, 110, 101, 108]
        lines = [{"n": i + 1, "zone": "body", "kind": "commentary", "text": "ਇਹ ਸਤਰ ਨੰਬਰ %d ਹੈ ਜੋ ਪੈਰੇ ਦਾ ਹਿੱਸਾ ਹੈ" % i,
                  "bbox": [360, 400 + 120 * i, 3200, 400 + 120 * i + h]} for i, h in enumerate(heights)]
        paras = page_paragraphs(lines, 4750, None, layout="columns")
        self.assertEqual(len(paras), 1)
        big = dict(lines[0], bbox=[360, 200, 1800, 200 + 150], text="ਸਿਰਲੇਖ")     # a real heading, 1.4x the body
        self.assertEqual([p["style"] for p in page_paragraphs([big] + lines, 4750, None, layout="columns")],
                         ["heading", "body"])

    def test_reader_dispatch_shape(self):
        import importlib
        ingest = importlib.import_module("12_ingest_writings")
        with tempfile.TemporaryDirectory() as d:
            book = os.path.join(d, "b")
            os.makedirs(os.path.join(book, "merged"))
            json.dump({"pdf": "x.pdf", "pages": []}, open(os.path.join(book, "pages.json"), "w"))
            lines = [{"_meta": {"page": 1, "hints": {"book_page": 23}, "columns": [], "page_w": 1000, "page_h": 1000}},
                     {"n": 1, "zone": "body", "bbox": [100, 100, 900, 140], "kind": "commentary", "text": "ਇਹ ਪਹਿਲੀ ਸਤਰ ਹੈ।"},
                     {"n": 2, "zone": "body", "bbox": [100, 150, 900, 190], "kind": "commentary", "text": "ਇਹ ਦੂਜੀ ਸਤਰ ਹੈ।"},
                     {"n": 3, "zone": "body", "bbox": [200, 260, 800, 300], "kind": "gurbani", "bold": True,
                      "text": "ਨਿਰਭਉ ਜਪੈ ਸਗਲ ਭਉ ਮਿਟੈ ॥", "matches": [{"line_id": 7, "shabad_id": 3, "ang": 22, "score": 1.0}]},
                     {"n": 4, "zone": "footnote", "bbox": [100, 900, 900, 930], "text": "੧ ਟਿੱਪਣੀ"}]
            with open(os.path.join(book, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
                for ln in lines:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            from lib.writings_ocr import read_ocr_book
            doc = read_ocr_book(book)
            page = doc["pages"][0]
            self.assertEqual((page["page"], page["marker"], page["spread"]), (1, "23", False))
            styles = [p["style"] for p in page["paragraphs"]]
            self.assertIn("quote", styles)
            self.assertIn("footnote", styles)
            quote = next(p for p in page["paragraphs"] if p["style"] == "quote")
            self.assertEqual((quote["line_ids"], quote["ang"], quote["match_method"]), ([7], 22, "ocr-corpus-match"))
            self.assertEqual((quote["source"], quote["line_from"], quote["line_to"]), ("G", 7, 7))
            self.assertTrue(all(k in doc for k in ("path", "body_size", "pages")))


class CoverageTests(unittest.TestCase):
    """lib/ocr_coverage.py: the ink no recognised line covers, on pages drawn for the purpose."""

    W, H, LINE_H = 1000, 1400, 36

    def page(self, rows, rule_x=None):
        """A white page with "text" drawn as rows of glyph-like blocks; rows = [(x0, x1, y0)]."""
        import numpy as np
        img = np.full((self.H, self.W), 255, dtype=np.uint8)
        for x0, x1, y0 in rows:
            for x in range(x0, x1 - 8, 24):                          # a third of the row inked, as text is
                img[y0:y0 + self.LINE_H, x:x + 8] = 0
        if rule_x is not None:
            img[150:1250, rule_x - 1:rule_x + 2] = 0
        return img

    @staticmethod
    def box(x0, x1, y0, h=36):
        return {"bbox": [x0, y0, x1, y0 + h], "text": "ਸਤਰ", "zone": "body"}

    def test_a_skipped_band_is_found_and_a_covered_page_yields_nothing(self):
        from lib.ocr_coverage import uncovered_regions
        rows = [(80, 380, y) for y in (300, 360, 420, 480, 540, 600)] + [(440, 940, y) for y in (300, 360, 420, 480, 540, 600)]
        img = self.page(rows, rule_x=400)
        lines = [self.box(x0, x1, y0) for x0, x1, y0 in rows]
        skipped = lines.pop(3)                                        # the fourth left-column line was never segmented
        got = uncovered_regions(img, lines, [(0, 400), (400, 1000)], 36, self.W, self.H, rule_x=400)
        cands = [r for r in got["regions"] if r["status"] == "candidate"]
        self.assertEqual(len(cands), 1)
        x0, y0, x1, y1 = cands[0]["bbox"]
        self.assertEqual(cands[0]["col"], 0)
        self.assertTrue(x0 >= 80 and x1 <= 380 and y0 >= 480 - 5 and y1 <= 480 + 36 + 5, cands[0])
        self.assertGreater(got["residual_share"], 0.05)
        whole = uncovered_regions(img, lines + [skipped], [(0, 400), (400, 1000)], 36, self.W, self.H, rule_x=400)
        self.assertEqual(whole["regions"], [])
        self.assertLess(whole["residual_share"], 0.003)

    def test_a_line_cut_to_its_x_height_is_still_a_line_and_a_sliver_is_not(self):
        # the row projection cuts a run at the valley under the headline, so a
        # line's region is often its x-height alone: 0.41-0.48 of the line on
        # the Santhya, where 0.5 refused 968 real lines
        from lib.ocr_coverage import uncovered_regions
        img = self.page([])
        for x in range(80, 900, 24):
            img[500:516, x:x + 8] = 0                                     # 16 of a 36 px line: 0.44
        for x in range(80, 900, 24):
            img[800:804, x:x + 8] = 0                                     # 4 px: a sliver of matras
        got = uncovered_regions(img, [], [(0, self.W)], 36, self.W, self.H)
        by_status = {(r["status"], r.get("why")): r for r in got["regions"]}
        self.assertIn(("candidate", None), by_status)
        self.assertAlmostEqual(by_status[("candidate", None)]["h_rel"], 0.44, places=1)
        self.assertIn(("rejected", "short"), by_status)
        self.assertLess(by_status[("rejected", "short")]["h_rel"], 0.2)

    def test_rules_pictures_and_dust_are_not_lines(self):
        import numpy as np
        from lib.ocr_coverage import uncovered_regions
        rows = [(80, 920, y) for y in (300, 360, 420)]
        img = self.page(rows)
        img[700:702, 100:900] = 0                                       # a horizontal rule
        img[800:1100, 300:600] = 0                                      # a solid block: a picture
        rng = np.random.default_rng(1)
        for _ in range(40):                                              # dust
            y, x = rng.integers(1150, 1250), rng.integers(100, 900)
            img[y, x] = 0
        lines = [self.box(x0, x1, y0) for x0, x1, y0 in rows]
        got = uncovered_regions(img, lines, None, 36, self.W, self.H)
        self.assertEqual([r for r in got["regions"] if r["status"] == "candidate"], [])
        whys = {r["why"] for r in got["regions"]}
        self.assertTrue(whys & {"short", "rule", "tall", "dense"}, whys)

    def test_a_region_inside_a_short_box_is_that_box(self):
        from lib.ocr_coverage import uncovered_regions
        rows = [(80, 920, 300)]
        img = self.page(rows)
        short = self.box(80, 700, 300)                                  # the box stops 220 px short of the ink
        got = uncovered_regions(img, [short], None, 36, self.W, self.H)
        # what shows past the padding is a region, but most of nothing lies in the box: a candidate
        # only if it is clear of the line; here the leftover is beside it, so it stands on its own
        self.assertTrue(all(r["status"] in ("candidate", "rejected") for r in got["regions"]))
        for r in got["regions"]:
            self.assertGreaterEqual(r["bbox"][0], 700 - 9 - 1)

    def test_crop_box_and_text_readings(self):
        from lib.ocr_coverage import crop_box, is_text_reading
        self.assertEqual(crop_box({"bbox": [100, 500, 380, 536]}, (0, 400), 36, 1000, 1400), [82, 482, 398, 554])
        self.assertEqual(crop_box({"bbox": [10, 5, 380, 40]}, (0, 400), 36, 1000, 1400), [0, 0, 398, 58])
        self.assertIsNone(is_text_reading("ਵਾਜੇ ਤੇਰੇ ਨਾਦ ਅਨੇਕ ਅਸੰਖਾ", "pa"))
        self.assertEqual(is_text_reading("", "pa"), "empty")
        self.assertEqual(is_text_reading("| | ॥ ॥ . .", "pa"), "no-letters")
        self.assertEqual(is_text_reading("ਾਾਾਾ", "pa"), "junk")

    def test_typical_height_falls_back_to_the_page(self):
        from lib.ocr_coverage import typical_height
        self.assertAlmostEqual(typical_height([], 3000), 54.0)
        self.assertEqual(typical_height([self.box(0, 100, y, h) for y, h in ((0, 40), (50, 42), (100, 44), (150, 46), (200, 48))], 3000), 44)


class SpeckTests(unittest.TestCase):
    """22_ocr_merge.drop_specks: a fragment of the rule is not a line."""

    def test_rule_fragments_are_dropped_and_lines_are_kept(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")
        lines = [{"zone": "body", "split": True, "bbox": [480, 909, 547, 935], "text": "HL ਮੁਲ"},        # letters, but 67 by 26 px of a 73 px line
                 {"zone": "body", "split": True, "bbox": [1375, 923, 1460, 926], "text": "---------"},   # 3 px
                 {"zone": "body", "split": True, "bbox": [555, 1061, 574, 1074], "text": "="},           # no letters
                 {"zone": "body", "split": True, "bbox": [272, 960, 727, 1039], "text": "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ"},
                 {"zone": "body", "split": False, "bbox": [100, 200, 130, 210], "text": "-"},            # not a split piece: left alone
                 {"zone": "header", "split": True, "bbox": [100, 20, 130, 25], "text": "-"}]
        self.assertEqual(m.drop_specks(lines, 73.0, "pa"), 3)
        self.assertEqual([ln["zone"] for ln in lines], ["speck", "speck", "speck", "body", "body", "header"])


class AlignTests(unittest.TestCase):
    """lib/ocr_vote.align_boxes: what another engine's reading of a pivot line is."""

    @staticmethod
    def words(text, x0, y0, step=60, h=50):
        out = []
        for k, w in enumerate(text.split()):
            out.append({"text": w, "bbox": [x0 + k * step, y0, x0 + k * step + step - 10, y0 + h], "conf": 0.9})
        return out

    def test_a_whole_row_is_cut_to_the_half_the_pivot_split_off(self):
        from lib.ocr_vote import align_boxes
        row = self.words("ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ | ਕੋਈ ਜਿਸ ਨੂੰ ਕਿ ਜਨਮ ਤੋਂ", 100, 1000)
        other = [{"text": " ".join(w["text"] for w in row), "bbox": [100, 1000, row[-1]["bbox"][2], 1050], "words": row}]
        left = {"text": "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ", "bbox": [100, 1000, 390, 1050], "words": row[:5]}
        right = {"text": "ਕੋਈ ਜਿਸ ਨੂੰ ਕਿ ਜਨਮ ਤੋਂ", "bbox": [460, 1000, 810, 1050], "words": row[6:]}
        got = align_boxes([left, right], other)
        self.assertEqual(got[0][0]["text"], "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ")
        self.assertEqual(got[1][0]["text"], "ਕੋਈ ਜਿਸ ਨੂੰ ਕਿ ਜਨਮ ਤੋਂ")            # the rule read as a bar falls between the halves
        self.assertTrue(got[0][0]["clipped"] and got[1][0]["clipped"])
        # a tiny piece the segmenter made of a rule gets nothing, not the sentence
        speck = {"text": "=", "bbox": [555, 1061, 574, 1074], "words": []}
        self.assertEqual(align_boxes([speck], other).get(0, []), [])

    def test_a_line_of_the_same_width_is_left_as_read(self):
        from lib.ocr_vote import align_boxes
        ws = self.words("ਇਹ ਪ੍ਰਸ਼ਨ ਹੁੰਦਾ ਹੈ", 100, 200)
        other = [{"text": "ਇਹ ਪ੍ਰਸਨ ਹੁੰਦਾ ਹੈ", "bbox": [100, 200, 340, 250], "words": ws}]
        pivot = [{"text": "ਇਹ ਪ੍ਰਸ਼ਨ ਹੁੰਦਾ ਹੈ", "bbox": [98, 202, 335, 252], "words": ws}]
        self.assertEqual(align_boxes(pivot, other)[0][0]["text"], "ਇਹ ਪ੍ਰਸਨ ਹੁੰਦਾ ਹੈ")
        self.assertNotIn("clipped", align_boxes(pivot, other)[0][0])


class RecoverTests(unittest.TestCase):
    """22_ocr_merge.Merger.recover_regions: candidates become lines with their provenance, or are refused."""


    def test_a_crop_tesseract_dies_on_is_refused_and_the_page_goes_on(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")

        class TesseractError(Exception):
            pass

        class Dies:
            def recognise_region(self, img, crop, lang, psm):
                raise TesseractError(-8, "")
        merger = m.Merger.__new__(m.Merger)
        merger.pivot, merger.engines, merger.lang = "tesseract-pan", ["tesseract-pan"], "pa"
        merger.coverage = {"psm": 7, "max": 12, "min_agreement": 0.0, "route": False}
        merger.recognisers = {"tesseract-pan": Dies()}
        regions = [{"bbox": [200, 1000, 700, 1050], "col": 0, "h_rel": 1.0, "w_rel": 10.0, "density": 0.2,
                    "status": "candidate", "why": None}]
        lines, _ = merger.recover_regions(None, regions, None, 50.0, 2000, 3000, None)
        self.assertEqual(lines, [])
        self.assertEqual((regions[0]["status"], regions[0]["why"]), ("rejected", "engine-error"))
    def test_candidates_are_read_and_provenance_kept(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")

        class FakeTess:
            def __init__(self, readings):
                self.readings, self.calls = readings, []

            def recognise_region(self, img, crop, lang, psm):
                self.calls.append((tuple(crop), psm))
                text = self.readings.pop(0)
                if not text:
                    return []
                return [{"bbox": [crop[0] + 5, crop[1] + 5, crop[2] - 5, crop[3] - 5], "text": text, "conf": 0.8,
                         "words": [], "block": 0, "par": 0, "zone": None, "psm": psm}]
        merger = m.Merger.__new__(m.Merger)
        merger.pivot, merger.engines, merger.lang = "tesseract-pan", ["tesseract-pan", "tesseract-gurmukhi"], "pa"
        merger.coverage = {"psm": 7, "max": 12, "min_agreement": 0.0, "route": False}
        # the empty and the no-letters crops are each tried again in raw mode (psm 13) and refused again
        pivot = FakeTess(["ਸੁਣਿਐ ਸਤੁ ਸੰਤੋਖੁ ਗਿਆਨੁ ॥", "", "", "॥ | ॥", "॥ | ॥"])
        other = FakeTess(["ਸੁਣਿਐ ਸਤ ਸੰਤੋਖ ਗਿਆਨ ॥"])
        merger.recognisers = {"tesseract-pan": pivot, "tesseract-gurmukhi": other}
        regions = [{"bbox": [200, 1000, 700, 1050], "col": 0, "h_rel": 1.0, "w_rel": 10.0, "density": 0.2, "status": "candidate", "why": None},
                   {"bbox": [200, 1200, 700, 1250], "col": 0, "h_rel": 1.0, "w_rel": 10.0, "density": 0.2, "status": "candidate", "why": None},
                   {"bbox": [900, 1400, 1400, 1450], "col": 1, "h_rel": 1.0, "w_rel": 10.0, "density": 0.2, "status": "candidate", "why": None},
                   {"bbox": [900, 1600, 1400, 1650], "col": 1, "h_rel": 3.0, "w_rel": 10.0, "density": 0.2, "status": "rejected", "why": "tall"}]
        lines, alts = merger.recover_regions(None, regions, [(0, 800), (800, 2000)], 50, 2000, 3000, rule_y=2800)
        self.assertEqual([r["status"] for r in regions], ["recovered", "rejected", "rejected", "rejected"])
        self.assertEqual([r["why"] for r in regions], [None, "empty", "no-letters", "tall"])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["zone"], "body")
        self.assertEqual(lines[0]["recovered"]["region"], 0)
        self.assertEqual(lines[0]["recovered"]["engine"], "tesseract-pan")
        self.assertEqual(lines[0]["recovered"]["crop"], [175, 975, 725, 1075])      # padded by half a line, in its band
        self.assertEqual(lines[0]["bbox"], [200, 988, 700, 1062])                     # the ink's box and a quarter line, not the crop
        self.assertEqual(lines[0]["recovered"]["psm"], 7)
        # the other variant read only the crop that was kept, for the vote
        self.assertEqual(len(alts["tesseract-gurmukhi"]), 1)
        self.assertEqual(other.calls, [((175, 975, 725, 1075), 7)])
        self.assertEqual([psm for _, psm in pivot.calls], [7, 7, 13, 7, 13])

    def test_a_crop_the_segmenter_refuses_is_read_in_raw_mode(self):
        # Tesseract's own layout analysis rejected the line on the page and
        # rejects the crop of it too (a grey band behind the text); raw line
        # mode has no layout analysis to reject it with
        from importlib import import_module
        m = import_module("22_ocr_merge")

        class FakeTess:
            def __init__(self):
                self.calls = []

            def recognise_region(self, img, crop, lang, psm):
                self.calls.append(psm)
                if psm != 13:
                    return []
                return [{"bbox": list(crop), "text": "ਸੰਤ ਹੋਏ, ਤਾਂ ਮੈਨੂੰ ਅਕਾਲ ਪੁਰਖ ਦਾ ਦਰਸ਼ਨ ਕਰਵਾ ਦੇਣਗੇ।", "conf": 0.9,
                         "words": [], "block": 0, "par": 0, "zone": None, "psm": psm}]
        merger = m.Merger.__new__(m.Merger)
        merger.pivot, merger.engines, merger.lang = "tesseract-pan", ["tesseract-pan", "tesseract-gurmukhi"], "pa"
        merger.coverage = {"psm": 7, "max": 40, "min_agreement": 0.0, "route": False}
        pivot, other = FakeTess(), FakeTess()
        merger.recognisers = {"tesseract-pan": pivot, "tesseract-gurmukhi": other}
        regions = [{"bbox": [200, 1000, 1700, 1060], "col": 0, "h_rel": 0.6, "w_rel": 15.0, "density": 0.25, "status": "candidate", "why": None}]
        lines, alts = merger.recover_regions(None, regions, None, 100, 2000, 3000, rule_y=None)
        self.assertEqual(regions[0]["status"], "recovered")
        self.assertTrue(regions[0].get("raw"))
        self.assertEqual(pivot.calls, [7, 13])
        self.assertEqual(lines[0]["bbox"], [200, 975, 1700, 1085])                  # the region's box and a quarter line, not the crop's
        self.assertEqual(lines[0]["recovered"]["psm"], 13)                       # the mode that read it is the one recorded
        self.assertEqual(other.calls, [13])                                      # the other voter reads it the same way
        self.assertEqual(len(alts["tesseract-gurmukhi"]), 1)

    def test_a_recovered_line_sits_where_its_words_are(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")

        class WordyTess:
            def recognise_region(self, img, crop, lang, psm):
                words = [{"text": "ਸੰਤ", "bbox": [crop[0] + 30, crop[1] + 28, crop[0] + 130, crop[1] + 92], "conf": 0.9},
                         {"text": "ਹੋਏ,", "bbox": [crop[0] + 150, crop[1] + 25, crop[0] + 260, crop[1] + 95], "conf": 0.9}]
                return [{"bbox": list(crop), "text": "ਸੰਤ ਹੋਏ,", "conf": 0.9, "words": words, "block": 0, "par": 0, "zone": None}]
        merger = m.Merger.__new__(m.Merger)
        merger.pivot, merger.engines, merger.lang = "tesseract-pan", ["tesseract-pan"], "pa"
        merger.coverage = {"psm": 7, "max": 40, "min_agreement": 0.0, "route": False}
        merger.recognisers = {"tesseract-pan": WordyTess()}
        regions = [{"bbox": [200, 1000, 700, 1050], "col": 0, "h_rel": 1.0, "w_rel": 10.0, "density": 0.2, "status": "candidate", "why": None}]
        lines, _ = merger.recover_regions(None, regions, None, 50, 2000, 3000, rule_y=None)
        crop = lines[0]["recovered"]["crop"]
        # across: the words; up and down: the ink band (1000-1050) and a quarter of a 50 px line
        self.assertEqual(lines[0]["bbox"], [crop[0] + 30, 988, crop[0] + 260, 1062])

    def test_a_footnote_region_and_a_tall_one(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")

        class FakeTess:
            def recognise_region(self, img, crop, lang, psm):
                return [{"bbox": list(crop), "text": "ਟਿੱਪਣੀ ਦੀ ਸਤਰ", "conf": None, "words": [], "block": 0, "par": 0, "zone": None}]
        merger = m.Merger.__new__(m.Merger)
        merger.pivot, merger.engines, merger.lang = "tesseract", ["tesseract"], "pa"
        merger.coverage = {"psm": 7, "max": 12, "min_agreement": 0.0, "route": False}
        merger.recognisers = {"tesseract": FakeTess()}
        regions = [{"bbox": [200, 2850, 700, 2950], "col": 0, "h_rel": 2.0, "w_rel": 10.0, "density": 0.2, "status": "candidate", "why": None}]
        lines, alts = merger.recover_regions(None, regions, None, 50, 2000, 3000, rule_y=2800)
        self.assertEqual(lines[0]["zone"], "footnote")
        self.assertEqual(lines[0]["recovered"]["psm"], 6)                          # taller than a line: a block
        self.assertIsNone(lines[0]["recovered"]["conf"])
        self.assertEqual(alts, {})


class PairTests(unittest.TestCase):
    """lib/ocr_pairs + page_paragraphs(layout="auto"): a two-column page read in bands."""

    H = 55

    def ln(self, n, y, x0, x1, text, kind="commentary", bold=False, matches=None):
        d = {"n": n, "zone": "body", "bbox": [x0, y, x1, y + self.H], "text": text, "kind": kind, "bold": bold,
             "col": 0 if x0 < 743 else 1}
        if matches:
            d["matches"] = matches
        return d

    def page70(self):
        # page 70 of the Santhya, in outline: a preface across the page that the
        # merge cut at the gutter, a section heading, the ਮੂਲ|ਅਰਥ row, then the
        # paired band, verse left and arth right
        L = self.ln
        return [
            L(6, 279, 239, 733, "ਆਪਾ ਦਿੜ ਕਰਦੀ ਹੈ. ਇਸ"), L(7, 278, 747, 1908, "ਭਾਵ ਤੋਂ ਮੋਹ ਮਾਇਆ ਵਿਚ ਫਸਦੀ ਹੈ."),
            L(11, 495, 923, 1240, "( ਪਉੜੀ ੩ )", kind="heading", bold=True),
            L(12, 628, 388, 733, "ਪ੍ਰਾਕਥਨ- 'ਜਿਸ", kind="heading", bold=True), L(13, 626, 747, 1910, "ਦੇ ਹੁਕਮ ਦਾ ਵਰਣਨ ਕਰ ਰਹੇ ਹੋ"),
            L(15, 705, 241, 739, "ਇਹ ਪ੍ਰਸ਼ਨ ਹੁੰਦਾ ਹੈ। ਉੱਤਰ ਦੇਂ"), L(16, 705, 747, 1916, "ਰਹੇ ਹਨ ਕਿ ਉਸ ਨੂੰ ਸਾਰਾ ਵਰਣਨ ਕਰ ਸਕਣਾ"),
            L(19, 885, 482, 543, "ਮੂਲ", kind="heading", bold=True), L(20, 885, 1375, 1463, "ਅਰਥ", kind="heading", bold=True),
            L(22, 970, 270, 724, "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ", kind="heading", bold=True,
              matches=[{"line_id": 40, "shabad_id": 12, "ang": 2, "score": 0.9, "source": "G"}]),
            # each arth is two lines set close, and a wider gap before the next
            L(21, 962, 769, 1914, "ਕੋਈ ਜਿਸ ਨੂੰ ਕਿ (ਜਨਮ ਤੋਂ) ਬਲ ਆਇਆ ਹੈ (ਉਹ)"),
            L(23, 1025, 764, 1917, "ਉਸ ਦੇ ਬਲ ਨੂੰ ਗਾਉਂਦਾ ਹੈ।"),
            L(24, 1130, 269, 723, "ਤਾਣੁ॥ ਗਾਵੈ ਕੋ ਦਾਤਿ", kind="gurbani-unmatched", bold=True),
            L(25, 1122, 764, 1917, "ਕੋਈ (ਉਸ ਦੀ) ਦਾਤ ਨੂੰ ਗਾਉਂਦਾ ਹੈ, (ਜਿਸ ਨੂੰ)"),
            L(27, 1185, 765, 1919, "ਉਸ ਦੀ ਨਿਸ਼ਾਨੀ ਸਮਝ ਰਿਹਾ ਹੈ।"),
            L(26, 1290, 270, 509, "ਜਾਣੈ ਨੀਸਾਣੁ॥", kind="gurbani-unmatched", bold=True),
            L(29, 1282, 765, 1919, "(ਇਸ ਤਰ੍ਹਾਂ) ਕੋਈ ਉਸ ਦੇ ਗੁਣਾਂ ਨੂੰ"),
            L(31, 1345, 765, 1919, "ਗਾਉਂ ਰਿਹਾ ਹੈ।"),
        ]

    def test_a_line_across_both_columns_is_a_full_row_and_a_bar_does_not_move_a_line(self):
        # our own merge splits a line only where the rule runs, so the preface
        # above the band comes whole; and a bar read out of the rule sits at
        # the left edge of an arth line, over the gutter
        from lib.ocr_pairs import bands, column_of, is_wide
        L = self.ln
        lines = [L(1, 626, 388, 1910, "ਪ੍ਰਾਕਥਨ- 'ਜਿਸ ਦੇ ਹੁਕਮ ਦਾ ਵਰਣਨ ਕਰ ਰਹੇ ਹੋ, ਉਸ ਹੁਕਮ ਦੇ ਹੁਕਮੀ"),
                 L(2, 705, 241, 1916, "ਇਹ ਪ੍ਰਸ਼ਨ ਹੁੰਦਾ ਹੈ। ਉੱਤਰ ਦੇਂ ਰਹੇ ਹਨ ਕਿ ਉਸ ਨੂੰ ਸਾਰਾ ਵਰਣਨ ਕਰ ਸਕਣਾ"),
                 L(3, 885, 482, 1463, "ਮੂਲ ਅਰਥ", kind="heading", bold=True),
                 L(4, 970, 270, 724, "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ ਤਾਣੁ ॥", kind="gurbani", bold=True),
                 L(5, 962, 732, 1914, "| ਕੋਈ ਜਿਸ ਨੂੰ ਕਿ (ਜਨਮ ਤੋਂ) ਬਲ ਆਇਆ ਹੈ (ਉਹ)"),
                 L(6, 1130, 269, 723, "ਗਾਵੈ ਕੋ ਦਾਤਿ ਜਾਣੈ ਨੀਸਾਣੁ ॥", kind="gurbani", bold=True),
                 L(7, 1122, 764, 1917, "ਕੋਈ (ਉਸ ਦੀ) ਦਾਤ ਨੂੰ ਗਾਉਂਦਾ ਹੈ, (ਜਿਸ ਨੂੰ)")]
        self.assertTrue(is_wide(lines[0], 743, self.H) and is_wide(lines[2], 743, self.H))
        self.assertFalse(is_wide(lines[4], 743, self.H))
        self.assertEqual(column_of(lines[4], 743), 1)                       # left edge 732, centre far right
        got = bands(lines, [(0, 743), (743, 2130)], self.H)
        self.assertEqual([b["mode"] for b in got], ["full", "paired"])
        self.assertEqual([ln["n"] for ln in got[0]["lines"]], [1, 2, 3])
        self.assertEqual(sorted(ln["n"] for ln in got[1]["verse"]), [4, 6])
        self.assertEqual(sorted(ln["n"] for ln in got[1]["prose"]), [5, 7])

    def test_a_shabad_quoted_with_its_raag_title_is_one_quotation(self):
        # Jiwan Sant Attar Singh p.154: ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ, the raag title (kind
        # heading: short and bold), then the matched lines; read as prose
        # because one line of the block was not "gurbani"
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "merged"))
            with open(os.path.join(d, "pages.json"), "w", encoding="utf-8") as fh:
                json.dump({"book": "b", "language": "pa", "pages": [{"page": 1}]}, fh)
            H = 100
            def line(n, y, text, kind, bold=True, match=None):
                ln = {"n": n, "zone": "body", "bbox": [300, y, 3200, y + H], "text": text, "kind": kind, "bold": bold, "col": 0}
                if match:
                    ln["matches"] = [{"line_id": match, "shabad_id": 2900, "ang": 684, "score": 1.0, "source": "G"}]
                return ln
            lines = [line(1, 100, "ਸੰਤ ਜੀ ਨੇ ਇਹ ਸ਼ਬਦ ਪੜ੍ਹਿਆ:", "commentary", bold=False),
                     line(2, 220, "ੴ ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ ॥", "gurbani-unmatched"),
                     line(3, 340, "ਧਨਾਸਰੀ ਮਹਲਾ ੯ ॥", "heading"),
                     line(4, 460, "ਕਾਹੇ ਰੇ ਬਨ ਖੋਜਨ ਜਾਈ ॥", "gurbani", match=29001),
                     line(5, 580, "ਸਰਬ ਨਿਵਾਸੀ ਸਦਾ ਅਲੇਪਾ ਤੋਹੀ ਸੰਗਿ ਸਮਾਈ ॥੧॥ ਰਹਾਉ ॥", "gurbani", match=29002),
                     line(6, 720, "ਸਾਰੀ ਸੰਗਤ ਤੇ ਵੈਰਾਗ ਦੀ ਦਸ਼ਾ ਛਾ ਗਈ।", "commentary", bold=False)]
            with open(os.path.join(d, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_meta": {"page": 1, "page_w": 3600, "page_h": 4750, "body_h": H, "columns": None, "hints": {}}}) + "\n")
                for ln in lines:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            from lib.writings_ocr import read_ocr_book
            recs = read_ocr_book(d, layout="auto", kind="essay")["pages"][0]["paragraphs"]
        self.assertEqual([r["style"] for r in recs], ["body", "quote", "body"])
        self.assertIn("ਧਨਾਸਰੀ ਮਹਲਾ ੯", recs[1]["text"])
        self.assertEqual(recs[1]["shabad_id"], 2900)

    def test_a_verse_from_outside_the_work_s_angs_is_a_quotation_not_the_verse_explained(self):
        from lib.writings_ocr import within_angs
        self.assertTrue(within_angs({"ang": 8}, [1, 53]))
        self.assertFalse(within_angs({"ang": 1043}, [1, 53]))
        self.assertTrue(within_angs({"ang": 1043}, None))                  # a work that states no range
        self.assertTrue(within_angs({"unmatched": True}, [1, 53]))         # an unmatched verse is not ruled out
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "merged"))
            with open(os.path.join(d, "pages.json"), "w", encoding="utf-8") as fh:
                json.dump({"book": "b", "language": "pa", "angs": [1, 53], "pages": [{"page": 1}]}, fh)
            H = 55
            def line(n, y, text, kind, ang=None):
                ln = {"n": n, "zone": "body", "bbox": [200, y, 1800, y + H], "text": text, "kind": kind, "col": 0}
                if ang is not None:
                    ln["matches"] = [{"line_id": ang * 10, "shabad_id": ang, "ang": ang, "score": 1.0, "source": "G"}]
                return ln
            lines = [line(1, 100, "ਇਥੇ ਭਾਈ ਸਾਹਿਬ ਇਕ ਹੋਰ ਤੁਕ ਦਾ ਹਵਾਲਾ ਦੇਂਦੇ ਹਨ:-", "commentary"),
                     line(2, 200, "ਨਿਉਲੀ ਕਰਮ ਭੁਇਅੰਗਮ ਭਾਠੀ ॥", "gurbani", ang=1043),
                     line(3, 300, "ਇਹ ਤੁਕ ਸਿਰਫ਼ ਮਿਸਾਲ ਵਜੋਂ ਹੈ, ਪਉੜੀ ਬਾਰੇ ਨਹੀਂ।", "commentary"),
                     line(4, 500, "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥", "gurbani", ang=1),
                     line(5, 600, "ਇਹ ਪਉੜੀ ਦੀ ਤੁਕ ਹੈ ਤੇ ਇਸ ਦਾ ਅਰਥ ਇਹ ਹੈ।", "commentary")]
            with open(os.path.join(d, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_meta": {"page": 1, "page_w": 2000, "page_h": 3000, "body_h": H, "columns": None,
                                               "hints": {"ang_from": 1, "ang_to": 1, "ang_source": "header"}}}) + "\n")
                for ln in lines:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            from lib.writings_ocr import read_ocr_book
            recs = read_ocr_book(d, layout="auto", kind="commentary")["pages"][0]["paragraphs"]
        find = lambda prefix: next(r for r in recs if r["text"].startswith(prefix))
        self.assertNotIn("explains", find("ਇਹ ਤੁਕ ਸਿਰਫ਼"))                  # after the ang 1043 quotation: no link
        self.assertNotIn("pair", find("ਨਿਉਲੀ"))
        self.assertEqual(find("ਇਹ ਪਉੜੀ")["explains"][0]["ang"], 1)         # after the pauri's own line: explained

    def test_full_rows_are_rejoined_and_the_paired_band_reads_verse_then_arth(self):
        from lib.ocr_layout import page_paragraphs
        paras = page_paragraphs(self.page70(), 3000, [(0, 743), (743, 2130)], layout="auto", typical_h=self.H)
        texts = [p["text"] for p in paras]
        # the preface lines the gutter split are whole again, prose, in order
        self.assertIn("ਆਪਾ ਦਿੜ ਕਰਦੀ ਹੈ. ਇਸ ਭਾਵ ਤੋਂ ਮੋਹ ਮਾਇਆ ਵਿਚ ਫਸਦੀ ਹੈ.", " ".join(texts[:2]))
        joined = next(p for p in paras if p["text"].startswith("ਪ੍ਰਾਕਥਨ-"))
        self.assertIn("ਦੇ ਹੁਕਮ ਦਾ ਵਰਣਨ", joined["text"])
        self.assertFalse(joined.get("verse"))
        # one paired band: each arth preceded by the verse beside it, sharing a pair
        pairs = [(p.get("pair"), bool(p.get("verse")), "explains_lines" in p) for p in paras if p.get("pair")]
        self.assertEqual(pairs, [(1, True, False), (1, False, True), (2, True, False), (2, False, True),
                                 (3, True, False), (3, False, True)])
        first_verse = next(p for p in paras if p.get("pair") == 1 and p.get("verse"))
        self.assertEqual(first_verse["text"], "ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ")
        arth = next(p for p in paras if p.get("pair") == 1 and not p.get("verse"))
        self.assertEqual([l["n"] for l in arth["explains_lines"]], [22])
        self.assertEqual([l["n"] for l in next(p for p in paras if p.get("pair") == 3 and not p.get("verse"))["explains_lines"]], [26])
        # the ਮੂਲ|ਅਰਥ row is one heading line before the band
        self.assertIn("ਮੂਲ ਅਰਥ", texts)
        self.assertLess(texts.index("ਮੂਲ ਅਰਥ"), texts.index("ਗਾਵੈ ਕੋ ਤਾਣੁ ਹੋਵੈ ਕਿਸੈ"))

    def test_the_columns_layout_is_the_order_before_pairing(self):
        from lib.ocr_layout import page_paragraphs
        paras = page_paragraphs(self.page70(), 3000, [(0, 743), (743, 2130)], layout="columns")
        self.assertFalse(any(p.get("pair") for p in paras))
        # left column whole, then the right: the last left-column text comes before the first arth
        texts = [p["text"] for p in paras]
        self.assertLess(texts.index(next(t for t in texts if "ਨੀਸਾਣੁ" in t)), texts.index(next(t for t in texts if "ਕੋਈ ਜਿਸ ਨੂੰ" in t)))

    def test_one_paired_row_alone_is_a_lead_word_not_a_band(self):
        from lib.ocr_pairs import bands
        L = self.ln
        lines = [L(1, 100, 239, 733, "ਇਕ ਸਤਰ ਜੋ"), L(2, 100, 747, 1908, "ਗਟਰ ਤੇ ਕੱਟੀ ਗਈ"),
                 L(3, 200, 388, 733, "ਪ੍ਰਾਕਥਨ-", kind="heading", bold=True), L(4, 200, 747, 1910, "ਇਸ ਸ਼ਬਦ ਵਿਚ"),
                 L(5, 300, 239, 733, "ਹੋਰ ਵਾਰਤਕ"), L(6, 300, 747, 1908, "ਪੂਰੀ ਚੌੜਾਈ ਵਿਚ")]
        got = bands(lines, [(0, 743), (743, 2130)], self.H)
        self.assertEqual([b["mode"] for b in got], ["full"])
        self.assertEqual([l["text"] for l in got[0]["lines"]][1], "ਪ੍ਰਾਕਥਨ- ਇਸ ਸ਼ਬਦ ਵਿਚ")
        self.assertEqual(got[0]["lines"][1]["kind"], "commentary")
        self.assertIn("paired", [b["mode"] for b in bands(lines, [(0, 743), (743, 2130)], self.H, force=True)])

    def test_a_reader_of_the_book_writes_pair_and_explains(self):
        from lib.ocr_engines import write_page
        from lib.writings_ocr import read_ocr_book
        with tempfile.TemporaryDirectory() as d:
            json.dump({"pdf": "x.pdf", "pages": [{"page": 1}]}, open(os.path.join(d, "pages.json"), "w", encoding="utf-8"))
            write_page(os.path.join(d, "merged", "0001.jsonl"),
                       {"page": 1, "page_w": 2130, "page_h": 3000, "columns": [[0, 743], [743, 2130]], "body_h": self.H,
                        "hints": {"ang_from": 2, "ang_to": 2, "book_page": 52, "section": None, "ang_source": "header"}},
                       self.page70())
            doc = read_ocr_book(d, layout="auto", kind="commentary")
        page = doc["pages"][0]
        self.assertEqual((page["layout"], page["ang_source"], page["marker"]), ("paired", "header", "52"))
        recs = page["paragraphs"]
        verse = next(r for r in recs if r.get("pair") == 1 and r["style"] == "quote")
        self.assertEqual((verse["line_ids"], verse["source"], verse["line_from"], verse["line_to"]), ([40], "G", 40, 40))
        arth = next(r for r in recs if r.get("pair") == 1 and r["style"] == "body")
        self.assertEqual(arth["explains"], [{"pair": 1, "shabad_id": 12, "line_from": 40, "line_to": 40, "ang": 2,
                                             "source": "G", "score": 0.9, "method": "ocr-corpus-match"}])
        arth2 = next(r for r in recs if r.get("pair") == 2 and r["style"] == "body")
        self.assertEqual(arth2["explains"], [{"pair": 2, "unmatched": True}])
        # the bold lead word's line is prose, the section heading a heading: neither a quote
        self.assertEqual(next(r for r in recs if r["text"].startswith("ਪ੍ਰਾਕਥਨ-"))["style"], "body")
        self.assertEqual(next(r for r in recs if r["text"] == "( ਪਉੜੀ ੩ )")["style"], "heading")


class SanthyaFixtureTests(unittest.TestCase):
    """On the real merged pages of Santhya Vol. 1, where they exist (SANTHYA_MERGED, or data/ocr/santhya-vol-1)."""

    @classmethod
    def setUpClass(cls):
        from lib.paths import OCR_DIR
        cls.book = os.path.dirname(os.environ.get("SANTHYA_MERGED") or "") or os.path.join(OCR_DIR, "santhya-vol-1")
        if not os.path.isdir(os.path.join(cls.book, "merged")):
            raise unittest.SkipTest("no merged Santhya under %s" % cls.book)
        from lib.writings_ocr import read_ocr_book
        cls.doc = read_ocr_book(cls.book, layout="auto", kind="commentary")
        cls.pages = {p["page"]: p for p in cls.doc["pages"]}

    def test_page_70_reads_in_bands(self):
        page = self.pages[70]
        self.assertEqual(page["layout"], "paired")
        recs = page["paragraphs"]
        # ਪ੍ਰਾਕਥਨ- ("preface"), however the engines spell its conjunct
        preface = next(r for r in recs if "ਕਥਨ-" in r["text"][:12] or "ਕ੍ਰਥਨ-" in r["text"][:12])
        self.assertEqual(preface["style"], "body")
        self.assertIn("ਵਰਣਨ", preface["text"])                     # the right half is on the same record
        first = next(r for r in recs if r.get("pair") == 1 and r["style"] == "body")
        self.assertEqual(first["explains"][0]["pair"], 1)
        self.assertGreaterEqual(sum(1 for r in recs if r["style"] == "quote" and r.get("pair")), 4)

    def test_page_30_pairs_by_alternation(self):
        recs = self.pages[30]["paragraphs"]
        self.assertEqual(self.pages[30]["layout"], "single")
        verse = next(r for r in recs if r["style"] == "quote" and r.get("pair"))
        after = [r for r in recs if r.get("pair") == verse["pair"] and r["style"] == "body"]
        self.assertTrue(after and all(r["explains"][0]["pair"] == verse["pair"] for r in after))

    def test_every_page_s_ang_stays_close_to_its_neighbours(self):
        seq = [(p["page"], p["ang_from"]) for p in self.doc["pages"] if p.get("ang_from")]
        jumps = [(a, b) for (_, a), (_, b) in zip(seq, seq[1:]) if not (-2 <= b - a <= 3)]
        self.assertEqual(jumps, [])
        self.assertTrue(all(p.get("ang_source") in ("header", "smoothed", "manifest", None) for p in self.doc["pages"]))


class OrphanTests(unittest.TestCase):
    def test_an_orphan_is_judged_in_its_own_column(self):
        from importlib import import_module
        m = import_module("22_ocr_merge")
        pivot = [{"bbox": [1300, 500, 2300, 550], "text": "ਸੱਜੇ ਪਾਸੇ ਦੀ ਸਤਰ"}]
        # the other engine read a line at the same height in the LEFT column,
        # and the same right-column line the pivot has
        other = {"e": [{"bbox": [100, 502, 900, 548], "text": "ਖੱਬੇ ਪਾਸੇ ਦੀ ਸਤਰ"},
                       {"bbox": [1310, 500, 2290, 550], "text": "ਸੱਜੇ ਪਾਸੇ ਦੀ ਸਤਰ"}]}
        got = m.adopt_orphans(pivot, other, 50)
        self.assertEqual([o["text"] for o in got], ["ਖੱਬੇ ਪਾਸੇ ਦੀ ਸਤਰ"])
        self.assertEqual(got[0]["adopted_from"], "e")


class EvalTests(unittest.TestCase):
    """24_ocr_eval.py: two merges side by side, and recovered lines scored apart."""

    def test_corrections_are_judged_by_the_truth(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        fixes = {"made": 0, "right": 0, "wrong": 0, "unsure": 0, "wrong_detail": []}
        hits = [{"corrections": [["ਪੁਸ਼ਾਦ", "ਪ੍ਰਸ਼ਾਦ", "confusion"],        # the truth has the fix: right
                                 ["ਲੱਗ", "ਲਗ", "confusion"],              # the truth has the reading: wrong
                                 ["ਕਖ", "ਗਖ", "confusion"]]}]              # neither: unsure
        ev.audit_corrections(fixes, "ਕੜਾਹ-ਪ੍ਰਸ਼ਾਦ ਛਕਣ ਲੱਗ ਪਏ।", hits, "b:1:1")
        self.assertEqual((fixes["made"], fixes["right"], fixes["wrong"], fixes["unsure"]), (3, 1, 1, 1))
        self.assertEqual(fixes["wrong_detail"][0]["was"], "ਲੱਗ")

    def test_every_merge_directory_is_an_engine(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        self.assertEqual(ev.engine_dir("b", "merged"), os.path.join("b", "merged"))
        self.assertEqual(ev.engine_dir("b", "merged-cov"), os.path.join("b", "merged-cov"))
        self.assertEqual(ev.engine_dir("b", "tesseract-pan"), os.path.join("b", "ocr", "tesseract-pan"))
        with tempfile.TemporaryDirectory() as d:
            for sub in ("ocr/tesseract-pan", "merged", "merged-cov", "gt"):
                os.makedirs(os.path.join(d, sub))
            self.assertEqual(ev.merged_dirs(d), ["merged", "merged-cov"])
            self.assertEqual(ev.engines_with_output(d), ["tesseract-pan", "merged", "merged-cov"])
            self.assertEqual(ev.engines_with_output(d, ["merged-cov"]), ["tesseract-pan", "merged-cov"])

    def test_a_recovered_line_is_scored_apart_and_a_not_text_crop_counts_a_false_recovery(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        from lib.ocr_engines import write_page
        with tempfile.TemporaryDirectory() as d:
            meta = {"page": 1, "page_w": 1000, "page_h": 1400}
            lines = [{"n": 1, "zone": "body", "bbox": [100, 300, 900, 340], "text": "ਇਹ ਪਹਿਲੀ ਸਤਰ ਹੈ", "kind": "commentary"},
                     {"n": 2, "zone": "body", "bbox": [100, 400, 900, 440], "text": "ਇਹ ਦੂਜੀ ਸਤਰ ਹੈ", "kind": "commentary",
                      "recovered": {"region": 1, "psm": 7, "engine": "tesseract-pan", "crop": [90, 390, 910, 450], "conf": 0.8}},
                     {"n": 3, "zone": "body", "bbox": [100, 600, 500, 640], "text": "॥ ॥", "kind": "commentary",
                      "recovered": {"region": 2, "psm": 7, "engine": "tesseract-pan", "crop": [90, 590, 510, 650], "conf": 0.3}}]
            write_page(os.path.join(d, "merged-x", "0001.jsonl"), meta, lines)
            gt = [{"id": "b:1:1", "page": 1, "bbox": [100, 300, 900, 340], "kind": "commentary", "text": "ਇਹ ਪਹਿਲੀ ਸਤਰ ਹੈ"},
                  {"id": "b:1:2", "page": 1, "bbox": [100, 400, 900, 440], "kind": "commentary", "text": "ਇਹ ਦੂਜੀ ਸਤਰ ਹੈ"},
                  {"id": "b:1:r600-100", "page": 1, "bbox": [100, 600, 500, 640], "kind": "commentary", "text": "",
                   "not_text": True},
                  {"id": "b:1:r800-100", "page": 1, "bbox": [100, 800, 500, 840], "kind": "commentary", "text": "",
                   "not_text": True}]
            r = ev.evaluate(d, "merged-x", gt, None, "pa")
        self.assertEqual(r["lines"], 2)                                 # the not-text rows are not scored
        self.assertEqual(r["by_provenance"]["page"]["n"], 1)
        self.assertEqual(r["by_provenance"]["recovered"]["n"], 1)
        self.assertEqual(r["by_provenance"]["recovered"]["word_acc"], 1.0)
        self.assertEqual(r["not_text"], {"n": 2, "false_recoveries": 1})
        self.assertTrue(next(s for s in r["lines_detail"] if s["id"] == "b:1:2")["recovered"])

    def test_a_coverage_row_scores_the_recovered_lines_apart_and_not_the_engine(self):
        # 23 --coverage-sample draws lines the layout missed; they judge the
        # coverage pass, not the engines that missed them
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "ocr", "e")); os.makedirs(os.path.join(d, "merged"))
            page = {"_meta": {"page": 1, "page_w": 2000, "page_h": 3000}}
            found = {"n": 1, "zone": "body", "bbox": [100, 100, 900, 150], "text": "ਇਹ ਸਤਰ ਪੜ੍ਹੀ ਗਈ", "words": []}
            missed = {"n": 2, "zone": "body", "bbox": [100, 300, 900, 350], "text": "ਇਹ ਸਤਰ ਛੁੱਟ ਗਈ ਸੀ", "words": [],
                      "recovered": {"region": 0, "psm": 13}}
            with open(os.path.join(d, "ocr", "e", "0001.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps(page) + "\n" + json.dumps(found, ensure_ascii=False) + "\n")
            with open(os.path.join(d, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps(page) + "\n" + json.dumps(found, ensure_ascii=False) + "\n" + json.dumps(missed, ensure_ascii=False) + "\n")
            gt = [{"id": "b:1:1", "page": 1, "bbox": [100, 100, 900, 150], "kind": "commentary", "text": "ਇਹ ਸਤਰ ਪੜ੍ਹੀ ਗਈ", "verified": True},
                  {"id": "b:1:c0", "page": 1, "bbox": [100, 300, 900, 350], "kind": "commentary", "text": "ਇਹ ਸਤਰ ਛੁੱਟ ਗਈ ਸੀ",
                   "verified": True, "origin": "coverage"},
                  {"id": "b:1:c1", "page": 1, "bbox": [100, 500, 900, 510], "kind": "commentary", "text": "", "verified": True,
                   "origin": "coverage", "not_text": True}]
            e = ev.evaluate(d, "e", gt, None, "pa")
            m = ev.evaluate(d, "merged", gt, None, "pa")
        self.assertEqual((e["lines"], e["word_acc"]), (1, 1.0))                 # the engine is judged on the sampled line only
        self.assertNotIn("recovered", e["by_provenance"])
        self.assertEqual((m["lines"], m["word_acc"]), (1, 1.0))
        self.assertEqual(m["by_provenance"]["recovered"], {"n": 1, "cer": 0.0, "word_acc": 1.0, "found": 1})
        self.assertEqual(m["not_text"], {"n": 1, "false_recoveries": 0})

    def test_a_danda_s_spacing_is_not_an_error(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        self.assertEqual(ev.canon("ਕੇਤੇ ਤੇਰੇ ਵਾਵਣਹਾਰੇ॥"), ev.canon("ਕੇਤੇ ਤੇਰੇ ਵਾਵਣਹਾਰੇ ॥"))
        self.assertEqual(ev.canon("ਧਾਰੇ॥ ਸੇਈ"), "ਧਾਰੇ ॥ ਸੇਈ")
        self.assertNotEqual(ev.canon("ਮਹਾ ਬਲ"), ev.canon("ਮਹਾਬਲ"))                 # a real difference stays one

    def test_a_row_folded_into_its_anchor_is_a_printed_line_and_a_half_line_is_a_hit(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "merged"))
            lines = [{"n": 1, "zone": "body", "bbox": [100, 100, 700, 150], "text": "ਕੀਮਤਿ ਕਿਨੈ ਨ ਪਾਈਐ ਰਿਦ ਮਾਣਕ ਮੋਲਿ ਅਮੋਲਿ ॥੧॥", "matches": [{}]},
                     {"n": 2, "zone": "body", "bbox": [100, 160, 700, 210], "text": "", "merged_into": 1},
                     {"n": 3, "zone": "body", "bbox": [100, 220, 700, 270], "text": "੧॥", "merged_into": 1}]
            with open(os.path.join(d, "merged", "0001.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_meta": {"page": 1, "page_w": 2000, "page_h": 3000}}) + "\n")
                for ln in lines:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            lc = ev.line_counts(d, "merged", [{"page": 1, "verified": True, "body_lines": 3}])
        self.assertEqual((lc["found"], lc["recall"]), (3, 1.0))
        # a truth row drawn across a two-column page: both halves the merge split it into are its reading
        truth = [100, 1000, 2000, 1050]
        halves = [{"n": 5, "bbox": [100, 1000, 700, 1050], "text": "ਸੁਖੁ ਘਰਿ ਲੈ ਜਾਇ॥"},
                  {"n": 6, "bbox": [800, 1000, 2000, 1050], "text": "ਬ੍ਰਹਮਾ ਆਦਿ ਸ਼ਕਤੇ ਤੇ ਪਾਰਬਤੀ"}]
        self.assertEqual(ev.reading(halves, truth), "ਸੁਖੁ ਘਰਿ ਲੈ ਜਾਇ॥ ਬ੍ਰਹਮਾ ਆਦਿ ਸ਼ਕਤੇ ਤੇ ਪਾਰਬਤੀ")

    def test_line_counts_and_the_coverage_summary(self):
        from importlib import import_module
        ev = import_module("24_ocr_eval")
        from lib.ocr_engines import write_page
        with tempfile.TemporaryDirectory() as d:
            cov = {"on": True, "psm": 7, "residual_share": 0.02, "recovered": 1, "capped": False, "uncovered_after": 1,
                   "rejected": {"tall": 1}, "ms": 30,
                   "regions": [{"bbox": [1, 1, 2, 2], "status": "recovered"}, {"bbox": [1, 1, 2, 2], "status": "rejected", "why": "tall"}]}
            meta = {"page": 1, "page_w": 1000, "page_h": 1400, "columns": [[0, 500], [500, 1000]], "coverage": cov}
            lines = [{"n": 1, "zone": "body", "col": 0, "bbox": [50, 300, 450, 340], "text": "ਪਹਿਲੀ"},
                     {"n": 2, "zone": "body", "col": 0, "bbox": [50, 400, 450, 440], "text": "ਦੂਜੀ"},
                     {"n": 3, "zone": "body", "col": 1, "bbox": [550, 300, 950, 340], "text": "ਤੀਜੀ"},
                     {"n": 4, "zone": "stamp", "bbox": [400, 1300, 600, 1330], "text": "Page 1 of 2", "dropped": True}]
            write_page(os.path.join(d, "merged-x", "0001.jsonl"), meta, lines)
            write_page(os.path.join(d, "merged-x", "0002.jsonl"), {"page": 2, "page_w": 1000, "page_h": 1400, "coverage": {"on": True, "regions": [], "recovered": 0, "rejected": {}, "uncovered_after": 0, "capped": False}}, [])
            counts = [{"page": 1, "layout": "two-column", "columns": 2, "body_lines": 5, "by_col": [4, 1], "verified": True},
                      {"page": 2, "layout": "single", "columns": 0, "body_lines": 3, "verified": True},
                      {"page": 3, "layout": "single", "columns": 0, "body_lines": None, "verified": False}]
            lc = ev.line_counts(d, "merged-x", counts)
            cs = ev.coverage_summary(d, "merged-x")
        # page 1: col 0 found 2 of 4, col 1 found 1 of 1; page 2: 0 of 3 -> recall 3/8, no excess
        self.assertEqual((lc["pages"], lc["true"], lc["found"], lc["recall"], lc["excess"]), (2, 8, 3, 0.375, 0.0))
        self.assertEqual((lc["two_column"]["pages"], lc["two_column"]["recall"]), (1, 0.6))
        self.assertEqual(cs, {"pages": 2, "pages_with_residual": 1, "regions": 2, "recovered": 1, "uncovered_after": 1,
                              "rejected": {"tall": 1}})
        self.assertIsNone(ev.coverage_summary(d, "tesseract-pan"))


class GroundTruthTests(unittest.TestCase):
    """23_ocr_gt.py: the two samples that see what the engine did not find."""


    def test_a_punjabi_book_is_drafted_from_the_tesseract_variant_it_has(self):
        from importlib import import_module
        gt = import_module("23_ocr_gt")
        with tempfile.TemporaryDirectory() as d:
            for e in ("tesseract-gurmukhi", "tesseract-pan"):
                os.makedirs(os.path.join(d, "ocr", e))
            self.assertEqual(gt.draft_engine(d, "tesseract"), "tesseract-pan")
            self.assertEqual(gt.draft_engine(d, "tesseract-gurmukhi"), "tesseract-gurmukhi")
    def _book(self, d):
        import cv2
        import numpy as np
        from lib.ocr_engines import write_page
        os.makedirs(os.path.join(d, "pages"))
        cv2.imwrite(os.path.join(d, "pages", "0001.png"), np.full((1400, 1000), 255, dtype=np.uint8))
        cv2.imwrite(os.path.join(d, "pages", "0002.png"), np.full((1400, 1000), 255, dtype=np.uint8))
        meta = {"language": "pa", "pages": [{"page": 1, "file": "0001.png", "h": 1400}, {"page": 2, "file": "0002.png", "h": 1400}]}
        json.dump(meta, open(os.path.join(d, "pages.json"), "w", encoding="utf-8"))
        regions = [{"bbox": [50, 300, 450, 340], "col": 0, "status": "recovered", "why": None},
                   {"bbox": [550, 300, 950, 340], "col": 1, "status": "recovered", "why": None},
                   {"bbox": [50, 700, 450, 900], "col": 0, "status": "rejected", "why": "tall"},
                   {"bbox": [550, 700, 950, 720], "col": 1, "status": "rejected", "why": "short"}]
        lines = [{"n": 1, "zone": "body", "col": 0, "bbox": [52, 302, 448, 338], "text": "ਮੁੜ ਪੜ੍ਹੀ ਸਤਰ", "kind": "commentary",
                  "recovered": {"region": 0, "psm": 7, "engine": "tesseract-pan", "crop": [30, 280, 470, 360], "conf": 0.8}},
                 {"n": 2, "zone": "body", "col": 1, "bbox": [552, 302, 948, 338], "text": "ਦੂਜੀ ਮੁੜ ਪੜ੍ਹੀ ॥", "kind": "gurbani-unmatched",
                  "recovered": {"region": 1, "psm": 7, "engine": "tesseract-pan", "crop": [530, 280, 970, 360], "conf": 0.7}},
                 {"n": 3, "zone": "body", "col": 0, "bbox": [50, 500, 450, 540], "text": "ਆਮ ਸਤਰ", "kind": "commentary"}]
        write_page(os.path.join(d, "merged-cov", "0001.jsonl"),
                   {"page": 1, "page_w": 1000, "page_h": 1400, "columns": [[0, 500], [500, 1000]],
                    "coverage": {"on": True, "regions": regions, "recovered": 2, "rejected": {"tall": 1, "short": 1},
                                 "uncovered_after": 1, "capped": False}}, lines)
        write_page(os.path.join(d, "merged-cov", "0002.jsonl"),
                   {"page": 2, "page_w": 1000, "page_h": 1400, "columns": [], "coverage": {"on": True, "regions": [], "recovered": 0, "rejected": {}, "uncovered_after": 0, "capped": False}},
                   [{"n": 1, "zone": "body", "bbox": [50, 500, 950, 540], "text": "ਇਕ ਕਾਲਮ"}])
        return meta

    def test_coverage_regions_and_page_counts_become_candidates_and_are_promoted(self):
        from importlib import import_module
        gt = import_module("23_ocr_gt")
        with tempfile.TemporaryDirectory() as d:
            meta = self._book(d)
            args = SimpleNamespace(book="b", coverage_sample=4, merged_name="merged-cov", seed=0, count_pages=2)
            gt.do_coverage_sample(args, d, meta)
            rows = gt.read_jsonl(os.path.join(d, "gt", "candidates.jsonl"))
            self.assertEqual(len(rows), 4)
            from collections import Counter
            self.assertEqual(Counter(r["status"] for r in rows), {"recovered": 2, "rejected": 2})
            self.assertTrue(all(r["origin"] == "coverage" and r["not_text"] is False for r in rows))
            recovered = [r for r in rows if r["status"] == "recovered"]
            self.assertEqual({r["draft"] for r in recovered}, {"ਮੁੜ ਪੜ੍ਹੀ ਸਤਰ", "ਦੂਜੀ ਮੁੜ ਪੜ੍ਹੀ ॥"})
            self.assertEqual({r["kind"] for r in recovered}, {"commentary", "gurbani"})
            self.assertTrue(all(os.path.exists(os.path.join(d, "gt", r["crop"])) for r in rows))
            gt.do_count_pages(args, d, meta)
            counts = gt.read_jsonl(os.path.join(d, "gt", "page-counts.candidates.jsonl"))
            self.assertEqual([c["page"] for c in counts], [1, 2])
            self.assertEqual(counts[0]["draft"], {"body": 3, "by_col": [2, 1]})
            self.assertTrue(os.path.exists(os.path.join(d, "gt", counts[0]["image"])))
            # the reviewer: one recovered line confirmed, one refusal was an ornament, one page counted
            for r in rows:
                if r["status"] == "recovered" and r["kind"] == "commentary":
                    r.update(verified=True, text="ਮੁੜ ਪੜ੍ਹੀ ਸਤਰ")
                elif r["status"] == "rejected" and r["why"] == "tall":
                    r.update(verified=True, not_text=True, text="")
            gt.write_jsonl(os.path.join(d, "gt", "candidates.jsonl"), rows)
            counts[0].update(verified=True, body_lines=5, by_col=[4, 1])
            gt.write_jsonl(os.path.join(d, "gt", "page-counts.candidates.jsonl"), counts)
            self.assertEqual(gt.do_check(d, meta), 0)
            gt.do_promote(d)
            lines = gt.read_jsonl(os.path.join(d, "gt", "lines.jsonl"))
            self.assertEqual(len(lines), 2)
            self.assertEqual([l["bbox"][1] for l in lines], [300, 700])            # by place on the page
            self.assertTrue(any(l.get("not_text") for l in lines))
            self.assertEqual(gt.read_jsonl(os.path.join(d, "gt", "page-counts.jsonl")),
                             [{"page": 1, "layout": "two-column", "columns": 2, "body_lines": 5, "by_col": [4, 1], "verified": True, "note": ""}])


if __name__ == "__main__":
    unittest.main()
