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

    def test_no_manifest_is_todays_behaviour(self):
        meta = parse_source("C:/x/1492003384127-Dhooja-Bhau-Part-2.pdf", None)
        self.assertEqual((meta["essay"], meta["part"], meta["work"], meta["reader"]), (127, 2, "dhooja-bhau", "pdf-text"))
        self.assertEqual(meta["kind"], "prose")

    def test_a_notation_book_carries_its_kind_and_style(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.pdf", "b.pdf", "c.pdf"):
                open(os.path.join(d, name), "wb").close()
            json.dump({"author": "X", "language": "pa", "reader": "ocr", "kind": "notation",
                       "style": {"table": "bars", "shabad_position": "after"},
                       "works": [{"file": "a.pdf", "title": "A", "book": "a"},
                                 {"file": "b.pdf", "title": "B", "book": "b", "style": {"table": "ruled", "labels": True}},
                                 {"file": "c.pdf", "title": "C", "book": "c", "kind": "prose"}]},
                      open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            m = load_manifest(d)
            a, b, c = (parse_source(os.path.join(d, n), m) for n in ("a.pdf", "b.pdf", "c.pdf"))
            self.assertEqual(a["kind"], "notation")
            # the folder's style under the defaults; a work's keys over the folder's
            self.assertEqual((a["style"]["table"], a["style"]["shabad_position"], a["style"]["swar_row"]), ("bars", "after", "above"))
            self.assertEqual((b["style"]["table"], b["style"]["labels"], b["style"]["shabad_position"]), ("ruled", True, "after"))
            # a prose work in a notation folder has no style at all
            self.assertEqual((c["kind"], c.get("style")), ("prose", None))
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
        # a prose book in the same folder still takes the prose route
        mixed = dict(manifest, kind="prose")
        mixed["works"] = manifest["works"] + [{"file": "p.pdf", "work": "prose", "book": "prose-1", "title": "P", "kind": "prose"}]
        mixed["works"][0] = dict(mixed["works"][0], kind="notation")
        both = self._plan(mixed)
        self.assertIn("ingest", [s for s, _ in both])
        self.assertIn("notation", [s for s, _ in both])

    def test_the_review_window_sits_at_the_middle_of_the_book(self):
        from lib.notation import mid_window
        self.assertEqual(mid_window(373, 7), (168, 174))
        self.assertEqual(mid_window(10, 5), (4, 8))
        self.assertEqual(mid_window(3, 5), (1, 3))

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
        self.assertNotIn("embed-en", [s for s, _ in steps])


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
        w = m.auto_weights(acc, keep)
        self.assertEqual(w["surya"], round(0.921 ** 4, 3))
        self.assertEqual(m.auto_weights({}, ["x"])["x"], 0.5)


class TranslationTests(unittest.TestCase):
    """26_translate_writings.py's checks and sentence handling, 28_translate_bench.py's sampling and scoring."""

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
            self.assertTrue(all(k in doc for k in ("path", "body_size", "pages")))


if __name__ == "__main__":
    unittest.main()
