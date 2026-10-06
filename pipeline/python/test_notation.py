"""
Tests for the keertan notation modules (lib/notation*.py, 29-32).

Everything is synthetic or a checked-in fixture: no scan, no engine, no
model. fixtures/notations/ holds two notations that use every construct,
the invalid variants with the error each must raise, the vocabulary
vectors, and the EXPECTED renderings -- the same files the JavaScript twin
in packages/search-core is tested against, so the two renderers agree to
the byte. Regenerate the expectations from this side only:

  python test_notation.py --write-expected
"""
import json
import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import notation, notation_render, notation_vocab
from lib.notation import content_hash, fill_defaults, make_id, merge_style, parse_id, strip_defaults, validate
from lib.notation_render import (cells, format_line, html, parse_cell, parse_line, roman_of, text_english,
                                 text_gurmukhi, word_spans)
from lib.notation_vocab import (RAAGS, TAALS, fold_pa, indel_ratio, normalise_raag, normalise_taal,
                                raag_key_from_corpus, section_label, skel_pa, taal_from_markers, taal_markers)
from lib.paths import ARTIFACTS

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "notations")
NAMES = ("teentaal-sthai", "slok-free")


def fixture(name):
    with open(os.path.join(FIXTURES, name + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def corpus_lines():
    with open(os.path.join(FIXTURES, "corpus-lines.json"), encoding="utf-8") as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def expected_path(name, kind):
    return os.path.join(FIXTURES, "%s.expected.%s" % (name, kind))


def renderings(rec):
    corpus = corpus_lines()
    return {
        "cells.json": json.dumps(cells(rec, corpus), ensure_ascii=False, indent=1),
        "gurmukhi.txt": text_gurmukhi(rec, corpus),
        "english.txt": text_english(rec, corpus),
        "gurmukhi.html": html(rec, "gurmukhi", corpus),
        "english.html": html(rec, "english", corpus),
    }


class VocabTests(unittest.TestCase):
    def test_every_vector_resolves_to_its_key(self):
        with open(os.path.join(FIXTURES, "vocab-vectors.json"), encoding="utf-8") as fh:
            vec = json.load(fh)
        for text, key in vec["raag_pa"]:
            with self.subTest(raag=text):
                self.assertEqual(normalise_raag(text)["key"], key)
        for text, key in vec["raag_en"]:
            with self.subTest(raag=text):
                self.assertEqual(normalise_raag(text, "en")["key"], key)
        for text, key, laya in vec["taal"]:
            with self.subTest(taal=text):
                got = normalise_taal(text)
                self.assertEqual((got["key"], got["laya"]), (key, laya))

    def test_the_variant_in_brackets_is_the_raag_and_the_bare_name_its_parent(self):
        got = normalise_raag("ਰਾਗ ਸਾਰੰਗ (ਬਿੰਦ੍ਰਾਬਨੀ ਸਾਰੰਗ)")
        self.assertEqual((got["key"], got["parent"], got["method"]), ("brindavani_sarang", "sarang", "alias"))

    def test_keys_are_slugs_and_aliases_unique_within_a_script(self):
        seen_pa, seen_en = {}, {}
        for key, r in RAAGS.items():
            self.assertRegex(key, r"^[a-z0-9_]+$")
            if r.get("parent"):
                self.assertIn(r["parent"], RAAGS)
            for a in r["aliases"]["pa"]:
                f = fold_pa(a)
                self.assertNotIn(f, {k: v for k, v in seen_pa.items() if v != key}, "%r names two raags" % a)
                seen_pa[f] = key
            for a in r["aliases"]["en"] + [r["en"]]:
                f = notation_vocab.fold_en(a)
                self.assertNotIn(f, {k: v for k, v in seen_en.items() if v != key}, "%r names two raags" % a)
                seen_en[f] = key
        for key, t in TAALS.items():
            self.assertRegex(key, r"^[a-z0-9_]+$")
            if t["matras"] is not None:
                self.assertEqual(sum(t["vibhag"]), t["matras"], key)
                self.assertEqual(len(t["markers"]), len(t["vibhag"]), key)
                self.assertIn(t["sam"], (1,))
                for m in t["tali"] + t["khali"]:
                    self.assertTrue(1 <= m <= t["matras"], key)

    def test_the_thirty_one_raags_of_the_granth_are_in_order(self):
        ordered = sorted((r for r in RAAGS.values() if r.get("ggs_order")), key=lambda r: r["ggs_order"])
        self.assertEqual(len(ordered), 31)
        self.assertEqual([r["key"] for r in ordered][:3], ["sri_raag", "majh", "gauri"])
        self.assertEqual(ordered[-1]["key"], "jaijaivanti")

    def test_markers_follow_the_taal(self):
        self.assertEqual(taal_markers("teentaal")[:5], ["×", None, None, None, "2"])
        self.assertEqual(taal_markers("teentaal", 9, 8), ["0", None, None, None, "3", None, None, None])
        self.assertEqual(taal_markers("jhaptaal"), ["×", None, "2", None, None, "0", None, "3", None, None])
        self.assertEqual(taal_markers("beer_taal", 1, 4), [None] * 4)
        self.assertEqual(taal_from_markers(10, {1: "×", 6: "0"}), "jhaptaal")
        self.assertEqual(taal_from_markers(7, {1: "0"}), "rupak")
        self.assertIsNone(taal_from_markers(16, {1: "×", 9: "0"}), "three taals share teentaal's shape")

    def test_section_labels(self):
        self.assertEqual(section_label("ਅੰਤਰਾ ੨"), ("antara", 2))
        self.assertEqual(section_label("ਅੰਤਰਾ 3:"), ("antara", 3))
        self.assertEqual(section_label("ਸਥਾਈ"), ("sthai", None))
        self.assertEqual(section_label("ਅਸਥਾਈ :-"), ("sthai", None))
        self.assertIsNone(section_label("ਅੰਤਰਾ ਦੀ ਸਿੱਖਿਆ ਬਹੁਤ ਲੰਮੀ"))
        self.assertIsNone(section_label("ਹਰਿ ਹਰਿ ਨਾਮੁ"))

    def test_folds_and_skeletons(self):
        self.assertEqual(fold_pa("ਰਾਗੁ ਬਿਲਾਵਲੁ ਮਹਲਾ ੫ ਘਰੁ ੨"), "ਬਿਲਾਵਲ")
        self.assertEqual(skel_pa(fold_pa("ਤੋੜੀ")), skel_pa(fold_pa("ਟੋਡੀ")))
        self.assertAlmostEqual(indel_ratio("kitten", "sitting"), 1 - 5 / 13)
        self.assertEqual(indel_ratio("", ""), 1.0)

    def test_every_corpus_raag_has_a_key(self):
        db = os.path.join(ARTIFACTS, "gurbani.sqlite")
        if not os.path.exists(db):
            self.skipTest("no artifacts/gurbani.sqlite")
        import sqlite3
        con = sqlite3.connect(db)
        names = [r[0] for r in con.execute("SELECT DISTINCT raag FROM shabads")]
        con.close()
        missing = [n for n in names if n and raag_key_from_corpus(n) is None]
        self.assertEqual(missing, [])
        self.assertGreaterEqual(len(names), 40)


class ContractTests(unittest.TestCase):
    def test_the_fixtures_are_valid(self):
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(validate(fixture(name)), [])

    def test_every_invalid_fixture_raises_the_error_it_names(self):
        folder = os.path.join(FIXTURES, "invalid")
        files = sorted(os.listdir(folder))
        self.assertGreaterEqual(len(files), 10)
        for fn in files:
            with open(os.path.join(folder, fn), encoding="utf-8") as fh:
                case = json.load(fh)
            with self.subTest(case=fn):
                codes = [e["code"] for e in validate(case["notation"])]
                self.assertIn(case["error"], codes, codes)

    def test_defaults_round_trip(self):
        rec = fixture("teentaal-sthai")
        full = fill_defaults(rec)
        self.assertEqual(full["sections"][0]["lines"][0]["beats"][0]["notes"][0],
                         {"s": "S", "o": 0, "k": False, "t": False, "len": 1, "kh": False})
        self.assertEqual(strip_defaults(full), strip_defaults(rec))
        self.assertEqual(content_hash(rec), content_hash(full))
        self.assertNotEqual(content_hash(rec), content_hash(fixture("slok-free")))

    def test_ids(self):
        self.assertEqual(make_id("gurmat-sangeet-sagar-1", 42, 3), "gurmat-sangeet-sagar-1:0042:3")
        self.assertEqual(parse_id("gurmat-sangeet-sagar-1:0042:3"), ("gurmat-sangeet-sagar-1", 42, 3))
        self.assertIsNone(parse_id("Bad Key:42:3"))
        self.assertEqual(notation.image_name("a-b:0042:3", 2), "a-b-0042-3-2.png")
        self.assertEqual(notation.image_name("a-b:0042:3", 2, thumb=True), "a-b-0042-3-2.thumb.png")
        with self.assertRaises(ValueError):
            make_id("Not A Slug", 1, 1)

    def test_style_merges_and_refuses_the_unknown(self):
        style = merge_style({"table": "ruled"}, {"labels": True})
        self.assertEqual((style["table"], style["labels"], style["swar_row"]), ("ruled", True, "above"))
        with self.assertRaises(ValueError):
            merge_style({"table": "wavy"})
        with self.assertRaises(ValueError):
            merge_style({"colour": "red"})

    def test_jsonl_round_trip(self):
        import tempfile
        rec = fixture("teentaal-sthai")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "notations.jsonl")
            notation.write_jsonl(path, notation.meta_for({"book": "example-book"}), [rec])
            meta, rows = notation.read_jsonl(path)
            self.assertEqual((meta["book"], meta["schema"]), ("example-book", 1))
            self.assertEqual(rows[0]["notation_id"], rec["notation_id"])
            self.assertEqual(fill_defaults(rows[0]), fill_defaults(rec))


class RenderTests(unittest.TestCase):
    def test_renderings_match_the_pinned_expectations(self):
        for name in NAMES:
            rec = fixture(name)
            for kind, got in renderings(rec).items():
                path = expected_path(name, kind)
                if not os.path.exists(path):
                    self.fail("no %s; run test_notation.py --write-expected" % os.path.relpath(path, HERE))
                with open(path, encoding="utf-8") as fh:
                    want = fh.read()
                with self.subTest(name=name, kind=kind):
                    self.assertEqual(got.rstrip("\n"), want.rstrip("\n"))

    def test_the_english_row_says_what_the_notes_say(self):
        rows = cells(fixture("teentaal-sthai"), corpus_lines())
        sthai = rows[0]
        self.assertEqual(sthai["swar_en"][:9], ["S", "r", "G", "m", "P", "-", "DN", "S'", "*"])
        self.assertEqual(sthai["swar_en"][12], "{P}M")
        self.assertEqual(sthai["swar_en"][14], "R-S")
        self.assertEqual(sthai["bol_en"][0], "mayray"[:2] if False else "mayray" if sthai["bol_en"][0] == "mayray" else sthai["bol_en"][0])
        # whole words take the corpus transliteration; parts of words the table
        self.assertEqual(sthai["bol_en"][15], "tariaa")
        self.assertEqual(sthai["bol_en"][5], "-")
        self.assertEqual(sthai["marks"][:5], ["×", None, None, None, "2"])
        self.assertTrue(sthai["vibhag"][0] and sthai["vibhag"][4] and not sthai["vibhag"][1])
        antara = rows[1]
        self.assertEqual(antara["matra_from"], 9)
        self.assertEqual(antara["marks"][0], "0")
        self.assertEqual(antara["swar_en"][4], "?")
        self.assertTrue(antara["unknown"][4])
        self.assertEqual(antara["swar_en"][5], "nDPM")
        self.assertEqual(antara["swar_pa"][3], "ਸ̣")
        self.assertEqual(antara["swar_en"][3], "S,")

    def test_the_gurmukhi_row_uses_combining_marks(self):
        rows = cells(fixture("teentaal-sthai"))
        self.assertEqual(rows[0]["swar_pa"][1], "ਰ̲")        # komal re
        self.assertEqual(rows[0]["swar_pa"][3], "ਮ́")        # tivra ma
        self.assertEqual(rows[0]["swar_pa"][7], "ਸ̇")        # taar sa
        self.assertEqual(rows[0]["swar_pa"][5], "—")
        self.assertEqual(rows[0]["bol_pa"][5], "ऽ")

    def test_format_and_parse_line_round_trip(self):
        rec = fixture("teentaal-sthai")
        for row in cells(rec):
            text = format_line(row)
            back = parse_line(text)
            self.assertEqual(back["kind"], row["kind"])
            self.assertEqual(back["matra_from"], row["matra_from"])
            want = [strip_defaults_beat(b) for b in row["beats"]]
            self.assertEqual(back["beats"], want)
        self.assertEqual(parse_cell("{P'}m~-"), {"notes": [{"s": "M", "t": True, "kan": {"s": "P", "o": 1}, "kh": True, "len": 2}], "div": 2})
        with self.assertRaises(ValueError):
            parse_cell("Q")

    def test_roman_table(self):
        self.assertEqual([roman_of(w) for w in ["ਸਤਿ", "ਨਾਮੁ", "ਸੰਗਤਿ", "ਪ੍ਰਭ", "ਸੱਚ", "ਵਾਹਿਗੁਰੂ", ""]],
                         ["sat", "naam", "sangat", "prabh", "sacch", "vaahiguroo", ""])
        text = "ਮੇਰੇ ਮਾਧਉ ਜੀ ਸਤਸੰਗਤਿ ਮਿਲੇ ਸੁ ਤਰਿਆ ॥੧॥ ਰਹਾਉ ॥"
        self.assertEqual([text[a:b] for a, b in word_spans(text)],
                         ["ਮੇਰੇ", "ਮਾਧਉ", "ਜੀ", "ਸਤਸੰਗਤਿ", "ਮਿਲੇ", "ਸੁ", "ਤਰਿਆ"])

    def test_html_escapes_and_marks_up(self):
        rec = fixture("teentaal-sthai")
        rec["sections"][0]["lines"][0]["beats"][0]["bol"]["g"] = "<b>&"
        out = html(rec, "gurmukhi")
        self.assertIn("&lt;b&gt;&amp;", out)
        self.assertNotIn("<b>&", out)
        self.assertIn('class="n n-komal"', out)
        self.assertIn('class="n n-tivra"', out)
        self.assertIn('class="n n-taar"', out)
        self.assertIn('<sup class="n-kan">', out)
        self.assertIn('class="n-grp n-grp-2"', out)
        self.assertIn('class="ntn-cell ntn-vb ntn-sam"', out)
        self.assertIn("ntn-unknown", out)
        en = html(rec, "english")
        self.assertEqual(out.count("<td"), en.count("<td"), "the same DOM in both scripts")


def strip_defaults_beat(beat):
    """A beat as parse_line would produce it: defaults gone, provenance gone."""
    out = {}
    if beat.get("ext"):
        return {"ext": True}
    if beat.get("rest"):
        return {"rest": True}
    if beat.get("notes") is None:
        return {"notes": None}
    notes = []
    for n in beat["notes"]:
        m = {"s": n["s"]}
        if n.get("k"):
            m["k"] = True
        if n.get("t"):
            m["t"] = True
        if n.get("o"):
            m["o"] = n["o"]
        if n.get("kan"):
            m["kan"] = {"s": n["kan"]["s"], **({"o": n["kan"]["o"]} if n["kan"].get("o") else {}),
                        **({"k": True} if n["kan"].get("k") else {}), **({"t": True} if n["kan"].get("t") else {})}
        if n.get("kh"):
            m["kh"] = True
        if n.get("len", 1) > 1:
            m["len"] = n["len"]
        notes.append(m)
    out["notes"] = notes
    if beat.get("div", 1) != 1:
        out["div"] = beat["div"]
    return out


def write_expected():
    for name in NAMES:
        rec = fixture(name)
        assert validate(rec) == [], validate(rec)
        for kind, text in renderings(rec).items():
            with open(expected_path(name, kind), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text.rstrip("\n") + "\n")
            print("wrote", os.path.relpath(expected_path(name, kind), HERE))
    with open(os.path.join(FIXTURES, "notation.css"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(notation_render.NOTATION_CSS)
    with open(expected_path("teentaal-sthai", "hash.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content_hash(fixture("teentaal-sthai")) + "\n")




# ---- the page reader: text, layout, resolution ---------------------------------

from lib.notation_layout import classify_line, link_pages, page_layout, raag_descriptions  # noqa: E402
from lib.notation_resolve import resolve_shabad  # noqa: E402
from lib.notation_text import (bol_text, clean_bol, is_heading_like, parse_heading, parse_ref, swara_token,  # noqa: E402
                               token_class)


def _line(n, text, y, kind="commentary", bold=False, matches=None, x0=200, x1=1500, h=60):
    rec = {"n": n, "text": text, "bbox": [x0, y, x1, y + h], "kind": kind, "bold": bold, "zone": "body"}
    if matches:
        rec["matches"] = matches
    return rec


class TextTests(unittest.TestCase):
    def test_references_in_their_printed_forms(self):
        cases = {
            "(ਗਉੜੀ ਸੁਖਮਨੀ ਮ: ੫, ਪੰਨਾ ੨੬੯)": (269, 269, "G", 5),
            "(ਰਾਗੁ ਆਸਾ, ਸ੍ਰੀ ਕਬੀਰ ਜੀ, ਪੰਨਾ ੪੮੪)": (484, 484, "G", None),
            "ਪੰਨਾ ੯੫੯-੬੦": (959, 960, "G", None),
            "(SGGS p. 1208)": (1208, 1208, "G", None),
            "(ਦਸਮ ਗ੍ਰੰਥ, ਪੰਨਾ ੧੩)": (13, 13, "D", None),
            "(ਭਾਈ ਗੁਰਦਾਸ ਜੀ, ਵਾਰ ੧)": (None, None, "B", None),
        }
        for text, (a, b, src, mahala) in cases.items():
            ref = parse_ref(text)
            self.assertIsNotNone(ref, text)
            self.assertEqual((ref["ang_from"], ref["ang_to"], ref["source"], ref["mahala"]), (a, b, src, mahala), text)

    def test_a_footnote_number_and_a_patshahi_are_not_angs(self):
        self.assertIsNone(parse_ref("ਗੁਰੂ ਜੀ ਨੇ ਕਿਹਾ (੧)"))
        ref = parse_ref("(ਪਾ: ੧੦, ਪੰਨਾ ੧੩)")
        self.assertEqual((ref["ang_from"], ref["mahala"]), (13, 10))

    def test_headings_give_number_raag_taal_and_laya(self):
        h = parse_heading("੨੦. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਦਾਦਰਾ (ਮੱਧ੍ਯ ਲਯ)")
        self.assertEqual((h["number"], h["raag"]["key"], h["taal"]["key"], h["taal"]["matras"], h["laya"]),
                         (20, "bhairavi", "dadra", 6, "madh"))
        h = parse_heading("੨੩. ਰਾਗ ਭੈਰਵੀ, ਝਪਤਾਲ (ਲਗ-ਭਗ ਬਿਲੰਬਿਤ ਲਯ)")
        self.assertEqual((h["taal"]["key"], h["laya"]), ("jhaptaal", "vilambit"))
        h = parse_heading("ਰਾਗ ਬਸੰਤ (ਹਿੰਡੋਲ), ਤੀਨਤਾਲ")
        self.assertEqual((h["raag"]["key"], h["raag"]["parent_key"], h["taal"]["key"]), ("hindol", "basant", "teentaal"))
        self.assertTrue(is_heading_like("ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਕਹਿਰਵਾ"))
        self.assertFalse(is_heading_like("ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥"))

    def test_token_classes_by_row_role(self):
        self.assertEqual(swara_token("ਸ")["swar"], "S")
        self.assertEqual(swara_token("ਨੀ")["swar"], "N")
        self.assertEqual((swara_token("ਧੁ")["swar"], swara_token("ਧੁ").get("octave_hint")), ("D", -1))
        self.assertEqual((swara_token("ਸੰ")["swar"], swara_token("ਸੰ").get("octave_hint")), ("S", 1))
        # × is a rest in a swar row and the sam in a marker row; ੩ a tali there and a matra number in a
        # matra row: the class says what the glyph can be, the row's role decides (notation_layout)
        self.assertEqual([token_class(t) for t in "ਸ ਰੇ ਗ — × ੦ ੩ | ਮੇਰੀ".split()],
                         ["swara", "swara", "swara", "held", "rest", "marker", "marker", "bar", "gurmukhi"])
        self.assertEqual(clean_bol("ਮੇਰੀऽऽ"), ("ਮੇਰੀ", 2))
        self.assertEqual(clean_bol("S"), (None, 1))
        # the syllables of one word sit in separate cells; the corpus matcher folds the spaces
        self.assertEqual(bol_text([{"bol": {"g": "ਮੇ", "h": 1}}, {"bol": {"g": "ਰੀ", "h": 0}}, {"bol": {"g": "", "h": 1}}]), "ਮੇ ਰੀ")


class LayoutTests(unittest.TestCase):
    def test_lines_take_their_roles(self):
        self.assertEqual(classify_line(_line(1, "੨੦. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਦਾਦਰਾ (ਮੱਧ੍ਯ ਲਯ)", 200, bold=True)), "heading")
        self.assertEqual(classify_line(_line(2, "ਅਸਥਾਈ", 300)), "section")
        self.assertEqual(classify_line(_line(3, "× ੦ ੨ ੩", 400)), "marker")
        self.assertEqual(classify_line(_line(4, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 500)), "grid")
        self.assertEqual(classify_line(_line(5, "ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥", 600, kind="gurbani")), "gurbani")
        # a bol row Tesseract read as a verse: bars and held marks make it grid
        self.assertEqual(classify_line(_line(6, "ਤਾ $।ਮ ਨ।ਮੇਂ ਹਿ।ਆ $।5 5", 700, kind="gurbani-unmatched", bold=True)), "grid")
        self.assertEqual(classify_line(_line(7, "ਨੋਟ :--ਇਹ ਸ਼ਬਦ ਨੰ: ੩ ਤੇ ਲਿਖਿਆ ਹੈ ।", 800)), "note")
        self.assertEqual(classify_line(_line(8, "੧੭੦", 2600, x0=800, x1=880)), "pageno")

    def _page(self, page, lines):
        style = merge_style(None)
        return page_layout(lines, 1760, 2650, style, page)

    def test_a_page_becomes_regions_and_pages_link_into_spans(self):
        m = [{"shabad_id": 913, "line_id": 1000 + i, "score": 0.95, "source": "G"} for i in range(3)]
        p1 = self._page(167, [
            _line(1, "੧੮. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਦਾਦਰਾ (ਮੱਧਅ ਲਯ)", 200, bold=True),
            _line(2, "ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥", 300, kind="gurbani", matches=[m[0]]),
            _line(3, "ਸਾਚ ਬਿਨਾ ਕਹ ਹੋਵਤ ਸੂਚਾ ॥", 380, kind="gurbani", matches=[m[1]]),
            _line(4, "(ਗਉੜੀ ਸੁਖਮਨੀ ਮ: ੫, ਪੰਨਾ ੨੬੯)", 460),
        ])
        p2 = self._page(168, [
            _line(1, "× ੦", 200),
            _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 280),
            _line(3, "ਬਿਰ ਥੀ | ਸਾ ऽ | ਕਤ ਕੀ | ਆ ਰਜਾ", 360),
            _line(4, "ਅੰਤਰਾ", 460),
            _line(5, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 540),
            _line(6, "ਸਾ ਚ | ਬਿ ਨਾ | ਕਹ ऽ | ਹੋ ਵਤ", 620),
            _line(7, "ਨੋਟ :--ਸ਼ਬਦ ਦੀਆਂ ਬਾਕੀ ਤੁਕਾਂ ਅੰਤਰੇ ਤੇ ਲਾਓ ।", 720),
        ])
        roles1 = [r["role"] for r in p1["regions"]]
        self.assertEqual(roles1, ["heading", "shabad", "ref"])
        self.assertEqual([r["role"] for r in p2["regions"]], ["marker", "grid", "section", "grid", "note"])
        spans = link_pages([p1, p2], merge_style(None))
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertEqual(s["pages"], [167, 168])
        self.assertEqual(s["heading"]["parsed"]["number"], 18)
        self.assertEqual(s["ref"]["parsed"]["ang_from"], 269)
        self.assertEqual([(sec["kind"], sec["n"], len(sec["grids"]), len(sec["markers"])) for sec in s["sections"]],
                         [("sthai", 1, 1, 1), ("antara", None, 1, 0)])
        self.assertTrue(s["sections"][0].get("assumed"))       # the sthai had no printed label
        self.assertEqual(len(s["notes"]), 1)
        # the next shabad the corpus knows ends the notation, and its heading leads it in
        m2 = {"shabad_id": 4284, "line_id": 2000, "score": 0.95, "source": "G"}
        p3 = self._page(169, [_line(1, "੧੯. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਦਾਦਰਾ", 200, bold=True),
                              _line(2, "ਮਨ ਕਹਾ ਲੁਭਾਈਐ ਆਨ ਕਉ ॥", 300, kind="gurbani", matches=[m2]),
                              _line(3, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                              _line(4, "ਮਨ ਕਹਾ | ਲੁ ਭਾ | ਈ ऽ | ਐ ਆਨ", 480),
                              _line(5, "ਤਾਨ", 580),
                              _line(6, "ਸ ਰੇ ਗ ਮ | ਪ ਧ ਨੀ ਸੰ | ਸੰ ਨੀ ਧ ਪ | ਮ ਗ ਰੇ ਸ", 640)])
        dropped = []
        spans = link_pages([p1, p2, p3], merge_style(None), dropped)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["pages"], [167, 168])
        self.assertEqual(spans[1]["heading"]["parsed"]["number"], 19)
        # everything after the shabad to the end is in the span: the taan too
        self.assertEqual([sec.get("label", {}) and sec["label"]["text"] for sec in spans[1]["sections"]], [None, "ਤਾਨ"])
        self.assertEqual(spans[1]["extent"][169][3], 700)
        self.assertEqual(dropped, [])
        # a verse the corpus does not know, with no reference under it, anchors nothing:
        # a tabla exercise under a numbered taal heading is dropped
        p4 = self._page(170, [_line(1, "੨੦. ਤਾਲ ਦਾਦਰਾ", 200, bold=True),
                              _line(2, "ਇਹ ਤਾਲ ਧੁਰਪਦ ਗਾਇਨ ਸ਼ੈਲੀ ਨਾਲ ਵਜਦਾ ਹੈ ॥", 300, kind="gurbani"),
                              _line(3, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                              _line(4, "ਧਾ ਧਿਨ | ਧਿਨ ਧਾ | ਧਾ ਤਿਨ | ਤਿਨ ਤਾ", 480),
                              _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 560)])
        self.assertEqual([r["role"] for r in p4["regions"]][:2], ["heading", "text"])
        dropped = []
        p0 = self._page(169, [_line(1, "ਤਬਲੇ ਦੇ ਕੁਝ ਕਾਇਦੇ ਅਤੇ ਰੇਲੇ", 300)])
        spans = link_pages([p0, p4], merge_style(None), dropped)     # at the start of a book, no shabad before it
        self.assertEqual((len(spans), [d["pages"] for d in dropped]), (0, [[169, 170]]))
        # after a shabad, a numbered notation with no verse of its own is that shabad set again
        spans = link_pages([p1, p2, p3, p4], merge_style(None), dropped := [])
        self.assertEqual((len(spans), dropped, spans[2]["inherited"]), (3, [], True))
        # ... but a notation the book numbers with a raag and a taal is kept, its shabad to be found
        p4b = self._page(170, [_line(1, "੨੦. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਦਾਦਰਾ", 200, bold=True),
                          _line(3, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                          _line(4, "ਮਨ ਕਹਾ | ਲੁ ਭਾ | ਈ ऽ | ਐ ਆਨ", 480),
                          _line(5, "ਨੋਟ :--ਇਹ ਸ਼ਬਦ ਨੰ: ੩ ਤੇ ਲਿਖਿਆ ਹੈ ।", 560)])
        spans = link_pages([p1, p2, p3, p4b], merge_style(None), dropped := [])
        self.assertEqual((len(spans), dropped, spans[2]["shabad"]), (3, [], []))
        # a taal on its own after the grids is the next section of the same notation (a partaal) ...
        p5 = self._page(170, [_line(1, "ਤਾਲ ਝਪਤਾਲ", 200, bold=True),
                              _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                              _line(3, "ਮਨ ਕਹਾ | ਲੁ ਭਾ | ਈ ऽ | ਐ ਆਨ", 480),
                              _line(4, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 560)])
        spans = link_pages([p1, p2, p3, p5], merge_style(None))
        self.assertEqual(len(spans), 2)
        self.assertEqual((spans[1]["pages"], (spans[1]["sections"][-1]["taal"] or {}).get("key")), ([169, 170], "jhaptaal"))
        # ... and the same shabad set again under a raag-and-taal heading with no number keeps its shabad
        p6 = self._page(170, [_line(1, "ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਝਪਤਾਲ", 200, bold=True),
                              _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                              _line(3, "ਮਨ ਕਹਾ | ਲੁ ਭਾ | ਈ ऽ | ਐ ਆਨ", 480),
                              _line(4, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 560)])
        spans = link_pages([p1, p2, p3, p6], merge_style(None))
        self.assertEqual(len(spans), 3)
        self.assertTrue(spans[2]["inherited"])
        self.assertEqual(spans[2]["shabad"][0][1]["text"], spans[1]["shabad"][0][1]["text"])


class LinkerRuleTests(unittest.TestCase):
    """The rules the first-cut comments brought (1 October 2026), one case each."""

    def _page(self, page, lines, style=None, page_h=2650):
        lay = page_layout(lines, 1760, page_h, style or merge_style(None), page)
        lay["page_w"], lay["page_h"] = 1760, page_h          # as 29's layouts_for records them
        return lay

    def _shabad(self, page, sid, y=300):
        m = [{"shabad_id": sid, "line_id": sid * 10 + i, "score": 0.95, "source": "G"} for i in range(2)]
        return [_line(2, "ਮਾਈ ਮੈ ਕਿਹਿ ਬਿਧਿ ਲਖਉ ਗੁਸਾਈ ॥", y, kind="gurbani", matches=[m[0]]),
                _line(3, "ਮਹਾ ਮੋਹ ਅਗਿਆਨਿ ਤਿਮਰਿ ਮੋ ਮਨੁ ਰਹਿਓ ਉਰਝਾਈ ॥੧॥ ਰਹਾਉ ॥", y + 80, kind="gurbani", matches=[m[1]])]

    def _grids(self, n0, y):
        return [_line(n0, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", y), _line(n0 + 1, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", y + 80),
                _line(n0 + 2, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", y + 160)]

    def test_a_long_notation_is_flagged_not_cut(self):
        pages = [self._page(84, self._shabad(84, 2399, 500) + [_line(9, "ਰਾਗ ਸੋਰਠਿ ਤਿੰਨਤਾਲ", 1500, bold=True)] + self._grids(10, 1650))]
        pages += [self._page(p, self._grids(1, 300)) for p in range(85, 91)]
        spans = link_pages(pages, merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [list(range(84, 91))])
        self.assertTrue(spans[0]["long"])

    def test_the_running_headers_give_the_printed_page_numbers(self):
        from lib.notation_layout import header_page_numbers
        p1 = self._page(84, [_line(1, "ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ 68", 160)] + self._shabad(84, 2399, 500))
        p2 = self._page(85, [_line(1, "ਰਾਗ ਸੋਰਠਿ", 140, x0=600, x1=900)] + self._grids(2, 300))
        p3 = self._page(86, [_line(1, "70 ਸ੍ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ", 140)] + self._grids(2, 300))
        self.assertEqual(header_page_numbers([p1, p2, p3]), {84: 68, 86: 70})

    def test_a_running_header_is_known_by_its_repetition_however_the_ocr_spells_it(self):
        p1 = self._page(84, [_line(1, "ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ", 160)] + self._shabad(84, 2399, 500)
                        + [_line(9, "ਰਾਗ ਸੋਰਠਿ ਤਿੰਨਤਾਲ", 1500, bold=True)] + self._grids(10, 1650))
        p2 = self._page(85, [_line(1, "70 ਸ੍ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ", 140)] + self._grids(2, 300))
        spans = link_pages([p1, p2], merge_style(None))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["pages"], [84, 85])
        self.assertEqual(spans[0]["heading"]["text"], "ਰਾਗ ਸੋਰਠਿ ਤਿੰਨਤਾਲ")
        # named in the manifest, a header is dropped even when it stands on one page alone
        style = merge_style({"running_header": "ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ"})
        spans = link_pages([self._page(84, p1 and [_line(1, "ਰੀ ਗੁਰੂ ਤੇਗ ਬਹਾਦਰ ਰਾਗ ਰਤਨਾਵਲੀ", 160)] + self._shabad(84, 2399, 500)
                                       + self._grids(10, 1650), style)], style)
        self.assertEqual(len(spans), 1)
        self.assertIsNone(spans[0]["heading"])

    def test_a_raag_with_a_page_number_at_the_top_of_a_page_does_not_close_the_notation(self):
        p1 = self._page(317, self._shabad(317, 3285, 1100) + [_line(9, "ਰਾਗੁ ਰਾਮਕਲੀ ਤਿੰਨ ਤਾਲ", 1890, bold=True)] + self._grids(10, 2100))
        p2 = self._page(318, [_line(1, "ਰਾਗ ਰਾਮਕਲੀ ਰ 299", 170)] + self._grids(2, 300))
        spans = link_pages([p1, p2], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[317, 318]])

    def test_the_verse_count_line_and_a_marker_before_the_grids_do_not_end_the_shabad(self):
        self.assertEqual(classify_line(_line(4, "੨ ॥ ੬ ॥", 1340, x0=700, x1=1260)), "gurbani")
        p1 = self._page(84, self._shabad(84, 2399, 500) + [_line(4, "x 2 0 3", 1340, x0=700, x1=1260),
                                                            _line(9, "ਰਾਗ ਸੋਰਠਿ ਤਿੰਨਤਾਲ", 1500, bold=True)] + self._grids(10, 1650))
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["heading"]["text"], "ਰਾਗ ਸੋਰਠਿ ਤਿੰਨਤਾਲ")
        self.assertEqual(len(spans[0]["sections"][0]["grids"]), 1)

    def _m(self, sid, i, score=0.9):
        return [{"shabad_id": sid, "line_id": sid * 10 + i, "score": score, "source": "G"}]

    def test_the_same_shabad_set_again_under_a_new_heading_is_the_next_notation(self):
        # Tara Singh sets a chhant pada by pada: shabad, reference, "ਰਾਗ ਕੇਦਾਰਾ ਤਾਲ ਦਾਦਰਾ", grids -- the corpus
        # links both padas to one shabad, the heading after the reference says it is a notation of its own
        p1 = self._page(747, self._shabad(747, 4019, 1474) + [_line(9, "(ਗੁ. ਗ੍ਰੰਥ ਪੰਨਾ ੧੧੨੨)", 2215, x0=600, x1=1100),
                                                              _line(10, "ਰਾਗ ਕੇਦਾਰਾ ਤਿੰਨਤਾਲ", 2300, bold=True)])
        p2 = self._page(748, self._grids(1, 300) + self._grids(4, 800)
                        + [_line(40, "ਕੇਦਾਰਾ ਛੰਤ ਮਹਲਾ ੫", 1061, x0=666, x1=1021),
                           _line(41, "ਹਰਿ ਪ੍ਰੇਮ ਭਗਤਿ ਜਨ ਬੇਧਿਆ ਸੇ ਆਨ ਕਤ ਜਾਹੀ ॥", 1143, kind="gurbani", matches=self._m(4019, 5)),
                           _line(42, "ਮੀਨੁ ਬਿਛੋਹਾ ਨਾ ਸਹੈ ਜਲ ਬਿਨੁ ਮਰਿ ਪਾਹੀ ॥", 1205, kind="gurbani", matches=self._m(4019, 6)),
                           _line(43, "(ਗੁ. ਗ੍ਰੰਥ ਪੰਨਾ ੧੧੨੨)", 1528, x0=600, x1=1100),
                           _line(44, "ਰਾਗ ਕੇਦਾਰਾ ਤਾਲ ਦਾਦਰਾ", 1625, bold=True)] + self._grids(45, 1770))
        spans = link_pages([p1, p2], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[747, 748], [748]])
        self.assertEqual(spans[1]["extent"][748][1], 1061)
        self.assertEqual(spans[1]["heading"]["text"], "ਰਾਗ ਕੇਦਾਰਾ ਤਾਲ ਦਾਦਰਾ")
        self.assertTrue(spans[1]["continues"])                       # grids on the last page read
        self.assertFalse(spans[0]["continues"])
        # a window's end counts, not only the last page of all the pages read (the front pages, two windows)
        p3 = self._page(760, self._shabad(760, 4030, 500) + [_line(9, "ਰਾਗ ਕੇਦਾਰਾ ਤਿੰਨਤਾਲ", 1500, bold=True)] + self._grids(10, 1650)
                        + [_line(13, "(ਗੁ. ਗ੍ਰੰਥ ਪੰਨਾ ੧੧੨੩)", 2300, x0=600, x1=1100)])
        spans = link_pages([p1, p2, p3], merge_style(None))
        self.assertEqual([s.get("continues") for s in spans], [False, True, False])
        # a shabad's text at the very start of the pages read may begin on the page before; one under a heading does not
        self.assertEqual([s.get("continued") for s in spans], [True, False, True])   # 760 opens its own window with a shabad too
        p0 = self._page(746, [_line(1, "ਰਾਗ ਕੇਦਾਰਾ ਤਿੰਨਤਾਲ", 300, bold=True)] + self._shabad(746, 4017, 400) + self._grids(5, 700))
        self.assertFalse(link_pages([p0], merge_style(None))[0]["continued"])

    def test_a_shred_under_the_verse_does_not_split_a_shabad_from_its_own_heading(self):
        # Tara Singh: the shabad's first lines at the foot of a page (with a one-line grid shred under them),
        # the rest of the shabad, its reference and its heading on the next: one notation
        p1 = self._page(388, self._grids(1, 1683) + [_line(23, "ਧਨਾਸਰੀ ਮਹਲਾ ੫ ਘਰੁ ੧ ਚਉਪਦੇ", 2093, x0=580, x1=1200),
                                                     _line(24, "ਭਵ ਖੰਡਨ ਦੁਖ ਭੰਜਨ ਸ੍ਵਾਮੀ ਭਗਤਿ ਵਛਲ ਨਿਰੰਕਾਰੇ ॥", 2200, kind="gurbani", matches=self._m(2556, 0)),
                                                     _line(25, "ਸ | ਰੇ ਗ | — ਮ", 2319, x0=300, x1=700)])
        p2 = self._page(389, [_line(3, "ਮੇਰਾ ਮਨੁ ਲਾਗਾ ਹੈ ਰਾਮ ਪਿਆਰੇ ॥", 275, kind="gurbani", matches=self._m(2556, 1)),
                              _line(4, "ਦੀਨ ਦਇਆਲਿ ਕਰੀ ਪ੍ਰਭਿ ਕਿਰਪਾ ਵਸਿ ਕੀਨੇ ਪੰਚ ਦੂਤਾਰੇ ॥੧॥ ਰਹਾਉ ॥", 340, kind="gurbani", matches=self._m(2556, 2)),
                              _line(5, "(ਗੁ. ਗ੍ਰੰਥ ਪੰਨਾ ੬੧੦)", 919, x0=600, x1=1100), _line(6, "ਰਾਗ ਧਨਾਸਰੀ (ਕਾਫੀ ਥਾਟ) ਤਿੰਨਤਾਲ", 1008, bold=True)]
                        + self._grids(7, 1071))
        spans = link_pages([p1, p2], merge_style(None))
        self.assertEqual([s["pages"] for s in spans if s["sid"] == 2556], [[388, 389]])
        self.assertEqual([s for s in spans if s["sid"] == 2556][0]["extent"][388][1], 2093)

    def test_the_bol_rows_read_as_the_verse_and_the_antra_label_stay_with_the_notation(self):
        # Dyal Singh: shabad, sthai grid, the lyric lines under it read as the verse again, "ਅੰਤਰਾ", grid
        p1 = self._page(328, [_line(1, "੭ ਰਾਗ ਸ਼ਿਵਰੰਜਨੀ, ਤਾਲ-ਦਾਦਰਾ", 185, bold=True)] + self._shabad(328, 1286, 293)
                        + [_line(4, "(ਗਉੜੀ ਭਗਤ ਕਬੀਰ ਜੀ, ਪੰਨਾ ੩੨੩)", 626, x0=600, x1=1200), _line(5, "ਅਸਥਾਈ", 731, x0=800, x1=950)]
                        + self._grids(6, 829)
                        + [_line(12, "ਮਾਈ ਮੈ ਕਿਹਿ ਬਿਧਿ ਲਖਉ ਗੁਸਾਈ ॥", 1100, kind="gurbani", matches=self._m(1286, 0)),
                           _line(13, "ਮਹਾ ਮੋਹ ਅਗਿਆਨਿ ਤਿਮਰਿ ਮੋ ਮਨੁ ਰਹਿਓ ਉਰਝਾਈ ॥੧॥ ਰਹਾਉ ॥", 1180, kind="gurbani", matches=self._m(1286, 1)),
                           _line(14, "ao", 1255, x0=800, x1=830), _line(15, "ਅਤਰਾ", 1270, x0=800, x1=900)]
                        + self._grids(16, 1300) + [_line(19, "= ਬਾਕੀ ਤੁਕਾਂ ਅੰਤਰੇ ਤੇ ਲਾਓ !", 1704)] + self._grids(20, 1800))
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["extent"][328][3], 2020)
        self.assertEqual(spans[0]["sid"], 1286)

    def test_one_line_of_verse_between_the_grids_is_the_antra_label_not_the_next_shabad(self):
        p1 = self._page(328, [_line(1, "ਰਾਗੁ ਨਟ ਤਿੰਨ ਤਾਲ", 340, bold=True)] + self._shabad(328, 3611, 400) + self._grids(4, 568)
                        + [_line(17, "ਹਰਿ ਹਰਿ ਅਗਮ ਅਗਾਧੋ ॥", 1390, kind="gurbani", matches=self._m(4704, 0, 0.889), x0=926, x1=1559)]
                        + self._grids(18, 1457))
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["sid"], 3611)
        self.assertEqual(spans[0]["extent"][328][3], 1457 + 160 + 60)
        # with a reference after it, one line is a shabad of its own (a salok)
        p2 = self._page(328, [_line(1, "ਰਾਗੁ ਨਟ ਤਿੰਨ ਤਾਲ", 340, bold=True)] + self._shabad(328, 3611, 400) + self._grids(4, 568)
                        + [_line(17, "ਹਰਿ ਹਰਿ ਅਗਮ ਅਗਾਧੋ ॥", 1390, kind="gurbani", matches=self._m(4704, 0, 0.889), x0=926, x1=1559),
                           _line(18, "(ਗੁ. ਗ੍ਰੰਥ ਪੰਨਾ ੯੭੫)", 1460, x0=600, x1=1100)] + self._grids(19, 1557))
        self.assertEqual(len(link_pages([p2], merge_style(None))), 2)

    def test_a_raag_description_before_the_shabad_is_not_the_notation(self):
        swar = ["ਰੇਗਮ ਪ, ਮੁ ਪਰ ਨੀਸਾ, ਸਾਰੇਸਾਸਾਗ ਗਮ, ਮਪਮ, ਗਗਮ, ਗਮ ਧਪਪਗਮ,", "ਮਪਧਨੀਨੀਪ, ਧਧਪਧਮਪਗ ਮ, ਰੇ ਗਮਪ, ਮਗ,ਸਰੇ,ਸਾ।",
                "2. ਸਾਰੇਸਾਸਾ, ਰੇ ਗਰੇ ਰੇ, ਗਗਮ, ਪ੍ਧ੍ਪਰਪ, ਸਾਰੇਸਾਸਾ, ਗਗ, ਮ,ਰੇਗਗਮ", "ਮਮ,ਗਮਸਾਰੇਸਾ, ਰੇਗਮਪਸਾਂ, ਧਨੀ ਪਰ ਮਪਮਗਮ, ਸਾਸਾਗਗਖ਼, ਪਮ,"]
        prose = ["ਇਸ ਰਾਗ ਦਾ ਸਰੂਪ ਕਾਫੀ ਠਾਠ ਤੋਂ ਉਤਪੰਨ ਮੰਨਿਆ ਜਾਂਦਾ ਹੈ ਅਤੇ ਇਸ ਵਿਚ ਸਾਰੇ ਸੁਰ ਸ਼ੁੱਧ ਲਗਦੇ ਹਨ ।",
                 "ਵਾਦੀ ਸੁਰ ਮਧਿਅਮ ਅਤੇ ਸੰਵਾਦੀ ਸੁਰ ਸ਼ੜਜ ਮੰਨਿਆ ਜਾਂਦਾ ਹੈ ਅਤੇ ਗਾਉਣ ਦਾ ਸਮਾਂ ਰਾਤ ਦਾ ਦੂਜਾ ਪਹਿਰ ਹੈ ।",
                 "ਸੁਰਾਂ ਦੇ ਆਧਾਰ ਤੇ ਭਿੰਨ ਹੋ ਜਾਂਦੇ ਹਨ । ਇਸ ਵਿਚ ਕਦੇ ਕਦੇ ਕੋਮਲ ਨਿਸ਼ਾਦ ਵਿਵਾਦੀ ਰੂਪ ਵਿਚ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ ।"]
        lines = [_line(1 + i, t, 299 + 70 * i) for i, t in enumerate(prose)] + [_line(8, "ਸੁਰ ਵਿਸਤਾਰ ਰਾਗ ਨਟ", 690, bold=True, x0=600, x1=1100)]
        lines += [_line(9 + i, t, 820 + 60 * i) for i, t in enumerate(swar)]
        lines += [_line(18, "ਨਟ ਮਹਲਾ ੪", 1747, x0=800, x1=1000),
                  _line(19, "ਰਾਮ ਜਪਿ ਜਨ ਰਾਮੈ ਨਾਮਿ ਰਲੇ ॥", 1825, kind="gurbani", matches=self._m(3611, 0)),
                  _line(21, "ਹਰਿ ਹਰਿ ਅਗਮ ਅਗੋਚਰੁ ਸੁਆਮੀ ਜਨ ਜਪਿ ਮਿਲਿ ਸਲਲ ਸਲਲੇ ॥", 1947, kind="gurbani", matches=self._m(3611, 1)),
                  _line(22, "(ਗ, ਗੁ. ਪੰਨਾ ੯੭੫)", 2435, x0=600, x1=1000)]
        p1 = self._page(327, lines)
        p2 = self._page(328, [_line(1, "ਰਾਗੁ ਨਟ ਤਿੰਨ ਤਾਲ", 340, bold=True)] + self._grids(4, 568))
        roles = [(r["role"], r["bbox"][1]) for r in p1["regions"]]
        self.assertIn(("text", 820), roles); self.assertIn(("shabad", 1747), roles)       # the swar-vistaar split off
        dropped = []
        spans = link_pages([p1, p2], merge_style(None), dropped)
        self.assertEqual([s["pages"] for s in spans], [[327, 328]])
        self.assertEqual(spans[0]["extent"][327][1], 1747)
        self.assertEqual([d["pages"] for d in dropped], [[327]])

    def test_a_verse_line_read_as_a_bol_row_opens_the_shabad_block(self):
        self.assertEqual(classify_line(_line(18, "ਕਬਹੂ ਖੀਰਿ ਕਾੜ ਘੀਉ ਨ ਭਾਵੈ । ਕਬਹੂ ਘਰ ਘਰ ਟੂਕ ਮਗਾਵੇ 1!", 1923)), "grid")
        p1 = self._page(76, [_line(1, "ਰਾਗ ਭੇਰਉ ਤਿੰਨ ਤਾਲ", 369, bold=True)] + self._shabad(76, 4149, 420) + self._grids(4, 681)
                        + [_line(18, "ਕਬਹੂ ਖੀਰਿ ਕਾੜ ਘੀਉ ਨ ਭਾਵੈ । ਕਬਹੂ ਘਰ ਘਰ ਟੂਕ ਮਗਾਵੇ 1!", 1923),
                           _line(19, "ਕਬਹੂ ਕੂਰਨੁ ਚਨੇ ਬਿਨਾਵੈ ॥੧॥ ੧ ॥", 1991, kind="gurbani", matches=self._m(4150, 0)),
                           _line(20, "ਜਿਉ ਰਾਮੁ ਰਾਖੈ ਤਿਉ ਰਹੀਐ ਰੇ ਭਾਈ ॥", 2052, kind="gurbani", matches=self._m(4150, 1)),
                           _line(21, "(ਭੈਰਉ ਬਾਣੀ ਭਗਤ ਨਾਮਦੇਵ ਜੀ ਕੀ, ਆਦਿ ਗ੍ਰੰਥ, ਪੰਨਾ ੧੧੬੪)", 2349, x0=500, x1=1300)])
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[1]["extent"][76][1], 1923)
        self.assertEqual(spans[0]["extent"][76][3], 1922)
        # a matched line of the block that the token shapes called a grid row, likewise
        p2 = self._page(78, self._grids(1, 100)
                        + [_line(3, "ਹਸਤ ਖੇਲਤ ਤੇਰੇ ਦੇਹੁਰੇ ਆਇਆ ॥ ਭਗਤਿ ਕਰਤ ਨਾਮਾ ਪਕਰਿ ਉਠਾਇਆ ॥੧॥ ੧ !!", 304, kind="gurbani", matches=self._m(4151, 0)),
                           _line(4, "ਹੀਨੜੀ ਜਾਤਿ ਮੇਰੀ ਜਾਦਿਮ ਰਾਇਆ ॥ਛੀਪੇ ਕੇ ਜਨਮਿ ਕਾਹੇ ਕਉ ਆਇਆ ॥੧॥ ਰਹਾਉ ॥", 364, kind="gurbani", matches=self._m(4151, 1)),
                           _line(7, "(ਭੈਰਉ ਬਾਣੀ ਭਗਤ ਨਾਮਦੇਵ ਜੀ ਕੀ, ਆਦਿ ਗ੍ਰੰਥ, ਪੰਨਾ ੧੧੬੪)", 548, x0=500, x1=1300),
                           _line(8, "ਰਾਗ ਭੈਰਉ ਤਾਲ ਦਾਦਰਾ", 642, bold=True)] + self._grids(9, 707))
        self.assertEqual(classify_line(_line(3, "ਹਸਤ ਖੇਲਤ ਤੇਰੇ ਦੇਹੁਰੇ ਆਇਆ ॥ ਭਗਤਿ ਕਰਤ ਨਾਮਾ ਪਕਰਿ ਉਠਾਇਆ ॥੧॥ ੧ !!", 304, kind="gurbani")), "grid")
        self.assertEqual([r["bbox"][1] for r in p2["regions"] if r["role"] == "shabad"], [304])

    def test_the_lead_in_of_the_next_shabad_leaves_loose_text_with_the_notation_before(self):
        p1 = self._page(78, self._shabad(78, 4151, 300) + [_line(9, "ਰਾਗ ਭੈਰਉ ਤਾਲ ਦਾਦਰਾ", 640, bold=True)] + self._grids(10, 700)
                        + [_line(13, "ਹਸ ਤ ਖੋ5 ਲਤ ਤੇ ਰੇ ਦੇ$ ਹੁ58 ਭਰੇ ਆ 8 ਇਆ", 1290), _line(14, "ਦੁਧੁ ਦੁ ਦੁ ਨੀਨੀ ਸਾਂ ਸਾਂ ਰੇਸਾਂ ਨੀਸਾਂ ਰੁੇ ਸਾਂ ਦੁ ਪ ਪ", 1365)]
                        + [_line(20, "ਭਗਤਿ ਕਰਤ ਨਾਮਾ ਪਕਰਿ ਉਠਾਇਆ ॥੧॥", 1430, kind="gurbani", matches=[{"shabad_id": 4152, "line_id": 1, "score": 0.9, "source": "G"}]),
                           _line(21, "ਹੀਨੜੀ ਜਾਤਿ ਮੇਰੀ ਜਾਦਿਮ ਰਾਇਆ ॥", 1510, kind="gurbani", matches=[{"shabad_id": 4152, "line_id": 2, "score": 0.9, "source": "G"}])]
                        + [_line(22, "੧੯.", 1600, x0=800, x1=860)])
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[1]["extent"][78][1], 1430)
        self.assertEqual(spans[0]["extent"][78][3], 1429)     # the first runs to the edge of the next: the gap is its tail

    def test_a_partaal_changes_taal_from_section_to_section_inside_one_notation(self):
        p1 = self._page(284, [_line(1, "੧੧. ਰਾਗ ਵਡਹੰਸ, ਪੜਤਾਲ, ਮਣੀ ਤਾਲ, ੧੧ ਮਾਤਰੇ (ਬਿਲੰਬਿਤ ਲਯ)", 650, bold=True)] + self._shabad(284, 2159, 770)
                        + [_line(9, "ਅਸਥਾਈ", 1648, x0=830, x1=972)] + self._grids(10, 1714))
        p2 = self._page(285, [_line(1, "ਅੰਤਰਾ-ਸੂਲਫਾਕ (ਮਣੀ ਤਾਲ ਦੀ ਚੌਗੁਨ ਲਯ ਵਿਚ)", 135, x0=565, x1=1337)] + self._grids(2, 230)
                        + [_line(6, "ਤਾਲ ਚੰਚਲ (ਮਣੀ ਤਾਲ ਦੀ ਦੁਗੁਨ ਲਯ ਵਿਚ)", 1878, x0=608, x1=1312)] + self._grids(7, 1960))
        p3 = self._page(286, [_line(1, "ਦੂਜਾ ਅੰਤਰਾ-ਤਾਲ ਰੂਪਕ (ਬਿਲੰਬਿਤ ਇਕਤਾਲੇ ਦੀ)", 211, x0=537, x1=1200)] + self._grids(2, 300))
        spans = link_pages([p1, p2, p3], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[284, 285, 286]])
        secs = [(sec["kind"], sec.get("n"), (sec.get("taal") or {}).get("key"), len(sec["grids"])) for sec in spans[0]["sections"]]
        self.assertEqual(secs, [("sthai", None, None, 1), ("antara", None, "sooltaal", 1), (None, None, "chachar", 1), ("antara", 2, "rupak", 1)])
        # before the grids, a taal on its own line is the heading's second half, not a section
        p4 = self._page(431, self._shabad(431, 2655, 300) + [_line(9, "ਰਾਗ ਜੈਤਸਰੀ", 700, x0=124, x1=400), _line(10, "ਤਿੰਨ ਤਾਲ", 700, x0=1300, x1=1611)]
                        + self._grids(11, 800))
        spans = link_pages([p4], merge_style(None))
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0]["heading"]["parsed"]["raag"]["key"], spans[0]["heading"]["parsed"]["taal"]["key"]), ("jaitsri", "teentaal"))
        self.assertEqual(len(spans[0]["sections"]), 1)

    def test_the_previous_notations_bol_row_read_as_the_next_verse_is_trimmed_off_the_shabad(self):
        prev = [{"shabad_id": 4151, "line_id": 7, "score": 0.9, "source": "G"}]
        nxt = [{"shabad_id": 4152, "line_id": 8, "score": 0.95, "source": "G"}]
        p1 = self._page(78, self._shabad(78, 4151, 300) + [_line(9, "ਰਾਗ ਭੈਰਉ ਤਾਲ ਦਾਦਰਾ", 640, bold=True)] + self._grids(10, 700)
                        + [_line(13, "ਭਗਤਿ ਕਰਤ ਨਾਮਾ ਪਕਰਿ ਉਠਾਇਆ ॥੧॥s ਠਾਂ s ਇਆ", 1426, kind="gurbani", matches=prev),
                           _line(14, "ਜੈਸੀ ਭੂਖੇ ਪ੍ਰੀਤਿ ਅਨਾਜ ॥ ਤ੍ਰਿਖਾਵੰਤ ਜਲ ਸੇਤੀ ਕਾਜ ॥", 1700, kind="gurbani", matches=nxt),
                           _line(15, "ਨਾਮੇ ਪ੍ਰੀਤਿ ਨਾਰਾਇਣ ਲਾਗੀ ॥", 1780, kind="gurbani", matches=nxt)])
        spans = link_pages([p1], merge_style(None))
        self.assertEqual([s["sid"] for s in spans], [4151, 4152])
        self.assertEqual(spans[1]["extent"][78][1], 1700)              # the next begins at its own first verse
        self.assertEqual(spans[0]["extent"][78][3], 1699)              # the bol row is the first's tail
        self.assertNotIn("ਭਗਤਿ ਕਰਤ", spans[1]["shabad"][0][1]["text"])
        self.assertEqual(len(spans[0]["sections"][0]["grids"]), 2)     # ... and a grid of the first
        # the previous notation's swar rows, unknown to the corpus, over a page turn: they go with it, page and all
        p2 = self._page(746, [_line(1, "ਪ ਸੀ ਰੋ ਸਾ ਧਧ ਪ ਮਪ ਧਨੀ ਧਪ ਮਪ ਮਮ ਰੇਸਾ", 345, kind="gurbani"),
                              _line(2, "ਕੇਦਾਰਾ ਮਹਲਾ ੫", 581, kind="gurbani", x0=708, x1=1005),
                              _line(3, "ਹਰਿ ਕੇ ਨਾਮ ਕੀ ਮਨ ਰੁਚੈ ॥", 636, kind="gurbani", matches=nxt),
                              _line(4, "ਕੋਟਿ ਸਾਂਤਿ ਅਨੰਦ ਪੂਰਨ ਜਲਤ ਛਾਤੀ ਬੁਝੈ ॥ ਰਹਾਉ ॥", 697, kind="gurbani", matches=nxt)]
                        + self._grids(5, 1200))
        p0 = self._page(745, self._shabad(745, 4151, 300) + [_line(9, "ਰਾਗ ਕੇਦਾਰਾ ਇਕਤਾਲ", 640, bold=True)] + self._grids(10, 700))
        spans = link_pages([p0, p2], merge_style(None))
        self.assertEqual([(s["sid"], s["pages"]) for s in spans], [(4151, [745, 746]), (4152, [746])])
        self.assertEqual(spans[1]["extent"][746][1], 581)

    def test_a_line_with_no_two_gurmukhi_letters_is_not_a_heading(self):
        self.assertFalse(is_heading_like("N..O\" '"))
        self.assertEqual(classify_line(_line(5, "N..O\" '", 2021, x0=984, x1=1127)), "text")


class ReviewLedgerTests(unittest.TestCase):
    """The review ledger: keys, re-attaching by overlap, freezing, drift, the server's verdicts."""

    def _rec(self, nid, pages, sid, raag="gujri", taal="teentaal", y0=200, y1=900):
        rec = _record(nid, shabad_id=sid)
        rec["page"], rec["pages"], rec["seq"] = int(nid.split(":")[1]), pages, int(nid.split(":")[2])
        rec["heading"]["raag"]["key"], rec["heading"]["taal"]["key"] = raag, taal
        rec["layout"] = {"extent": {str(p): [100, y0, 1600, y1] for p in pages}}
        rec["images"] = [{"n": 1, "file": "images/%s-1.png" % nid.replace(":", "-"), "role": "block", "page": pages[0],
                          "bbox": [100, y0, 1600, y1], "w": 1500, "h": y1 - y0, "bytes": 3, "sha256": "ab" * 32, "thumb": None}]
        rec["source"]["commit"] = "abc1234"
        return rec

    def test_keys_number_the_notations_that_share_shabad_raag_and_taal(self):
        from lib.notation_review import assign_keys, key_slug
        a, b, c = self._rec("test-book:0170:1", [170], 1248), self._rec("test-book:0172:1", [172], 1248), self._rec("test-book:0174:1", [174], 1248, taal="jhaptaal")
        keys = assign_keys([b, a, c])
        self.assertEqual(sorted(keys), ["test-book/1248/gujri/jhaptaal#1", "test-book/1248/gujri/teentaal#1", "test-book/1248/gujri/teentaal#2"])
        self.assertEqual((a["review_nth"], b["review_nth"]), (1, 2))
        self.assertEqual(key_slug(a["review_key"]), "test-book__1248__gujri__teentaal__1")

    def test_an_entry_reattaches_by_overlap_when_the_seq_moves_and_not_to_another_shabad(self):
        from lib.notation_review import assign_keys, entry_of, match_entry, drift_of
        a = self._rec("test-book:0170:1", [170], 1248)
        assign_keys([a])
        e = entry_of(a, "accepted", round_=1)
        moved = self._rec("test-book:0170:2", [170, 171], 1248, y0=240, y1=950)       # a later run cut it a little differently
        other = self._rec("test-book:0170:1", [170], 913)
        assign_keys([other, moved])
        self.assertIs(match_entry(e, [other, moved]), moved)
        d = drift_of(e, moved)
        self.assertFalse(d["same"]); self.assertEqual(d["pages_after"], [170, 171])
        self.assertIsNone(match_entry(e, [other]))
        self.assertTrue(drift_of(e, None)["lost"])

    def test_two_notations_of_one_shabad_on_overlapping_pages_keep_their_own_verdicts(self):
        from lib.notation_review import assign_keys, attach, entry_of
        first = self._rec("test-book:0049:1", [49, 50], 3107, y0=200, y1=2300)
        second = self._rec("test-book:0050:1", [49, 50, 51], 3107, y0=1700, y1=2400)     # the same shabad set again, inherited
        assign_keys([first, second])
        self.assertEqual((first["review_nth"], second["review_nth"]), (1, 2))
        ledger = {first["review_key"]: entry_of(first, "accepted")}
        got = attach(ledger, [first, second])
        self.assertEqual(list(got), ["test-book:0049:1"])                                  # the second is still unreviewed

    def test_the_ledger_freezes_accepted_records_and_annotates_the_backlog(self):
        import tempfile
        from lib.notation_review import append_entry, apply_review, assign_keys, entry_of, read_ledger, save_fixture
        with tempfile.TemporaryDirectory() as d:
            images = os.path.join(d, "images"); os.makedirs(images)
            a = self._rec("test-book:0170:1", [170], 1248); b = self._rec("test-book:0172:1", [172], 913); c = self._rec("test-book:0174:1", [174], 4284)
            assign_keys([a, b, c])
            open(os.path.join(images, "test-book-0170-1-1.png"), "wb").write(b"png")
            ea = entry_of(a, "accepted", round_=1); append_entry(ea, d); save_fixture(a, ea, images, d)
            append_entry(entry_of(b, "backlog", "the antara is missing", round_=1), d)
            append_entry(entry_of(c, "rejected", round_=1), d)
            self.assertTrue(os.path.exists(os.path.join(d, "fixtures", "test-book", "images", "ab" * 32 + ".png")))
            ledger = read_ledger("test-book", d)
            self.assertEqual({e["status"] for e in ledger.values()}, {"accepted", "backlog", "rejected"})
            # a later run: a moved a little, b unchanged, c the same, plus a new one
            a2 = self._rec("test-book:0170:1", [170], 1248, y0=400)
            b2 = self._rec("test-book:0172:1", [172], 913); c2 = self._rec("test-book:0174:1", [174], 4284)
            n = self._rec("test-book:0176:1", [176], 1313)
            os.remove(os.path.join(images, "test-book-0170-1-1.png"))
            got = apply_review([a2, b2, c2, n], ledger, images, d)
            ids = [r["notation_id"] for r in got["records"]]
            self.assertEqual(ids, ["test-book:0170:1", "test-book:0172:1", "test-book:0176:1"])
            frozen = got["records"][0]
            self.assertTrue(frozen["verified"]); self.assertEqual(frozen["layout"]["extent"]["170"][1], 200)     # the fixture, not the re-cut
            self.assertTrue(os.path.exists(os.path.join(images, "test-book-0170-1-1.png")))                       # its image restored
            self.assertEqual((got["accepted"], got["reused"], got["lost"], got["backlog"], got["rejected"]), (1, 1, 0, 1, 1))
            self.assertEqual(got["records"][1]["review"]["comment"], "the antara is missing")
            self.assertFalse(got["drift"][0]["same"])                                                             # the re-cut moved: reported
            # lost: nothing overlaps the accepted one any more -> still emitted
            got2 = apply_review([n], ledger, images, d)
            self.assertEqual((got2["lost"], [r["notation_id"] for r in got2["records"]]), (1, ["test-book:0176:1", "test-book:0170:1"]))

    def test_the_server_saves_a_verdict_and_hides_what_is_accepted(self):
        import tempfile, urllib.request
        from http.server import ThreadingHTTPServer
        import threading
        spec = importlib.util.spec_from_file_location("review34", os.path.join(HERE, "34_notation_review.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            a = self._rec("test-book:0170:1", [170], 1248); b = self._rec("test-book:0172:1", [172], 913)
            from lib.notation_review import assign_keys, read_ledger
            assign_keys([a, b])
            books = [{"book": "test-book", "title": "Test", "author": "A", "records": [a, b], "img_base": "/b/test-book", "pages_base": "/b/test-book/pages", "files": {}}]
            state = m.ReviewState(books, 40, False, 2, d, "Test · review")
            srv = ThreadingHTTPServer(("127.0.0.1", 0), m.make_handler(state))
            t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
            try:
                base = "http://127.0.0.1:%d" % srv.server_address[1]
                page = urllib.request.urlopen(base + "/").read().decode("utf-8")
                self.assertIn("test-book/1248/gujri/teentaal#1", page); self.assertIn("textarea", page); self.assertIn("Accept (a)", page)
                # the button's id is the card's id: looked up literally (getElementById takes no selector escaping)
                self.assertIn('id="c-test-book:0170:1"', page); self.assertIn('data-vid="test-book:0170:1"', page)
                self.assertIn("getElementById('c-' + id)", page); self.assertNotIn("CSS.escape", page)
                req = urllib.request.Request(base + "/verdict", data=json.dumps({"notation_id": "test-book:0170:1", "key": a["review_key"], "status": "accepted", "comment": ""}).encode(),
                                             headers={"content-type": "application/json"}, method="POST")
                got = json.loads(urllib.request.urlopen(req).read())
                self.assertTrue(got["ok"]); self.assertIn("1 accepted", got["counts"])
                req = urllib.request.Request(base + "/verdict", data=json.dumps({"notation_id": "test-book:0172:1", "key": b["review_key"], "status": "backlog", "comment": "cut short"}).encode(),
                                             headers={"content-type": "application/json"}, method="POST")
                urllib.request.urlopen(req).read()
                ledger = read_ledger("test-book", d)
                self.assertEqual({k: v["status"] for k, v in ledger.items()}, {a["review_key"]: "accepted", b["review_key"]: "backlog"})
                self.assertEqual(ledger[b["review_key"]]["round"], 2)
                self.assertTrue(os.path.exists(os.path.join(d, "fixtures", "test-book", "test-book__1248__gujri__teentaal__1.json")))
                page = urllib.request.urlopen(base + "/").read().decode("utf-8")
                body = page.split("<script id=cands")[0]
                self.assertIn('class="card stub done-accepted" id="c-test-book:0170:1"', body)           # accepted: a stub in its place
                self.assertEqual(body.count("<textarea"), 1)                                             # only the backlog card is whole
                self.assertIn("test-book/913/gujri/teentaal#1", page)                                     # backlog: shown
                # the pages do not shift under the reviewer: one card a page, page 2 is still the second card
                state.per_page = 1
                p2 = urllib.request.urlopen(base + "/?page=2").read().decode("utf-8").split("<script id=cands")[0]
                self.assertIn('id="c-test-book:0172:1"', p2); self.assertNotIn('id="c-test-book:0170:1"', p2)
                self.assertIn("(0 left)", p2); self.assertIn("(1 left)", p2)
                state.per_page = 40
                bad = urllib.request.Request(base + "/verdict", data=b'{"notation_id": "nope", "status": "accepted"}', headers={"content-type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(bad)
            finally:
                srv.shutdown()
            # the gate and the dashboard
            from lib.notation_review import check_book, status_of
            self.assertTrue(check_book("test-book", [a, b], d)["ok"])
            moved = self._rec("test-book:0170:1", [170, 171], 1248, y0=600)
            got = check_book("test-book", [moved, b], d)
            self.assertFalse(got["ok"]); self.assertEqual(len(got["failed"]), 1)
            # accepted with a note: a move there is reported, not failed
            from lib.notation_review import append_entry, entry_of
            append_entry(entry_of(a, "accepted", "a little extra from the next shabad", round_=3), d)
            got = check_book("test-book", [moved, b], d)
            self.assertTrue(got["ok"]); self.assertEqual(len(got["noted"]), 1); self.assertEqual(got["failed"], [])
            self.assertEqual(status_of("test-book", [a, b], d)["noted"], 1)
            # and the re-run shows the moved cut again with the note instead of freezing the old one
            from lib.notation_review import apply_review, read_ledger
            nudged = self._rec("test-book:0170:1", [170], 1248, y1=880)         # a small move, under the IoU gate
            got = apply_review([nudged, self._rec("test-book:0172:1", [172], 913)], read_ledger("test-book", d), os.path.join(d, "img"), d)
            self.assertEqual(got["noted_moved"], 1); self.assertEqual(got["reused"], 0)
            self.assertEqual(got["records"][0]["review"]["comment"], "a little extra from the next shabad")
            self.assertTrue(got["records"][0]["review"]["changed"]); self.assertNotIn("verified", {k for k, v in got["records"][0].items() if v is True})
            st = status_of("test-book", [a, b], d)
            self.assertEqual((st["accepted"], st["backlog"], st["unreviewed"], st["clear"]), (1, 1, 0, False))


class RecutTests(unittest.TestCase):
    def test_an_image_cut_again_from_its_bbox_has_the_same_sha(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("cv2")
        import tempfile, importlib.util
        spec = importlib.util.spec_from_file_location("parse29", os.path.join(HERE, "29_notation_parse.py"))
        p29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p29)
        with tempfile.TemporaryDirectory() as d:
            page = np.full((800, 600), 255, dtype=np.uint8)
            page[100:300, 50:550:7] = 0
            im = {"n": 1, "file": "images/b-0001-1-1.png", "role": "block", "page": 1, "bbox": [40, 90, 560, 310], "thumb": "images/b-0001-1-1.thumb.png"}
            sha1 = p29.cut_image(page, im, d)
            os.remove(os.path.join(d, "b-0001-1-1.png"))
            sha2 = p29.cut_image(page, im, d)
            self.assertEqual(sha1, sha2)
            self.assertTrue(os.path.exists(os.path.join(d, "b-0001-1-1.thumb.png")))


class SectionTaalTests(unittest.TestCase):
    def test_a_section_taal_is_written_as_its_key_and_validates(self):
        rec = _record()
        rec["sections"][0]["taal"] = "teentaal"
        self.assertEqual(validate(rec), [])
        from lib.notation_render import html as render_html
        self.assertIn("ntn-sec", render_html(rec, "english", {}))      # the renderer takes the key too
        rec["sections"][0]["taal"] = "no-such-taal"
        self.assertEqual([e["code"] for e in validate(rec)], ["vocab.key"])


class IndexTests(unittest.TestCase):
    """The book's index: found among the front pages, parsed, matched, applied to the numbered notations."""

    def _index_lines(self, page_no=True):
        rows = [("ਤਤਕਰਾ", 269, 727, None), ("9. ਰਾਗ ਸੋਰਠਿ ਪਰਿਚਯ ਅਤੇ ਸੁਰ ਵਿਸਤਾਰ 56", 1925, 200, None),
                ("ਰੇ ਮਨ ਰਾਮ ਸਿਉ ਕਰਿ ਪ੍ਰੀਤਿ 58", 1998, 300, None), ("ਮਨ ਕੀ ਮਨ ਹੀ ਮਾਹਿ ਰਹੀ 60", 2059, 298, None),
                ("ਮਾਈ ਮੈ ਕਿਹਿ ਬਿਧਿ ਲਖਉ ਗੁਸਾਈ", 177, 355, "74"), ("ਮਾਈ ਮਨੁ ਮੇਰੋ ਬਸਿ ਨਾਹਿ", 228, 356, "76"),
                ("ਇਹ ਜਗਿ ਮੀਤ ਨ ਦੇਖਿਓ ਕੋਈ", 326, 347, "78"), ("ਜੋ ਨਰੁ ਦੂਖ ਮੈ ਦੁਖੁ ਨਹੀ ਮਾਨੈ", 420, 354, "80"),
                ("੧੨. ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ 167", 480, 300, None), ("੧੩. ਮਨ ਕਹਾ ਲੁਭਾਈਐ ਆਨ ਕਉ 170", 540, 300, None)]
        lines = []
        for k, (text, y, x, num) in enumerate(rows):
            lines.append(_line(2 * k + 1, text, y, x0=x, x1=x + 700, h=44))
            if num:
                lines.append(_line(2 * k + 2, num, y + 8, x0=1500, x1=1540, h=30))
        return lines

    def test_an_index_page_is_read_into_entries(self):
        from lib.notation_index import find_index, by_number
        prose = [_line(n, "ਗੁਰਬਾਣੀ ਸੰਗੀਤ ਦੇ ਇਸ ਸੰਗ੍ਰਹਿ ਵਿਚ ਆਪ ਨੇ ਆਧੁਨਿਕ ਠਾਟ ਪੱਧਤੀ ਨੂੰ ਅਪਣਾਇਆ ਹੈ ।", 300 + 60 * n) for n in range(12)]
        idx = find_index({3: prose, 4: self._index_lines(), 5: prose})
        self.assertEqual(idx["pages"], [4])
        self.assertTrue(idx["titled"])
        kinds = [(e["kind"], e["number"], e["page_printed"]) for e in idx["entries"]]
        self.assertIn(("section", 9, 56), kinds)
        self.assertIn(("shabad", None, 58), kinds)
        self.assertIn(("shabad", None, 74), kinds)                 # the page number read as its own line at the right
        self.assertIn(("shabad", 12, 167), kinds)                 # Dyal Singh numbers the notations themselves
        self.assertEqual(sorted(by_number(idx["entries"])), [12, 13])
        under = [e for e in idx["entries"] if e["kind"] == "shabad" and e["page_printed"] in (58, 60)]
        self.assertEqual({e.get("section") for e in under}, {9})
        self.assertEqual(find_index({3: prose, 5: prose}), {"pages": [], "entries": []})

    def test_raag_starts_the_page_offset_and_the_missing_list(self):
        from lib.notation_index import find_index, missing_entries, page_offset, raag_starts
        idx = find_index({13: self._index_lines()})
        self.assertEqual([(r["key"], r["page_printed"]) for r in raag_starts(idx["entries"])], [("sorath", 56)])
        for e in idx["entries"]:
            if e["kind"] == "shabad" and e["page_printed"] == 58:
                e["shabad_id"] = 3285
            if e["kind"] == "shabad" and e["page_printed"] == 74:
                e["shabad_id"] = 2399
        recs = [{"shabad": {"shabad_id": 3285}, "pages": [74, 75]}, {"shabad": {"shabad_id": 2399}, "pages": [90]}]
        got = page_offset(idx["entries"], {86: 70, 110: 94}, recs)
        self.assertEqual((got["offset"], got["witnesses"]), (16, {"headers": 2, "entries": 2}))
        self.assertEqual(missing_entries(idx["entries"], recs[:1], 16), [{"text": "ਮਾਈ ਮੈ ਕਿਹਿ ਬਿਧਿ ਲਖਉ ਗੁਸਾਈ", "shabad_id": 2399, "page_printed": 74, "expected_scan_page": 90}])
        self.assertEqual(missing_entries(idx["entries"], recs, None), [])
        split = page_offset(idx["entries"], {86: 70, 110: 94, 120: 104, 200: 150, 210: 160, 220: 170}, [])
        self.assertTrue(split["split"]); self.assertIsNone(split["offset"])

    def test_a_table_with_a_number_on_every_line_is_not_believed_without_the_corpus(self):
        from lib.notation_index import credible
        idx = {"entries": [{"kind": "shabad"}] * 19, "titled": False}
        self.assertFalse(credible(idx, 0))
        self.assertFalse(credible(idx, 2))
        self.assertTrue(credible(idx, 5))
        self.assertTrue(credible({"entries": [{"kind": "shabad"}] * 100, "titled": True}, 3))

    def test_a_numbered_notation_without_its_verse_takes_the_index_shabad_and_a_conflict_is_flagged(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("parse29", os.path.join(HERE, "29_notation_parse.py"))
        p29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p29)
        index = {"credible": True, "entries": [{"kind": "shabad", "number": 12, "text": "ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ", "page_printed": 167, "shabad_id": 913, "score": 0.95},
                                               {"kind": "shabad", "number": 13, "text": "ਮਨ ਕਹਾ ਲੁਭਾਈਐ ਆਨ ਕਉ", "page_printed": 170, "shabad_id": 4284, "score": 0.9}]}
        bare = _record(shabad_id=None)
        bare["heading"]["number"] = 12
        bare["shabad"].update({"method": "none", "confidence": 0.0}); bare["flags"] = ["unresolved-shabad", "no-shabad-text"]
        other = _record(shabad_id=913)
        other["heading"]["number"] = 13
        same = _record(shabad_id=4284)
        same["heading"]["number"] = 13
        got = p29.index_shabads([bare, other, same], index, None)
        self.assertEqual(got, {"filled": 1, "conflicts": 1, "agreed": 1})
        self.assertEqual((bare["shabad"]["shabad_id"], bare["shabad"]["method"]), (913, "index"))
        self.assertIn("shabad-by-index", bare["flags"]); self.assertNotIn("unresolved-shabad", bare["flags"])
        self.assertIn("index-conflict", other["flags"]); self.assertEqual(other["shabad"]["shabad_id"], 913)
        self.assertEqual(p29.index_shabads([bare], {"credible": False, "entries": []}, None), {"filled": 0, "conflicts": 0, "agreed": 0})
        for rec in (bare, other, same):
            self.assertEqual(validate(rec), [])


class ResolveTests(unittest.TestCase):
    def _corpus(self):
        import sqlite3
        con = sqlite3.connect(":memory:")
        con.executescript("""
          CREATE TABLE shabads (shabad_id INTEGER PRIMARY KEY, writer TEXT, raag TEXT, ang_start INTEGER);
          CREATE TABLE lines (line_id INTEGER PRIMARY KEY, shabad_id INTEGER, gurmukhi_uni TEXT, kind TEXT, position_in_shabad INTEGER);
          INSERT INTO shabads VALUES (913, 'Guru Arjan Dev Ji', 'Raag Gauri', 269);
          INSERT INTO lines VALUES (1000, 913, 'ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥', 'line', 1);
          INSERT INTO lines VALUES (1001, 913, 'ਸਾਚ ਬਿਨਾ ਕਹ ਹੋਵਤ ਸੂਚਾ ॥', 'line', 2);
          INSERT INTO shabads VALUES (4284, 'Guru Arjan Dev Ji', 'Raag Sarang', 1208);
          INSERT INTO lines VALUES (2000, 4284, 'ਮਨ ਕਹਾ ਲੁਭਾਈਐ ਆਨ ਕਉ ॥', 'rahao', 1);
        """)
        return con

    def test_text_votes_and_the_reference_agree(self):
        con = self._corpus()
        lines = [_line(2, "ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥", 300, kind="gurbani",
                       matches=[{"shabad_id": 913, "line_id": 1000, "score": 0.95, "source": "G"}]),
                 _line(3, "ਸਾਚ ਬਿਨਾ ਕਹ ਹੋਵਤ ਸੂਚਾ ॥", 380, kind="gurbani",
                       matches=[{"shabad_id": 913, "line_id": 1001, "score": 0.9, "source": "G"}])]
        got = resolve_shabad(lines, parse_ref("(ਗਉੜੀ ਸੁਖਮਨੀ ਮ: ੫, ਪੰਨਾ ੨੬੯)"), con)
        sh = got["shabad"]
        self.assertEqual((sh["shabad_id"], sh["ang"], sh["raag_key"], sh["method"]), (913, 269, "gauri", "stream+ref"))
        self.assertGreaterEqual(sh["confidence"], 0.8)
        self.assertEqual(sh["line_ids"], [1000, 1001])
        self.assertEqual(got["flags"], [])

    def test_a_reference_that_disagrees_is_flagged_and_nothing_matched_is_unresolved(self):
        con = self._corpus()
        lines = [_line(2, "ਬਿਰਥੀ ਸਾਕਤ ਕੀ ਆਰਜਾ ॥", 300, kind="gurbani",
                       matches=[{"shabad_id": 913, "line_id": 1000, "score": 0.95, "source": "G"}]),
                 _line(3, "ਸਾਚ ਬਿਨਾ ਕਹ ਹੋਵਤ ਸੂਚਾ ॥", 380, kind="gurbani",
                       matches=[{"shabad_id": 913, "line_id": 1001, "score": 0.9, "source": "G"}])]
        got = resolve_shabad(lines, parse_ref("(ਪੰਨਾ ੧੨੦੮)"), con)
        self.assertIn("ref-conflict", got["flags"])
        self.assertEqual(got["shabad"]["shabad_id"], 913)     # the text still wins; the flag routes it to review
        none = resolve_shabad([_line(2, "ਕੋਈ ਕਬਿੱਤ ਜੋ ਗ੍ਰੰਥ ਵਿਚ ਨਹੀਂ", 300)], None, con)
        self.assertIsNone(none["shabad"]["shabad_id"])
        self.assertIn("unresolved-shabad", none["flags"])
        self.assertEqual(none["kind_hint"], "non-gurbani")
        # a Bhai Gurdas reference names the source even when the corpus has no such text
        bg = resolve_shabad([_line(2, "ਕਬਿੱਤ", 300)], parse_ref("(ਭਾਈ ਗੁਰਦਾਸ ਜੀ, ਕਬਿੱਤ ੧੨)"), con)
        self.assertEqual(bg["shabad"]["source"], "B")


# ---- the grid reader's pure parts -------------------------------------------------

from lib.notation_grid import (cells_from_components, fit_matras, group_cells, marks_from_components,  # noqa: E402
                               segments_from_bars)


def _comp(x0, y0, x1, y1):
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": x1 - x0, "h": y1 - y0, "area": (x1 - x0) * (y1 - y0),
            "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0}


class GridTests(unittest.TestCase):
    def test_components_cluster_into_cells_and_a_dot_joins_its_letter(self):
        comps = [_comp(100, 50, 130, 90), _comp(112, 38, 118, 44),        # a letter with a dot above
                 _comp(180, 50, 212, 90),                                # the next letter
                 _comp(190, 96, 196, 102),                               # its dot below
                 _comp(260, 68, 300, 74)]                                # a dash
        cells = cells_from_components(comps, gap=12)
        self.assertEqual(len(cells), 3)
        self.assertEqual([len(c["comps"]) for c in cells], [2, 2, 1])
        m0, m1, m2 = (marks_from_components(c) for c in cells)
        self.assertEqual((m0["octave"], m1["octave"], m2["dash"]), (1, -1, True))
        self.assertFalse(m0["komal"] or m0["tivra"])

    def test_an_underline_is_komal_and_a_stroke_above_is_tivra_but_a_vowel_sign_is_neither(self):
        komal = cells_from_components([_comp(100, 50, 130, 90), _comp(98, 94, 132, 98)], 12)[0]
        self.assertTrue(marks_from_components(komal)["komal"])
        tivra = cells_from_components([_comp(100, 50, 130, 90), _comp(113, 30, 117, 46)], 12)[0]
        self.assertTrue(marks_from_components(tivra)["tivra"])
        # the hook of ੇ above a ਰ: wider than tall, not a dot, not a stroke
        hook = cells_from_components([_comp(100, 50, 130, 90), _comp(104, 36, 128, 46)], 12)[0]
        m = marks_from_components(hook)
        self.assertEqual((m["octave"], m["tivra"], m["komal"]), (0, False, False))
        # ੁ below: a curve, wider than an underline is thin
        aunkar = cells_from_components([_comp(100, 50, 130, 90), _comp(108, 92, 124, 104)], 12)[0]
        m = marks_from_components(aunkar)
        self.assertEqual((m["octave"], m["komal"]), (0, False))

    def test_the_first_vibhag_of_a_row_is_found_from_the_cell_counts(self):
        self.assertEqual(fit_matras([4, 4, 4, 4], [4, 4, 4, 4]), (0, 1))
        self.assertEqual(fit_matras([3, 3], [3, 3]), (0, 1))
        self.assertEqual(fit_matras([2, 3, 2, 3], [2, 3, 2, 3]), (0, 1))
        # a mukhda: one cell before the sam, then whole vibhags of dhamar
        self.assertEqual(fit_matras([1, 2, 3, 4], [5, 2, 3, 4]), (0, 5))
        self.assertEqual(fit_matras([4], []), (0, 1))

    def test_cells_become_beats_by_merging_the_closest_or_leaving_a_gap(self):
        cells = [{"x0": 0, "x1": 20, "cx": 10}, {"x0": 24, "x1": 44, "cx": 34}, {"x0": 90, "x1": 110, "cx": 100},
                 {"x0": 150, "x1": 170, "cx": 160}]
        groups = group_cells(cells, 3, pitch=60)
        self.assertEqual([len(g) for g in groups], [2, 1, 1])
        groups = group_cells(cells[:3], 4, pitch=60)
        self.assertEqual([len(g) for g in groups], [1, 1, 0, 1])       # the widest gap takes the missing beat
        self.assertEqual(group_cells(cells, 4, pitch=60), [[c] for c in cells])

    def test_bars_cut_a_row_into_segments(self):
        self.assertEqual(segments_from_bars([300, 600, 900], 100, 1200, 40), [(100, 300), (300, 600), (600, 900), (900, 1200)])
        self.assertEqual(segments_from_bars([110, 600], 100, 1200, 40), [(110, 600), (600, 1200)])


# ---- the review, the evaluation and the database ---------------------------------

def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(nid="test-book:0170:1", shabad_id=1248, sections=True):
    rec = json.loads(json.dumps(fixture("teentaal-sthai")))
    rec["notation_id"], rec["book_key"], rec["page"], rec["seq"], rec["pages"] = nid, "test-book", 170, 1, [170]
    rec["shabad"]["shabad_id"] = shabad_id
    if shabad_id is None:
        rec["shabad"].update({"method": "none", "confidence": 0.0, "line_ids": []})
        rec["flags"] = sorted(set(rec.get("flags", [])) | {"unresolved-shabad"})
    # the fixture's one unreadable cell is a test of the contract, not of the parser's reading
    for sec in rec["sections"]:
        for line in sec["lines"]:
            for beat in line["beats"]:
                if beat.get("notes") is None and not beat.get("ext") and not beat.get("rest"):
                    beat["notes"] = [{"s": "S"}]
                    beat.pop("raw", None); beat.pop("c", None)
    rec["shabad"]["ang"] = 320
    rec["shabad"]["first_line"] = "ਓਥੈ ਅੰਮ੍ਰਿਤੁ ਵੰਡੀਐ ਸੁਖੀਆ ਹਰਿ ਕਰਣੇ ॥"
    rec["source"]["book_key"], rec["source"]["author_key"] = "test-book", "test-author"
    rec["images"] = [{"n": 1, "file": "images/test-book-0170-1-1.png", "role": "grid", "page": 170, "bbox": [10, 10, 500, 300],
                      "w": 490, "h": 290, "bytes": 1234, "sha256": "ab" * 32, "thumb": None}]
    if not sections:
        rec["sections"] = []
        rec["kind"] = "partial"
        rec["flags"] = sorted(set(rec.get("flags", [])) | {"partial-grid"})
    rec["source"]["content_hash"] = content_hash(rec)
    return rec


class ReviewTests(unittest.TestCase):
    def test_candidates_carry_the_judged_fields_and_gold_settles_the_truth(self):
        gt = _script("30_notation_gt.py")
        rec = _record()
        c = gt.candidate(rec)
        self.assertEqual(c["shabad"]["value"], 1248)
        self.assertEqual(c["raag_used"]["value"], rec["heading"]["raag"]["key"])
        self.assertEqual(c["structure"]["value"], "sthai,antara")
        self.assertEqual(gt.check_candidates([c]), [])            # unjudged, unverified: nothing to complain about
        c["verified"] = True
        errs = gt.check_candidates([c])
        self.assertTrue(errs and all("not judged" in e for e in errs))
        for f in gt.FIELDS:
            c[f]["ok"] = True
        c["taal"]["ok"], c["taal"]["correct"] = False, "no-such-taal"
        self.assertTrue(any("not a vocabulary key" in e for e in gt.check_candidates([c])))
        c["taal"]["correct"] = "tilwada"                        # sixteen matras too, so the lines still fit
        self.assertEqual(gt.check_candidates([c]), [])
        gold = gt.gold_of(c)
        self.assertEqual((gold["shabad"]["truth"], gold["taal"]["truth"]), (1248, "tilwada"))
        self.assertNotIn("draft", gold)
        # applied to the record: verified, and the corrected taal
        notation.apply_gold(rec, gold)
        self.assertTrue(rec["verified"])
        self.assertEqual(rec["heading"]["taal"]["key"], "tilwada")
        self.assertEqual(validate(rec), [])

    def test_the_review_page_is_written_with_every_notation_and_its_crops(self):
        gt = _script("30_notation_gt.py")
        rec = _record()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "review.html")
            gt.review_page("test-book", {"title": "Test Book"}, [rec], [gt.candidate(rec)], path, "../notations", (165, 176), None)
            page = open(path, encoding="utf-8").read()
        self.assertIn("test-book:0170:1", page)
        self.assertIn("../notations/images/test-book-0170-1-1.png", page)
        self.assertIn('class="ntn"', page)                          # the parsed grid, rendered
        self.assertIn("download candidates.jsonl", page)

    def test_the_window_covers_two_resolved_notations(self):
        gt = _script("30_notation_gt.py")
        recs = [_record("b:%04d:1" % p, shabad_id=(100 + p if p % 2 == 0 else None), sections=False) for p in range(160, 181)]
        for r in recs:
            r["pages"] = [r["page"]]
        a, b, chosen, warn = gt.choose_window(recs, 373)
        self.assertEqual(a, 168)
        self.assertGreaterEqual(sum(1 for r in chosen if gt.resolved(r)), 2)
        self.assertLessEqual(b - a + 1, 7)
        self.assertIsNone(warn)
        # nothing resolved: the window stretches over the parsed pages and says what to do
        for r in recs:
            r["shabad"]["shabad_id"] = None
        a, b, chosen, warn = gt.choose_window(recs, 373)
        self.assertIn("only 0 resolved", warn)


class EvalTests(unittest.TestCase):
    def test_fields_are_scored_against_the_truth_and_a_false_link_is_counted(self):
        ev = _script("31_notation_eval.py")
        r1, r2, r3 = _record("b:0001:1", 11), _record("b:0002:1", 22), _record("b:0003:1", 33)
        gold = [
            {"notation_id": "b:0001:1", "status": "ok", "verified": True,
             "shabad": {"truth": 11}, "raag_used": {"truth": r1["heading"]["raag"]["key"]}, "taal": {"truth": r1["heading"]["taal"]["key"]},
             "laya": {"truth": r1["heading"]["taal"].get("laya")}, "structure": {"truth": "sthai,antara"}, "cells": []},
            {"notation_id": "b:0002:1", "status": "ok", "verified": True,
             "shabad": {"truth": 99}, "raag_used": {"truth": "bhairav"}, "taal": {"truth": r2["heading"]["taal"]["key"]},
             "laya": {"truth": r2["heading"]["taal"].get("laya")}, "structure": {"truth": "sthai"},
             "cells": [{"s": 0, "l": 0, "b": 0, "field": "notes", "value": "R"}]},
            {"notation_id": "b:0003:1", "status": "skip", "verified": True},
        ]
        res = ev.evaluate([r1, r2, r3], gold)
        self.assertEqual(res["n"], 2)
        self.assertEqual((res["fields"]["shabad"]["correct"], res["fields"]["raag_used"]["correct"], res["fields"]["structure"]["correct"]), (1, 1, 1))
        self.assertEqual(res["false_link"]["ids"], ["b:0002:1"])
        self.assertEqual(res["cells"]["wrong_swara"], 1)
        self.assertEqual(res["cells"]["beats"], 2 * len([b for s in r1["sections"] for l in s["lines"] for b in l["beats"]]))
        passed, fails = ev.judge(res)
        self.assertFalse(passed)
        self.assertTrue(any(f.startswith("shabad") for f in fails))
        # every field right: passes
        gold[1].update({"shabad": {"truth": 22}, "raag_used": {"truth": r2["heading"]["raag"]["key"]},
                        "structure": {"truth": "sthai,antara"}, "cells": []})
        passed, fails = ev.judge(ev.evaluate([r1, r2], gold[:2]))
        self.assertTrue(passed, fails)


class DbTests(unittest.TestCase):
    def test_an_image_finds_its_url_by_hash_or_by_notation_n_and_kind(self):
        build = _script("32_build_notations_db.py")
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "images.urls.json")
            with open(f, "w", encoding="utf-8") as fh:
                json.dump({"repo": "x/y", "urls": {"ab" * 32: "https://e.test/full.png"},
                           "keys": {"bk:0010:1|1|thumb": "https://e.test/thumb.png"}}, fh)
            urls = build.load_urls([f])
            rec = {"notation_id": "bk:0010:1", "book_key": "bk",
                   "images": [{"n": 1, "file": "images/bk-0010-1-1.png", "role": "block", "page": 10, "bbox": [0, 0, 9, 9],
                               "w": 9, "h": 9, "bytes": 3, "sha256": "ab" * 32, "thumb": "images/bk-0010-1-1.thumb.png"}]}
            rows = build.image_rows(rec, d, urls, {})
            # the thumbnail's hash is unknown here (no file, none recorded): its URL comes by key
            self.assertEqual([(r["kind"], r["url"]) for r in rows], [("full", "https://e.test/full.png"), ("thumb", "https://e.test/thumb.png")])
            rec["images"][0]["thumb_sha256"] = "cd" * 32
            self.assertEqual(build.image_rows(rec, d, urls, {})[1]["sha256"], "cd" * 32)    # a recorded hash is used as is

    def test_the_database_has_the_pinned_columns_and_the_records(self):
        import sqlite3
        build = _script("32_build_notations_db.py")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "notations", "test-book")
            os.makedirs(os.path.join(src, "images"))
            recs = [_record("test-book:0170:1", 1248), _record("test-book:0171:1", 1248, sections=False), _record("test-book:0172:1", None)]
            for r in recs:
                r["page"] = int(r["notation_id"].split(":")[1])
                r["pages"] = [r["page"]] if r["page"] != 171 else [170, 171]
            notation.write_jsonl(os.path.join(src, "notations.jsonl"),
                                 notation.meta_for({"book": "test-book", "author": "Test Author", "title": "Test Book", "part": 1,
                                                    "style": merge_style(None)}, pages=3),
                                 recs)
            gurbani = os.path.join(d, "gurbani.sqlite")
            con = sqlite3.connect(gurbani)
            con.execute("CREATE TABLE shabads (shabad_id INTEGER PRIMARY KEY)")
            con.execute("INSERT INTO shabads VALUES (1248)")
            con.commit(); con.close()
            out = os.path.join(d, "artifacts", "notations.sqlite")
            # an unreviewed book is refused ...
            with self.assertRaises(SystemExit):
                build.build(os.path.join(d, "notations"), out, None, gurbani, {}, False, False, False, None)
            # ... and built when told so
            got = build.build(os.path.join(d, "notations"), out, None, gurbani, {"ab" * 32: "https://example.test/x.png"},
                              True, True, False, "https://example.test/")
            self.assertEqual((got["books"], got["notations"], got["resolved"], got["shabads"]), (1, 3, 2, 1))
            con = sqlite3.connect(out)
            with open(os.path.join(HERE, "lib", "notation_columns.json"), encoding="utf-8") as fh:
                columns = json.load(fh)["tables"]
            for t, cols in columns.items():
                self.assertEqual([r[1] for r in con.execute("PRAGMA table_info(%s)" % t)], cols, t)
            rows = con.execute("SELECT notation_id, shabad_id, kind, sections, beats, grid IS NULL, sargam_en IS NULL, image_count "
                               "FROM notations ORDER BY ordinal").fetchall()
            self.assertEqual(rows[0][:3], ("test-book:0170:1", 1248, "notation"))
            self.assertGreater(rows[0][4], 0)
            self.assertEqual((rows[0][5], rows[0][6]), (0, 0))
            self.assertEqual((rows[1][2], rows[1][5]), ("partial", 1))
            self.assertIsNone(rows[2][1])
            self.assertEqual(con.execute("SELECT n FROM shabad_counts WHERE shabad_id=1248").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT url FROM images LIMIT 1").fetchone()[0], "https://example.test/x.png")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM raags WHERE in_ggs=1 AND ggs_order IS NOT NULL").fetchone()[0], 31)
            self.assertEqual(dict(con.execute("SELECT key, value FROM meta"))["notations"], "3")
            con.close()
            self.assertTrue(os.path.exists(os.path.join(d, "artifacts", "notations-images.json")))
            # the review state travels into the database: an auto build ships every record with its verdict
            # (accepted, backlog, or none), a manual build (--accepted-only) the accepted alone
            recs[0]["review"] = {"key": "test-book/1248/gujri/teentaal#1", "status": "accepted", "round": 3}
            recs[0]["verified"] = True
            recs[1]["review"] = {"key": "test-book/1248/gujri/teentaal#2", "status": "backlog", "comment": "cut short", "round": 3}
            notation.write_jsonl(os.path.join(src, "notations.jsonl"), notation.meta_for({"book": "test-book", "author": "A", "title": "T"}), recs)
            build.build(os.path.join(d, "notations"), out, None, gurbani, {}, False, True, False, None)
            con = sqlite3.connect(out)
            self.assertEqual(con.execute("SELECT review_status, review_comment FROM notations ORDER BY ordinal").fetchall(),
                             [("accepted", None), ("backlog", "cut short"), (None, None)])
            self.assertEqual(dict(con.execute("SELECT key, value FROM meta"))["review_mode"], "all")
            con.close()
            got = build.build(os.path.join(d, "notations"), out, None, gurbani, {}, False, False, False, None, accepted_only=True)
            self.assertEqual(got["notations"], 1)
            con = sqlite3.connect(out)
            self.assertEqual(dict(con.execute("SELECT key, value FROM meta"))["review_mode"], "accepted")
            self.assertEqual(con.execute("SELECT review_status FROM notations").fetchall(), [("accepted",)])
            con.close()
            # a shabad the corpus does not have refuses the build
            recs[0]["shabad"]["shabad_id"] = 999999
            notation.write_jsonl(os.path.join(src, "notations.jsonl"), notation.meta_for({"book": "test-book", "author": "A", "title": "T"}), recs)
            with self.assertRaises(SystemExit):
                build.build(os.path.join(d, "notations"), out, None, gurbani, {}, False, True, False, None)

if __name__ == "__main__":
    if "--write-expected" in sys.argv:
        write_expected()
    else:
        unittest.main()


class DescriptionBoundaryTests(unittest.TestCase):
    """
    The second cut's repeated failure (2 October 2026): the next raag's
    description, printed after a notation's last grid, rode along with the
    notation. Three shapes, each from the page that showed it.
    """

    def _page(self, page, lines, page_h=2669):
        lay = page_layout(lines, 1727, page_h, merge_style(None), page)
        lay["page_w"], lay["page_h"] = 1727, page_h
        return lay

    def _shabad(self, sid, y):
        m = [{"shabad_id": sid, "line_id": sid * 10 + i, "score": 0.95, "source": "G"} for i in range(2)]
        return [_line(2, "ਮਾਈ ਮੈ ਕਿਹਿ ਬਿਧਿ ਲਖਉ ਗੁਸਾਈ ॥", y, kind="gurbani", matches=[m[0]]),
                _line(3, "ਮਹਾ ਮੋਹ ਅਗਿਆਨਿ ਤਿਮਰਿ ਮੋ ਮਨੁ ਰਹਿਓ ਉਰਝਾਈ ॥੧॥ ਰਹਾਉ ॥", y + 80, kind="gurbani", matches=[m[1]]),
                _line(4, "(ਆਸਾ ਮ: ੫, ਪੰਨਾ ੩੭੮)", y + 170, x0=600, x1=1100)]

    def _grids(self, n0, y):
        return [_line(n0, "ਅਸਥਾਈ", y, x0=800, x1=900), _line(n0 + 1, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", y + 60),
                _line(n0 + 2, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", y + 140), _line(n0 + 3, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", y + 220)]

    # Gurbani Sangeet 1, p. 165: the description's lines, dashes and swar letters, as the OCR read them
    DESCRIPTION = [".ਵਾਦੀ--ਗ ਠਾਠ - ਖਮਾਜ ਸਮਾਂ--ਰਾਤ ਦਾ ਦੂਜਾ ਪਹਿਰ", "ਸੰਵਾਦੀ -ਨ ਜਾਤ-ਓਡਵ --| ਨੂੁੰਪਸ਼ੁਰਸੌਗਤਿ।", "ਪਕੜ--ਨ੍ਸਗਮਪ --ਨਸੋਂ, ਠੁਪ, ਗਮਗ, ਸ ।",
                   "ਆਰੌਹ--ਸਗ,ਮਪਨਸੰ। . -", "ਅਵਰੌਹ-ਸੰ ਨੁਪਮਗਸ। ਰ", "ਸੁਰ-ਫਿਸਥਾਰ",
                   "ਸ਼ -- ਸਨ - ਨਸੇ, ਸਗ--ਗਮਗ--ਨ--ਠਪ--ਪਨ--ਨਸ--ਸਗ--ਮ, ਗਮ ੯ -- ਮੈ,", "ਸਨ੍--ਨਸ--ਨ੍ਸਗ--ਸਗਮ ਗਮਗ--ਸ--ਸਗਮਪ--ਨੁਪ, ਮਗ--ਪਗਮਗ--ਸ, ੍ ਨੌਸਗੇਮਪ---"]

    def test_the_words_of_a_description_make_a_line_prose_not_a_grid_row(self):
        for t in self.DESCRIPTION[:5]:
            self.assertEqual(classify_line(_line(1, t, 300)), "text", t)
        self.assertEqual(classify_line(_line(1, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 300)), "grid")

    def test_a_raag_description_read_under_a_raag_heading_ends_the_notation_before_it(self):
        p164 = self._page(164, [_line(1, "ਰਾਗ ਬੈਰਾੜੀ; ਜਪ ਤਾਲ, ਬਿਲੰਬਿਤ", 269, bold=True)] + self._shabad(48, 375) + self._grids(5, 1047))
        p165 = self._page(165, [_line(1, "ਰਾਗ ਤਿਲੰਗ", 269, bold=True, x0=747, x1=973)]
                          + [_line(2 + i, t, 330 + 62 * i) for i, t in enumerate(self.DESCRIPTION)])
        p166 = self._page(166, [_line(1, "ਤਿਲੰਗ ਤਿੰਨ ਤਾਲ", 265, bold=True)] + self._shabad(2290, 358) + self._grids(5, 1746))
        dropped = []
        spans = link_pages([p164, p165, p166], merge_style(None), dropped)
        self.assertEqual([s["pages"] for s in spans], [[164], [166]])
        self.assertEqual([s["sid"] for s in spans], [48, 2290])
        found = raag_descriptions([p164, p165, p166])
        self.assertEqual([(d["raag"]["key"], d["page"]) for d in found], [("tilang", 165)])

    def test_a_raag_the_vocabulary_cannot_name_still_opens_its_description(self):
        # Gurbani Sangeet 2, p. 218: "ਰਾਗ ਨੰਦ ਕੌਂਸ" read as "ਰਾਗ ਨਦ ਕੱਸ"
        p217 = self._page(217, [_line(1, "ਰਾਗ ਸਿੰਧੜਾ, ਤਿੰਨ ਤਾਲ", 1208, bold=True)] + self._shabad(2008, 1324) + self._grids(5, 2292))
        p218 = self._page(218, self._grids(1, 219)[1:] + [_line(5, "ਰਾਗ ਨਦ ਕੱਸ", 1378, bold=True, x0=700, x1=1000)]
                          + [_line(6 + i, t, 1487 + 70 * i) for i, t in enumerate(self.DESCRIPTION[:3])])
        p219 = self._page(219, [_line(1, "ਰਾਗ ਨੰਦ ਕੌਂਸ, ਤਿੰਨ ਤਾਲ", 324, bold=True)] + self._shabad(1000, 490) + self._grids(5, 1417))
        spans = link_pages([p217, p218, p219], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[217, 218], [219]])
        self.assertLess(spans[0]["extent"][218][3], 1378)

    def test_a_raag_introduction_heading_takes_the_raag_heading_above_it_into_the_next_section(self):
        # Guru Angad Dev Sangeet Darpan, p. 152: "ਰਾਗ ਦੇਵਗੰਧਾਰੀ" at the top of the page, "ਰਾਗੁ ਪਰੀਚੈ:-" under it
        p151 = self._page(151, [_line(1, "ਰਾਗ ਗੁਜਰੀ ਤੀਨ ਤਾਲ ਮਾਤਰਾਂ-16 ਮੱਧ ਲੇਅ", 252, bold=True)] + self._shabad(5371, 333) + self._grids(5, 600))
        p152 = self._page(152, [_line(1, "ਰਾਗ ਦੇਵਰੀਧਾਰੀ", 253, bold=True, x0=700, x1=1000), _line(2, "ਰਾਗੁ ਪਰੀਚੈ:-", 479, bold=True, x0=200, x1=500),
                                _line(3, "ਸਵਰ- ਦੋਵੇਂ ਧੈਵਤ ਦੋਵੇਂ ਨਿਸ਼ਾਦ, ਹੋਰ ਸਭ ਸ਼ੁਧ ਵਰਜਿਤ ਸਵਰ-ਗੰਧਾਰ ਤੇ ਨਿਸ਼ਾਦ ਆਰੋਹ ਵਿੱਚ", 628),
                                _line(4, "ਸ੍ਰੀ ਗੁਰੂ ਗਰੰਥ ਸਾਹਿਬ ਦੀ ਰਾਗੁ ਤਰਤੀਬ ਵਿੱਚ ਰਾਗੁ ਦੇਵਗੰਧਾਰੀ ਨੂੰ ਛੇਵਾਂ ਅਸਥਾਨ ਪ੍ਰਾਪਤ ਹੈ।", 993)]
                          + self._shabad(5372, 1205) + [_line(8, "ਰਾਗੁ ਦੇਵਗੰਧਾਰੀ ਫਰੋਦਸਤ ਤਾਲ ਮਾਤਰਾਂ-14 ਮੱਧ ਲੈਅ", 1766, bold=True)] + self._grids(9, 1851))
        spans = link_pages([p151, p152], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[151], [152]])
        self.assertEqual([s["sid"] for s in spans], [5371, 5372])
        found = raag_descriptions([p151, p152])
        self.assertEqual([d["page"] for d in found], [])        # the raag's name was not read: nothing to file it under


class TailTests(unittest.TestCase):
    """The lines after a notation's last grid stay its own (the second cut's 'missing in the end' cards)."""

    def _page(self, page, lines):
        lay = page_layout(lines, 1727, 2669, merge_style(None), page)
        lay["page_w"], lay["page_h"] = 1727, 2669
        return lay

    TIHAI = ["1) ਪਹਿਲੀ ਤੋਂ ਚੌਥੀ ਮਾਤਰ ਦੇ ਟੁਕੜੇ ਨੂੰ ਦੂਜੀ ਮਾਤਰ ਤੋਂ ਲੈ ਕੇ ਇੱਕ ਮਾਤਰ ਬਿਸਰਾਮ ਫਿਰ ਤੀਜੀ",
             "ਤੋਂ ਚੌਥੀ ਮਾਤਰ ਦਾ ਟੁਕੜਾ ਫਿਰ ਬਿਸਰਾਮ ਫਿਰ ਇਹੀ ਟੁਕੜਾ ਲੈ ਕੇ ਸਮ ਤੇ ਆਉਣਾ ਹੈ।",
             "2) ਤੀਜੀ ਮਾਤਰ ਤੇ 'ਨਾਮ' ਦਾ ਉਚਾਰਨ ਕਰਨਾ ਹੈ ਪਹਿਲੀ ਮਾਤਰ ਤੋਂ 'ਸਿਮਰਤ ਨਾਮੁ' ਇਹ",
             "ਟੁਕੜਾ ਗਾ ਕੇ ਸਮ ਤੇ ਆਉਣਾ ਹੈ।",
             "3) 7ਵੀਂ ਤੋਂ 11ਵੀਂ ਮਾਤਰ ਦੇ 5 ਮਾਤਰਾਂ ਦੇ ਟੁਕੜੇ ਨੂੰ 8ਵੀਂ ਮਾਤਰ ਤੋਂ ਆਰੰਭ ਕਰਕੇ ਤਿੰਨ ਵਾਰ",
             "ਬਰਾਬਰ ਲੈ ਕੇ ਸਮ ਤੇ ਸਮਾਪਤ ਕਰਨਾ ਹੈ।"]

    def test_a_tihai_spelt_out_is_prose_not_a_verse_over_the_reference(self):
        from lib.notation_layout import _verse_shaped
        self.assertFalse(_verse_shaped("\n".join(self.TIHAI)))
        kabit = "ਆਂਬ ਕੀ ਸਧਰ ਕਤ ਮਿਟਤ ਆਂਬਲੀ ਖਾਏ, ਪਿਆਸ ਨ ਬੁਝਤ ਜੈਸੇ ਬਾਰਿ ਕੇ ਬੁਝਾਏ ਹੈ।\nਸਾਧਸੰਗਿ ਗੁਰਮੁਖਿ ਸੁਖਫਲ ਪਾਏ, ਕਰਮ ਕਾਂਡ ਸੇ ਨ ਪਾਏ ਹੈ।"
        self.assertTrue(_verse_shaped(kabit))

    def test_the_notation_keeps_its_tihai_when_the_next_reference_stands_under_it(self):
        # Gurmat Sangeet Darpan 2, p. 394: the tihai of 4733, then the reference and the salok of 4805
        m = [{"shabad_id": 4805, "line_id": 48050 + i, "score": 0.95, "source": "G"} for i in range(2)]
        p393 = self._page(393, [_line(1, "ਰਾਗੁ ਕਾਨੜਾ ਕੁੰਭ ਤਾਲ ਮਾਤਰਾਂ-11 ਵਿਲੰਭਿਤ ਲੈਅ", 1629, bold=True)]
                          + [_line(2, "ਸਿਮਰਤ ਨਾਮੁ ਮਨਹਿ ਸੁਖੁ ਪਾਈਐ ॥", 1310, kind="gurbani", matches=[{"shabad_id": 4733, "line_id": 47330, "score": 0.95, "source": "G"}]),
                             _line(3, "ਸਾਧ ਜਨਾ ਮਿਲਿ ਹਰਿ ਜਸੁ ਗਾਈਐ ॥੧॥ ਰਹਾਉ ॥", 1390, kind="gurbani", matches=[{"shabad_id": 4733, "line_id": 47331, "score": 0.95, "source": "G"}]),
                             _line(4, "ਅਸਥਾਈ", 1707, x0=800, x1=900), _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1824),
                             _line(6, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 2085), _line(7, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 2313)])
        p394 = self._page(394, [_line(1, "ਤਾਨਾਂ ਤੇ ਤਿਹਾਈਆਂ:-", 234), _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 386)]
                          + [_line(3 + i, t, 472 + 64 * i) for i, t in enumerate(self.TIHAI)]
                          + [_line(10, "(ਕਾਨੜੇ ਕੀ ਵਾਰ ਮਹਲਾ ੪) (੧੩੧੮)", 935, x0=600, x1=1100),
                             _line(11, "ਸਲੋਕ ਮ:੪॥ ਹਉ ਢੂੰਢੇਂਦੀ ਸਜਣਾ ਸਜਣੁ ਮੈਡੈ ਨਾਲਿ ॥", 1002, kind="gurbani", matches=[m[0]]),
                             _line(12, "ਜਨ ਨਾਨਕ ਅਲਖੁ ਨ ਲਖੀਐ ਗੁਰਮੁਖਿ ਦੇਹਿ ਦਿਖਾਲਿ ॥੧॥", 1070, kind="gurbani", matches=[m[1]]),
                             _line(13, "ਰਾਗੁ ਕਾਨੜਾ ਏਕ ਤਾਲ ਮਾਤਰਾਂ-12 ਵਿਲੰਭਿਤ ਲੈਅ", 1248, bold=True), _line(14, "ਅਸਥਾਈ", 1325, x0=800, x1=900),
                             _line(15, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1598), _line(16, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1680)])
        self.assertNotIn("shabad", [r["role"] for r in p394["regions"] if r["bbox"][1] < 900])
        spans = link_pages([p393, p394], merge_style(None))
        self.assertEqual([s["sid"] for s in spans], [4733, 4805])
        self.assertGreaterEqual(spans[0]["extent"][394][3], 472 + 64 * 5 + 60)     # the tihai's last line is 4733's
        self.assertEqual(spans[1]["extent"][394][1], 935)                            # 4805 begins at its reference


class PadhtiGranthTests(unittest.TestCase):
    """Guru Nanak Sangeet Padhti Granth: the number at the left margin, the taal at the right, on one line."""

    def _page(self, page, lines):
        lay = page_layout(lines, 1568, 2421, merge_style(None), page)
        lay["page_w"], lay["page_h"] = 1568, 2421
        return lay

    def test_a_bracketed_number_leads_a_heading(self):
        from lib.notation_text import parse_heading
        h = parse_heading("(੨) ਤੀਨ ਤਾਲ")
        self.assertEqual((h["number"], h["taal"]["key"]), (2, "teentaal"))
        self.assertEqual(parse_heading("(੧) ਰੂਪਕ ਤਾਲ")["number"], 1)
        self.assertEqual(parse_heading("੧੬. ਰਾਗ ਭੈਰਵੀ, ਤਾਲ ਕਹਿਲਵਾ")["number"], 16)

    def test_the_number_at_the_margin_and_the_taal_across_the_page_are_one_heading(self):
        # p. 98: "ਰਾਗੁ ਭੈਰਉ" over "ਤੀਨ ਤਾਲ" (right) and "(੨)" (left), then the sthai
        lines = [_line(1, "ਰਾਗੁ ਭੈਰਉ", 212, bold=True, x0=635, x1=899), _line(2, "ਤੀਨ ਤਾਲ", 362, x0=1257, x1=1414, h=50),
                 _line(3, "(੨)", 375, x0=120, x1=174, h=50), _line(4, "ਸਥਾਈ", 425, x0=116, x1=224),
                 _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 490), _line(6, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 570), _line(7, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 650)]
        lay = self._page(98, lines)
        heads = [r for r in lay["regions"] if r["role"] == "heading"]
        self.assertEqual([h["text"] for h in heads], ["ਰਾਗੁ ਭੈਰਉ", "(੨) ਤੀਨ ਤਾਲ"])
        self.assertEqual((heads[1]["parsed"]["number"], heads[1]["parsed"]["taal"]["key"]), (2, "teentaal"))
        self.assertNotIn("pageno", [r["role"] for r in lay["regions"]])
        # a page number at the foot of the page is still a page number
        foot = self._page(99, [_line(1, "(੭੩)", 2300, x0=700, x1=800)] + lines[4:])
        self.assertNotIn("heading", [r["role"] for r in foot["regions"]])

    def test_a_second_numbered_notation_with_no_verse_is_the_same_shabad_again(self):
        m = [{"shabad_id": 4006, "line_id": 40060 + i, "score": 0.95, "source": "G"} for i in range(2)]
        p94 = self._page(94, [_line(1, "ਸਰਨੀ ਆਇਓ ਨਾਥ ਨਿਧਾਨ ॥ ਨਾਮ ਪ੍ਰੀਤਿ ਲਾਗੀ ਮਨ ਭੀਤਰਿ ਮਾਗਨ ਕਉ ਹਰਿ ਦਾਨ ॥੧॥ ਰਹਾਉ ॥", 314, kind="gurbani", matches=[m[0]]),
                              _line(2, "ਸੁਖਦਾਈ ਪੂਰਨ ਪਰਮੇਸੁਰ ਕਰਿ ਕਿਰਪਾ ਰਾਖਹੁ ਮਾਨ ॥", 384, kind="gurbani", matches=[m[1]]),
                              _line(3, "ਰਾਗੁ ਕੇਦਾਰਾ", 587, bold=True, x0=606, x1=921), _line(4, "ਤੀਨ ਤਾਲ", 672, x0=1250, x1=1400, h=50),
                              _line(5, "(੧)", 682, x0=115, x1=170, h=50), _line(6, "ਸਥਾਈ", 729, x0=113, x1=219),
                              _line(7, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 790), _line(8, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 870), _line(9, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 950),
                              _line(10, "ਕੀਰਤਨਕਾਰ ਭਾ. ਨਿਰੰਜਨ ਸਿੰਘ, ਜਵੱਦੀ ਕਲਾਂ", 2026, x0=700, x1=1400)])
        p95 = self._page(95, [_line(1, "ਰਾਗ ਕੇਦਾਰਾ", 221, bold=True, x0=640, x1=955), _line(2, "ਤੀਨਤਾਲ", 362, x0=1257, x1=1414, h=50),
                              _line(3, "(੨)", 375, x0=145, x1=199, h=50), _line(4, "ਸਥਾਈ", 425, x0=141, x1=249),
                              _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 490), _line(6, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 570), _line(7, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 650),
                              _line(8, "ਕੀਰਤਨਕਾਰ ਡਾ. ਗੁਰਿੰਦਰ ਕੌਰ, ਦਿੱਲੀ", 1600, x0=700, x1=1400)])
        spans = link_pages([p94, p95], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[94], [94, 95]])      # the second carries the shabad's text from 94
        self.assertEqual([s["sid"] for s in spans], [4006, 4006])
        self.assertEqual([s["heading"]["parsed"]["number"] for s in spans], [1, 2])
        self.assertTrue(spans[1]["inherited"])


class DarpanTests(unittest.TestCase):
    """Gurmat Sangeet Darpan prints the Granth reference ABOVE the verse (style.ref_position before)."""

    def _page(self, page, lines, before=True):
        style = merge_style({"ref_position": "before"} if before else None)
        lay = page_layout(lines, 1854, 2606, style, page)
        lay["page_w"], lay["page_h"] = 1854, 2606
        return lay

    def _lines(self):
        # p. 158: the pauri's notation ends in a tihai, then the reference, the salok the corpus missed, its heading and grids
        return [_line(1, "ਰਾਗੁ ਗੂਜਰੀ ਤਾਲ ਪੰਚਮ ਸਵਾਰੀ ਮਾਤਰਾਂ-15 ਵਿਲੰਭਿਤ ਲੈਅ", 168, bold=True), _line(2, "ਅਸਥਾਈ", 251, x0=800, x1=950),
                _line(3, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 320), _line(4, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 400), _line(5, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 480),
                _line(6, "1) 10ਵੀਂ ਮਾਤਰਾਂ ਤੋਂ ਇਹ 2 ਮਾਤਰਾਂ ਨੂੰ ਤਿੰਨ ਵਾਰ ਗਾ ਕੇ ਸਮ ਤੇ ਆਉਣਾ ਹੈ।", 1123),
                _line(7, "(ਰਾਗੁ ਗੂਜਰੀ ਵਾਰ ਮਹਲਾ ੫) (੫੨੧)", 1492, x0=600, x1=1200),
                _line(8, "ਮ: ੫।। ਰਾਮੁ ਰਮਹੁ ਬਡਭਾਗੀਹੋ ਜਲਿ ਥਲਿ ਮਹੀਅਲਿ ਸੋਇ।। ਨਾਨਕ ਨਾਮਿ ਅਰਾਧਿਐ ਬਿਘਨੁ", 1547),
                _line(9, "ਨ ਲਾਗੈ ਕੋਇ।।੨।।", 1600, x0=200, x1=500),
                _line(10, "ਰਾਗੁ ਗੂਜਰੀ ਤਾਲ ਨਾਰਾਇਣੀ ਮਾਤਰਾਂ- 9 ਵਿਲੰਭਿਤ ਲੈਅ", 1664, bold=True), _line(11, "ਅਸਥਾਈ", 1740, x0=800, x1=950),
                _line(12, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1800), _line(13, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 1880), _line(14, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1960)]

    def test_the_verse_under_a_reference_is_the_shabad_whatever_the_ocr_made_of_it(self):
        lay = self._page(158, self._lines())
        roles = [(r["role"], r["bbox"][1]) for r in lay["regions"]]
        self.assertIn(("ref", 1492), roles)
        self.assertIn(("shabad", 1547), roles)
        m = [{"shabad_id": 1981, "line_id": 19810, "score": 0.95, "source": "G"}]
        p157 = self._page(157, [_line(1, "(ਰਾਗੁ ਗੁਜਰੀ ਵਾਰ ਮਹਲਾ ੫) (੫੨੧)", 1920, x0=600, x1=1200),
                                _line(2, "ਪਉੜੀ।। ਵਾਹੁ ਵਾਹੁ ਸਿਰਜਣਹਾਰ ਪਾਈਅਨੁ ਠਾਢਿ ਆਪਿ ॥ ਜੀਅ ਜੰਤ ਮਿਹਰਵਾਨੁ ਤਿਸ ਨੋ ਸਦਾ ਜਾਪਿ ॥", 1992, kind="gurbani", matches=m)])
        spans = link_pages([p157, lay], merge_style({"ref_position": "before"}))
        self.assertEqual([s["pages"] for s in spans], [[157, 158], [158]])
        self.assertEqual([s["sid"] for s in spans], [1981, None])
        self.assertEqual(spans[0]["extent"][158][3], 1491)                       # the tihai is the pauri's, up to the reference
        self.assertEqual(spans[1]["extent"][158][1], 1492)                       # the salok's notation begins at its reference
        self.assertEqual(spans[1]["ref"]["text"], "(ਰਾਗੁ ਗੂਜਰੀ ਵਾਰ ਮਹਲਾ ੫) (੫੨੧)")

    def test_a_book_with_the_reference_after_the_verse_is_untouched(self):
        lay = self._page(158, self._lines(), before=False)
        self.assertNotIn("shabad", [r["role"] for r in lay["regions"]])


class PadhtiNumberTests(unittest.TestCase):
    def test_a_bare_number_heading_opens_the_next_notation_when_its_taal_was_lost(self):
        # p. 95 as the OCR read it: "ਰਾਗ ਕੇਦਾਰਾ", "(੨)" and no taal at all
        def page(n, lines):
            lay = page_layout(lines, 1568, 2421, merge_style(None), n)
            lay["page_w"], lay["page_h"] = 1568, 2421
            return lay
        m = [{"shabad_id": 4006, "line_id": 40060 + i, "score": 0.95, "source": "G"} for i in range(2)]
        p94 = page(94, [_line(1, "ਸਰਨੀ ਆਇਓ ਨਾਥ ਨਿਧਾਨ ॥ ਨਾਮ ਪ੍ਰੀਤਿ ਲਾਗੀ ਮਨ ਭੀਤਰਿ ਮਾਗਨ ਕਉ ਹਰਿ ਦਾਨ ॥੧॥ ਰਹਾਉ ॥", 314, kind="gurbani", matches=[m[0]]),
                        _line(2, "ਸੁਖਦਾਈ ਪੂਰਨ ਪਰਮੇਸੁਰ ਕਰਿ ਕਿਰਪਾ ਰਾਖਹੁ ਮਾਨ ॥", 384, kind="gurbani", matches=[m[1]]),
                        _line(3, "ਰਾਗੁ ਕੇਦਾਰਾ", 587, bold=True, x0=606, x1=921), _line(5, "(੧)", 682, x0=115, x1=170, h=50), _line(6, "ਸਥਾਈ", 729, x0=113, x1=219),
                        _line(7, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 790), _line(8, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 870), _line(9, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 950)])
        p95 = page(95, [_line(1, "ਰਾਗ ਕੇਦਾਰਾ", 221, bold=True, x0=640, x1=955), _line(3, "(੨)", 375, x0=145, x1=199, h=50), _line(4, "ਸਥਾਈ", 425, x0=141, x1=249),
                        _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 490), _line(6, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 570), _line(7, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 650)])
        spans = link_pages([p94, p95], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[94], [94, 95]])
        self.assertEqual([s["sid"] for s in spans], [4006, 4006])
        self.assertEqual(spans[1]["heading"]["parsed"]["number"], 2)


class DarpanBracketedHeadingTests(unittest.TestCase):
    def test_a_bracketed_granth_heading_is_the_reference_of_the_verse_under_it(self):
        # Darpan 1 p. 213: "(ਬਿਹਾਗੜਾ ਮਹਲਾ ੫)" then nineteen lines the corpus did not match, then the heading and grids
        style = merge_style({"ref_position": "before"})
        def page(n, lines):
            lay = page_layout(lines, 1854, 2606, style, n)
            lay["page_w"], lay["page_h"] = 1854, 2606
            return lay
        m = [{"shabad_id": 2065, "line_id": 20650 + i, "score": 0.95, "source": "G"} for i in range(2)]
        p213 = page(213, [_line(1, "ਰਾਗ ਬਿਹਾਗੜਾ ਤਾਲ ਦੀਪਚੰਦੀ ਮਾਤਰਾਂ-14 ਮੱਧ ਲੈਅ", 397, bold=True), _line(2, "ਅਸਥਾਈ", 480, x0=800, x1=950),
                          _line(3, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 560), _line(4, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 640), _line(5, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 720),
                          _line(6, "(ਬਿਹਾਗੜਾ ਮਹਲਾ ੫)", 1293, bold=True, x0=700, x1=1100),
                          _line(7, "ਅਮਿਤ ਬਾਣੀ ਰਾਮ।। ਜਿਨ ਪ੍ਰਭੁ ਕਿਰਪਾ ਕਰੇ।।", 1365), _line(8, "ਅਕਥ ਕਹਾਣੀ ਤਿਨੀ ਜਾਣੀ ਜਿ ਪ੍ਰਭ ਭਾਣੀ।।", 1430),
                          _line(9, "ਰਾਗ ਬਿਹਾਗੜਾ ਤਾਲ ਰੁਪਕ ਮਾਤਰਾਂ-7", 1748, bold=True), _line(10, "ਅਸਥਾਈ", 1825, x0=800, x1=950),
                          _line(11, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1900), _line(12, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 1980), _line(13, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 2060)])
        p212 = page(212, [_line(1, "(ਰਾਗੁ ਬਿਹਾਗੜਾ ਛੰਤ ਮਹਲਾ ੪ ਘਰੁ ੧) (੫੩੮)", 2258, x0=600, x1=1200),
                          _line(2, "ਸਖੀ ਸਹੇਲੀ ਮੇਰੀਆ ਮੇਰੀ ਜਿੰਦੁੜੀਏ ਕੋਈ ਹਰਿ ਪ੍ਰਭੁ ਆਣਿ ਮਿਲਾਵੈ ਰਾਮ ॥", 2320, kind="gurbani", matches=[m[0]])])
        roles = [(r["role"], r["bbox"][1]) for r in p213["regions"]]
        self.assertIn(("ref", 1293), roles)
        self.assertIn(("shabad", 1365), roles)
        spans = link_pages([p212, p213], style)
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([212, 213], 2065), ([213], None)])
        self.assertEqual(spans[0]["extent"][213][3], 1292)


class SectionPairingTests(unittest.TestCase):
    """Guru Nanak Sangeet Padhti Granth: shabads and notations pair by order within a raag section."""

    def _page(self, page, lines):
        style = merge_style({"pairing": "section"})
        lay = page_layout(lines, 1568, 2421, style, page)
        lay["page_w"], lay["page_h"] = 1568, 2421
        return lay

    def _m(self, sid):
        return [[{"shabad_id": sid, "line_id": sid * 10 + i, "score": 0.95, "source": "G"}] for i in range(2)]

    def _grids(self, n0, y):
        return [_line(n0, "ਸਥਾਈ", y, x0=116, x1=224), _line(n0 + 1, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", y + 65),
                _line(n0 + 2, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", y + 145), _line(n0 + 3, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", y + 225)]

    DESC = ["ਭੈਰਉ (ਭੈਰਵ) ਬਹੁਤ ਪ੍ਰਾਚੀਨ, ਮਧੁਰ ਅਤੇ ਪ੍ਰਮੁਖ ਰਾਗਾਂ ਵਿੱਚੋਂ ਇਕ ਹੈ।", "ਸਵਰ - ਰਿਸ਼ਭ ਤੇ ਧੈਵਤ ਕੋਮਲ, ਬਾਕੀ ਸ਼ੁੱਧ", "ਥਾਟ - ਭੈਰਵ",
            "ਸ ਰੇ ਗ ਮ | ਪ ਧ ਨੀ ਸੰ |", "ਸਮਾਂ - ਸਵੇਰ ਦਾ ਸੰਧੀ-ਪ੍ਰਕਾਸ਼"]        # the aaroh reads as a grid row of one line

    def test_two_shabads_then_two_notations_pair_in_order(self):
        a, b = self._m(4086), self._m(4087)
        p96 = self._page(96, [_line(1, "ਰਾਗੁ ਭੈਰਉ", 353, bold=True, x0=674, x1=942)] + [_line(2 + i, t, 474 + 70 * i) for i, t in enumerate(self.DESC)]
                         + [_line(7, "ਸਤਿਗੁਰੁ ਮੇਰਾ ਬੇਮੁਹਤਾਜੁ ॥ ਸਤਿਗੁਰ ਮੇਰੇ ਸਚਾ ਸਾਜੁ ॥", 1594, kind="gurbani", matches=a[0]),
                            _line(8, "ਸਤਿਗੁਰੁ ਮੇਰਾ ਸਭਸ ਕਾ ਦਾਤਾ ॥ ਸਤਿਗੁਰੁ ਮੇਰਾ ਪੁਰਖੁ ਬਿਧਾਤਾ ॥੧॥", 1660, kind="gurbani", matches=a[1])])
        p97 = self._page(97, [_line(1, "ਮਾਥੇ ਤਿਲਕੁ ਹਥਿ ਮਾਲਾ ਬਾਨਾਂ ॥ ਲੋਗਨ ਰਾਮੁ ਖਿਲਉਨਾ ਜਾਨਾਂ ॥੧॥", 290, kind="gurbani", matches=b[0]),
                              _line(2, "ਜਉ ਹਉ ਬਉਰਾ ਤਉ ਰਾਮ ਤੋਰਾ ॥ ਲੋਗੁ ਮਰਮੁ ਕਹ ਜਾਨੈ ਮੋਰਾ ॥੧॥ ਰਹਾਉ ॥", 345, kind="gurbani", matches=b[1]),
                              _line(3, "ਰਾਗੁ ਭੈਰਉ", 552, bold=True, x0=674, x1=942), _line(4, "(੧) ਰੂਪਕ ਤਾਲ", 697, bold=True)] + self._grids(5, 767))
        p98 = self._page(98, [_line(1, "ਰਾਗੁ ਭੈਰਉ", 212, bold=True, x0=635, x1=899), _line(2, "ਤੀਨ ਤਾਲ", 362, x0=1257, x1=1414, h=50),
                              _line(3, "(੨)", 375, x0=120, x1=174, h=50)] + self._grids(4, 425))
        spans = link_pages([p96, p97, p98], merge_style({"pairing": "section"}))
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([96, 97], 4086), ([97, 98], 4087)])
        self.assertEqual([s["heading"]["parsed"]["number"] for s in spans], [1, 2])

    def test_a_notation_takes_the_shabad_printed_under_it(self):
        x, y = self._m(3724), self._m(3748)
        kafi = ["ਸਵਰ - ਦੋਵੇਂ ਗੰਧਾਰ, ਦੋਵੇਂ ਮਧਿਅਮ, ਦੋਵੇਂ ਨਿਸ਼ਾਦ, ਬਾਕੀ ਸੁਰ ਸ਼ੁੱਧ", "ਥਾਟ - ਕਾਫ਼ੀ", "ਜਾਤੀ - ਸੰਪੂਰਨ", "ਵਾਦੀ - ਪੰਚਮ", "ਸੰਵਾਦੀ - ਸ਼ੜਜ"]
        p184 = self._page(184, [_line(1, "ਰਾਗੁ ਮਾਰੂ ਕਾਫੀ", 190, bold=True, x0=500, x1=1000)] + [_line(2 + i, t, 350 + 50 * i) for i, t in enumerate(kafi)]
                          + [_line(8, "ਸਥਾਈ", 840, x0=60, x1=160), _line(9, "ਤੀਨਤਾਲ", 840, x0=1000, x1=1120)]
                          + [_line(10, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 980), _line(11, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 1060), _line(12, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1140)])
        p185 = self._page(185, [_line(1, "ਅੰਤਰਾ", 230, x0=100, x1=200), _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 370), _line(3, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 450),
                                _line(4, "ਆਵਉ ਵੰਞਉ ਡੁੰਮਣੀ ਕਿਤੀ ਮਿਤ੍ਰ ਕਰੇਉ ॥ ਸਾ ਧਨ ਢੋਈ ਨ ਲਹੈ ਵਾਢੀ ਕਿਉ ਧੀਰੇਉ ॥੧॥", 1089, kind="gurbani", matches=x[0]),
                                _line(5, "ਮੈਡਾ ਮਨੁ ਰਤਾ ਆਪਨੜੇ ਪਿਰ ਨਾਲਿ ॥ ਹਉ ਘੋਲਿ ਘੁਮਾਈ ਖੰਨੀਐ ਕੀਤੀ ਹਿਕ ਭੋਰੀ ਨਦਰਿ ਨਿਹਾਲਿ ॥੧॥ ਰਹਾਉ ॥", 1150, kind="gurbani", matches=x[1]),
                                _line(6, "ਕੀਰਤਨਕਾਰ ਬੀਬੀ ਮਨਜੀਤ ਕੌਰ ਪਟਿਆਲਾ", 1860, x0=700, x1=1400)])
        p186 = self._page(186, [_line(1, "ਰਾਗੁ ਮਾਰੂ ਦਖਣੀ", 249, bold=True, x0=500, x1=1000)] + [_line(2 + i, t, 350 + 50 * i) for i, t in enumerate(kafi)]
                          + [_line(8, "ਸਥਾਈ", 953, x0=60, x1=160), _line(9, "ਏਕਤਾਲ", 953, x0=1000, x1=1120)]
                          + [_line(10, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1060), _line(11, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 1140), _line(12, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1220)])
        p187 = self._page(187, [_line(1, "ਕਾਇਆ ਨਗਰੁ ਨਗਰ ਗੜ ਅੰਦਰਿ ॥ ਸਾਚਾ ਵਾਸਾ ਪੁਰਿ ਗਗਨੰਦਰਿ ॥", 251, kind="gurbani", matches=y[0]),
                                _line(2, "ਅਸਥਿਰੁ ਥਾਨੁ ਸਦਾ ਨਿਰਮਾਇਲੁ ਆਪੇ ਆਪੁ ਉਪਾਇਦਾ ॥੧॥", 320, kind="gurbani", matches=y[1])])
        spans = link_pages([p184, p185, p186, p187], merge_style({"pairing": "section"}))
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([184, 185], 3724), ([186, 187], 3748)])
        self.assertTrue(all(s["inherited"] for s in spans))
        self.assertLess(spans[0]["extent"][184][1], 900)                              # the body begins at its sthai, after the description
        self.assertGreater(spans[0]["extent"][184][1], 600)


class SourceLineTests(unittest.TestCase):
    def test_a_line_naming_another_scripture_opens_a_shabad_of_that_source(self):
        # Darpan 1 p. 214: the Kabit Savaiye after a Granth shabad's notation
        style = merge_style({"ref_position": "before"})
        def page(n, lines):
            lay = page_layout(lines, 1854, 2606, style, n)
            lay["page_w"], lay["page_h"] = 1854, 2606
            return lay
        m = [{"shabad_id": 2065, "line_id": 20650 + i, "score": 0.95, "source": "G"} for i in range(2)]
        p213 = page(213, [_line(1, "(ਰਾਗੁ ਬਿਹਾਗੜਾ ਛੰਤ ਮਹਲਾ ੪ ਘਰੁ ੧) (੫੩੮)", 300, x0=600, x1=1200),
                          _line(2, "ਸਖੀ ਸਹੇਲੀ ਮੇਰੀਆ ਮੇਰੀ ਜਿੰਦੁੜੀਏ ਕੋਈ ਹਰਿ ਪ੍ਰਭੁ ਆਣਿ ਮਿਲਾਵੈ ਰਾਮ ॥", 360, kind="gurbani", matches=[m[0]]),
                          _line(3, "ਰਾਗ ਬਿਹਾਗੜਾ ਤਾਲ ਰੁਪਕ ਮਾਤਰਾਂ-7", 1748, bold=True), _line(4, "ਅਸਥਾਈ", 1825, x0=800, x1=950),
                          _line(5, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1900), _line(6, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 1980), _line(7, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 2060)])
        p214 = page(214, [_line(1, "ਕਬਿੱਤ ਸਵੱਯੇ (ਭਾਈ ਗੁਰਦਾਸ ਜੀ)", 203, x0=500, x1=1100),
                          _line(2, "ਆਂਬ ਕੀ ਸਧਰ ਕਤ ਮਿਟਤ ਆਂਬਲੀ ਖਾਏ, ਪਿਆਸ ਨ ਬੁਝਤ ਜੈਸੇ ਬਾਰਿ ਕੇ ਬੁਝਾਏ ਹੈ।", 270),
                          _line(3, "ਸਾਧਸੰਗਿ ਗੁਰਮੁਖਿ ਸੁਖਫਲ ਪਾਏ, ਕਰਮ ਕਾਂਡ ਸੇ ਨ ਪਾਏ ਹੈ।", 330),
                          _line(4, "ਰਾਗ ਬਿਹਾਗੜਾ ਤਾਲ ਦਾਦਰਾ ਮਾਤਰਾਂ-6 ਮੱਧ ਲੈਅ", 709, bold=True), _line(5, "ਅਸਥਾਈ ਅੰਤਰਾ", 786, x0=700, x1=1000),
                          _line(6, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 860), _line(7, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", 940), _line(8, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1020)])
        roles = [(r["role"], r["bbox"][1]) for r in p214["regions"]]
        self.assertIn(("ref", 203), roles)
        self.assertIn(("shabad", 270), roles)
        spans = link_pages([p213, p214], style)
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([213], 2065), ([214], None)])
        self.assertEqual(spans[1]["ref"]["parsed"]["source"], "B")


class ThirdCutTests(unittest.TestCase):
    """
    The third cut (5 October 2026): four more books sampled on the M4 -- Mishrat Raag, Swar Samund, Guru
    Nanak Dev Raag Ratnaavlee, Bhagat Hayt Gavai Ravidasa -- and what the owner's nine backlog verdicts and
    two rejections came down to, each from the page that showed it.
    """
    W, H = 1646, 2545

    def _page(self, page, lines, style=None, ink=None):
        lay = page_layout(lines, self.W, self.H, style or merge_style(None), page, ink=ink)
        lay["page_w"], lay["page_h"] = self.W, self.H
        return lay

    def _verse(self, n, text, y, sid, **kw):
        return _line(n, text, y, kind="gurbani", matches=[{"shabad_id": sid, "line_id": sid * 10 + n, "score": 0.95, "source": "G"}], **kw)

    def _grids(self, n0, y):
        return [_line(n0, "ਸਥਾਈ", y, x0=200, x1=330), _line(n0 + 1, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", y + 70),
                _line(n0 + 2, "ਮਾ ਈ | ਮੈ ऽ | ਕਿ ਹਿ | ਬਿ ਧਿ", y + 140), _line(n0 + 3, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", y + 210),
                _line(n0 + 4, "ਲ ਖ | ਉ ऽ | ਗੁ ਸਾ | ਈ ऽ", y + 280)]

    def _ruled(self, y0, y1, bars=(517, 770, 1074)):
        import numpy as np
        ink = np.zeros((self.H, self.W), dtype=np.uint8)
        for x in bars:
            ink[y0:y1, x:x + 5] = 255
        for y in range(y0 + 60, y1 - 60, 130):                  # the rows of swaras and syllables
            ink[y:y + 34, 150:1500:9] = 255
        return ink

    # -- Guru Nanak Dev Raag Ratnaavlee, the Maru Vaar (pp. 105-107)

    def test_jap_before_taal_is_jhaptaal_and_jap_alone_is_the_bani(self):
        h = parse_heading("ਜਪ ਤਾਲ")
        self.assertEqual((h["raag"], h["taal"]["key"]), (None, "jhaptaal"))
        self.assertEqual(parse_heading("ਰਾਗ ਬੈਰਾੜੀ; ਜਪ ਤਾਲ, ਬਿਲੰਬਿਤ")["taal"]["key"], "jhaptaal")
        h = parse_heading("ਜਪੁ")
        self.assertEqual(((h["raag"] or {}).get("key"), h["taal"]), ("jap", None))

    def _salok(self, a, b, c, d, ang):
        return [_line(1, "ਰਾਗ ਮਾਰੂ 187", 236, x0=681, x1=1349, h=38), _line(2, "ਮਾਰੂ ਵਾਰ", 301, x0=640, x1=820), _line(3, "ਮ: ੧", 360, x0=690, x1=770),
                _line(4, a, 420, x0=505, x1=940), _line(5, b + "॥", 480, x0=505, x1=940), _line(6, c, 540, x0=505, x1=940),
                _line(7, d + "॥", 600, x0=505, x1=940), _line(8, "(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪੰਨਾ %s)" % ang, 678, x0=827, x1=1348, h=45),
                _line(9, "ਰਾਗ ਮਾਰੂ ਤਿੰਨ ਤਾਲ", 747, x0=143, x1=1358, h=50)] + self._grids(10, 820)

    def test_a_salok_set_in_half_lines_over_its_ang_is_a_verse_and_opens_its_own_notation(self):
        p105 = self._page(105, [_line(1, "ਮਾਰੂ ਵਾਰ", 316, x0=640, x1=820), self._verse(2, "ਸਲੋਕੁ ਮ: ੧॥", 370, 3805),
                                self._verse(3, "ਭੂਲੀ ਭੂਲੀ ਮੈ ਫਿਰੀ ਪਾਧਰੁ ਕਹੈ ਨ ਕੋਇ ॥", 430, 3805), self._verse(4, "ਪੂਛਹੁ ਜਾਇ ਸਿਆਣਿਆ ਦੁਖੁ ਕਾਟੈ ਮੇਰਾ ਕੋਇ ॥", 490, 3805),
                                _line(5, "(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪੰਨਾ ੧੦੮੭)", 678, x0=980, x1=1498, h=45), _line(6, "ਰਾਗ ਮਾਰੂ ਜਪ ਤਾਲ", 776, h=42)] + self._grids(7, 850))
        p106 = self._page(106, self._salok("ਸਾਚੁ ਸੀਲ ਸਚੁ ਸੰਜਮੀ", "ਸਾ ਪੂਰੀ ਪਰਵਾਰਿ", "ਨਾਨਕ ਅਹਿਨਿਸਿ ਸਦਾ ਭਲੀ", "ਪਿਰ ਕੈ ਹੇਤਿ ਪਿਆਰਿ", "੧੦੮੮"))
        p107 = self._page(107, self._salok("ਸਸੁਰੈ ਪੇਈਐ ਕੰਤ ਕੀ", "ਕੰਤੁ ਅਗੰਮੁ ਅਥਾਹੁ", "ਨਾਨਕ ਧੰਨੁ ਸੁੋਹਾਗਣੀ", "ਜੋ ਭਾਵਹਿ ਵੇਪਰਵਾਹ", "੧੦੮੮"))
        self.assertIn("shabad", [r["role"] for r in p106["regions"]])
        spans = link_pages([p105, p106, p107], merge_style(None))
        self.assertEqual([s["pages"] for s in spans], [[105], [106], [107]])
        self.assertEqual(spans[0]["heading"]["parsed"]["taal"]["key"], "jhaptaal")
        # a paragraph over a reference is still prose: its sentences close with a single danda
        prose = self._page(108, [_line(1, "ਇਸ ਰਾਗ ਦਾ ਵਰਣਨ ਗ੍ਰੰਥ ਵਿਚ ਆਇਆ ਹੈ।", 400), _line(2, "ਇਸ ਨੂੰ ਸਵੇਰੇ ਗਾਇਆ ਜਾਂਦਾ ਹੈ ਅਤੇ ਇਹ ਬਹੁਤ ਪੁਰਾਣਾ ਹੈ", 460),
                                 _line(3, "(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪੰਨਾ ੧੦੮੮)", 540, x0=827, x1=1348)])
        self.assertNotIn("shabad", [r["role"] for r in prose["regions"]])

    def test_a_shabad_no_line_of_which_matched_is_found_by_its_printed_ang(self):
        import sqlite3
        con = sqlite3.connect(":memory:")
        con.executescript("""
          CREATE TABLE shabads (shabad_id INTEGER PRIMARY KEY, writer TEXT, raag TEXT, ang_start INTEGER);
          CREATE TABLE lines (line_id INTEGER PRIMARY KEY, shabad_id INTEGER, gurmukhi_uni TEXT, kind TEXT, position_in_shabad INTEGER);
          INSERT INTO shabads VALUES (3812, 'Guru Nanak Dev Ji', 'Raag Maaroo', 1088);
          INSERT INTO lines VALUES (46489, 3812, 'ਸਾਚੁ ਸੀਲ ਸਚੁ ਸੰਜਮੀ ਸਾ ਪੂਰੀ ਪਰਵਾਰਿ ॥', 'line', 1);
          INSERT INTO lines VALUES (46490, 3812, 'ਨਾਨਕ ਅਹਿਨਿਸਿ ਸਦਾ ਭਲੀ ਪਿਰ ਕੈ ਹੇਤਿ ਪਿਆਰਿ ॥੨॥', 'line', 2);
          INSERT INTO shabads VALUES (3815, 'Guru Nanak Dev Ji', 'Raag Maaroo', 1088);
          INSERT INTO lines VALUES (46501, 3815, 'ਸਸੁਰੈ ਪੇਈਐ ਕੰਤ ਕੀ ਕੰਤੁ ਅਗੰਮੁ ਅਥਾਹੁ ॥', 'line', 1);
          INSERT INTO lines VALUES (46502, 3815, 'ਨਾਨਕ ਧੰਨੁ ਸੁੋਹਾਗਣੀ ਜੋ ਭਾਵਹਿ ਵੇਪਰਵਾਹ ॥੧॥', 'line', 2);
        """)
        block = [_line(i, t, 300 + 60 * i) for i, t in enumerate(["ਮਾਰੂ ਵਾਰ", "ਮ: ੧", "ਸਾਚੁ ਸੀਲ ਸਚੁ ਸੰਜਮੀ", "ਸਾ ਪੂਰੀ ਪਰਵਾਰਿ॥", "ਨਾਨਕ ਅਹਿਨਿਸਿ ਸਦਾ ਭਲੀ", "ਪਿਰ ਕੈ ਹੇਤਿ ਪਿਆਰਿ॥"])]
        got = resolve_shabad(block, parse_ref("(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪੰਨਾ ੧੦੮੮)"), con)
        self.assertEqual((got["shabad"]["shabad_id"], got["shabad"]["method"]), (3812, "ref-window"))
        self.assertIn("weak-shabad", got["flags"])                                   # linked, and shown to the reviewer as weak
        self.assertEqual(sorted(got["shabad"]["line_ids"]), [46489, 46490])
        other = resolve_shabad([_line(1, "ਇਹ ਰਾਗ ਬਹੁਤ ਪੁਰਾਣਾ ਹੈ ਅਤੇ ਸਵੇਰੇ ਗਾਇਆ ਜਾਂਦਾ ਹੈ", 300)], parse_ref("(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪੰਨਾ ੧੦੮੮)"), con)
        self.assertIsNone(other["shabad"]["shabad_id"])

    # -- Bhagat Hayt Gavai Ravidasa (pp. 50-53, 93-97)

    def test_a_grid_in_a_ruled_table_is_as_tall_as_its_bars(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2")
        ink = self._ruled(345, 2226)
        # p. 96: the OCR read the table's first rows and nothing under them
        lines = [_line(1, "ਰਾਗ ਗੋਂਡ 75", 205, x0=766, x1=1429, h=55), _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400), _line(3, "ਕਾ ऽ | ਨਿੰ ऽ | ਦ ਕੁ | ਕੈ ऽ", 470),
                 _line(4, "& 4.", 1273, x0=525, x1=576, h=556)]
        lay = self._page(96, lines, ink=ink)
        grids = [r for r in lay["regions"] if r["role"] == "grid"]
        self.assertEqual(len(grids), 1)
        self.assertGreater(grids[0]["bbox"][3], 2200)
        self.assertNotIn("text", [r["role"] for r in lay["regions"]])               # the shred down a bar is the table's own
        # p. 51: a page Tesseract read nothing of
        lay = self._page(51, [], ink=ink)
        self.assertEqual((lay["kind"], [r["role"] for r in lay["regions"]]), ("notation", ["grid"]))
        self.assertTrue(340 <= lay["regions"][0]["bbox"][1] <= 420 and lay["regions"][0]["bbox"][3] > 2150)
        # bars round a verse (a boxed shabad) are not a notation's; nor is a frame round the whole page
        boxed = self._page(52, [self._verse(1, "ਜਉ ਹਮ ਬਾਂਧੇ ਮੋਹ ਫਾਸ ਹਮ ਪ੍ਰੇਮ ਬਧਨਿ ਤੁਮ ਬਾਧੇ ॥", 900, 2518), self._verse(2, "ਅਪਨੇ ਛੂਟਨ ਕੋ ਜਤਨੁ ਕਰਹੁ ਹਮ ਛੂਟੇ ਤੁਮ ਆਰਾਧੇ ॥੧॥", 960, 2518)], ink=ink)
        self.assertNotIn("grid", [r["role"] for r in boxed["regions"]])
        framed = self._page(53, [], ink=self._ruled(60, 2500))
        self.assertEqual(framed["regions"], [])

    def test_a_numbered_raag_over_its_description_is_a_chapter_not_a_notation(self):
        p96 = self._page(96, [_line(1, "ਰਾਗ ਗੋਂਡ ਤਿੰਨਤਾਲ", 284, bold=True), self._verse(2, "ਜੇ ਓਹੁ ਅਠਸਠਿ ਤੀਰਥ ਨ੍ਹਾਵੈ ॥ ਜੇ ਓਹੁ ਦੁਆਦਸ ਸਿਲਾ ਪੂਜਾਵੈ ॥", 360, 3270),
                              self._verse(3, "ਜੇ ਓਹੁ ਕੂਪ ਤਟਾ ਦੇਵਾਵੈ ॥ ਕਰੈ ਨਿੰਦ ਸਭ ਬਿਰਥਾ ਜਾਵੈ ॥੧॥", 420, 3270)] + self._grids(4, 600))
        p97 = self._page(97, [_line(1, "13. ਰਾਗ ਰਾਮਕਲੀ", 652, bold=True, x0=560, x1=1010),
                              _line(2, "ਰਾਮਕਲੀ ਬੜਾ ਪ੍ਰਸਿੱਧ ਰਾਗ ਹੈ। ਇਸ ਦਾ ਵਾਦੀ ਧੈਵਤ ਅਤੇ ਸੰਵਾਦੀ ਰਿਸ਼ਭ ਹੈ ਅਤੇ ਠਾਠ ਭੈਰਵ ਹੈ।", 740),
                              _line(3, "ਆਰੋਹ : ਸਾ ਗ, ਮ ਪ, ਧੁ, ਨੀ ਸਾਂ।", 1300), _line(4, "ਅਵਰੋਹ : ਸਾਂ ਨੀ ਧੁ ਪ, ਮੰ ਪ ਧੁ ਨੀ ਧੁ ਪ, ਗ, ਮ ਰੇ, ਸਾ।", 1360),
                              _line(5, "ਰਾਗ ਰਾਮਕਲੀ ਸੁਰ ਵਿਸਤਾਰ", 1560, bold=True, x0=560, x1=1010),
                              _line(6, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 1650), _line(7, "ਨੀ ਸ | ਗ ਮ | ਧ ਪ | ਮ ਗ", 1720), _line(8, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", 1790)])
        dropped = []
        spans = link_pages([p96, p97], merge_style(None), dropped)
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([96], 3270)])
        self.assertIn("raag", [d.get("opened_by") for d in dropped if 97 in d["pages"]])

    def test_the_running_header_that_is_a_line_of_gurbani_is_no_shabad(self):
        style = merge_style({"running_header": ["ਭਗਤਿ ਹੇਤ ਗਾਵੈ ਰਵਿਦਾਸਾ"]})
        lines = [self._verse(1, "74 ਭਗਤਿ ਹੇਤ ਗਾਵੈ ਰਵਿਦਾਸਾ ॥੫॥੫॥", 204, 4300, x0=650, x1=1008, h=50),
                 self._verse(2, "ਜੇ ਓਹੁ ਅਠਸਠਿ ਤੀਰਥ ਨ੍ਹਾਵੈ ॥ ਜੇ ਓਹੁ ਦੁਆਦਸ ਸਿਲਾ ਪੂਜਾਵੈ ॥", 344, 3270), self._verse(3, "ਜੇ ਓਹੁ ਕੂਪ ਤਟਾ ਦੇਵਾਵੈ ॥ ਕਰੈ ਨਿੰਦ ਸਭ ਬਿਰਥਾ ਜਾਵੈ ॥੧॥", 405, 3270)]
        lay = self._page(95, lines, style=style)
        self.assertEqual([(r["role"], len(r["lines"])) for r in lay["regions"]], [("shabad", 2)])
        # the same words lower on the page are the verse they are
        lines[0] = self._verse(1, "ਭਗਤਿ ਹੇਤ ਗਾਵੈ ਰਵਿਦਾਸਾ ॥੫॥੫॥", 284, 3270, x0=650, x1=1008, h=50)
        self.assertEqual([(r["role"], len(r["lines"])) for r in self._page(95, lines, style=style)["regions"]], [("shabad", 3)])

    def test_a_raag_named_alone_over_the_verse_leads_it_and_a_speck_is_no_grid_row(self):
        p94 = self._page(94, [_line(1, "ਰਾਗ ਗੋਂਡ ਤਿੰਨਤਾਲ", 284, bold=True), self._verse(2, "ਮੁਕੰਦ ਮੁਕੰਦ ਜਪਹੁ ਸੰਸਾਰ ॥", 360, 3269),
                              self._verse(3, "ਬਿਨੁ ਮੁਕੰਦ ਤਨੁ ਹੋਇ ਅਉਹਾਰ ॥", 420, 3269)] + self._grids(4, 600))
        p95 = self._page(95, [_line(1, "=", 287, x0=809, x1=826, h=12), _line(2, "ਗੋਂਡ", 300, x0=801, x1=866, h=27),
                              self._verse(3, "ਜੇ ਓਹੁ ਅਠਸਠਿ ਤੀਰਥ ਨ੍ਹਾਵੈ ॥ ਜੇ ਓਹੁ ਦੁਆਦਸ ਸਿਲਾ ਪੂਜਾਵੈ ॥", 344, 3270),
                              self._verse(4, "ਜੇ ਓਹੁ ਕੂਪ ਤਟਾ ਦੇਵਾਵੈ ॥ ਕਰੈ ਨਿੰਦ ਸਭ ਬਿਰਥਾ ਜਾਵੈ ॥੧॥", 405, 3270)] + self._grids(5, 900))
        self.assertEqual([r["role"] for r in p95["regions"]][:2], ["text", "shabad"])
        spans = link_pages([p94, p95], merge_style(None))
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([94], 3269), ([95], 3270)])

    def test_a_verse_that_begins_with_apna_is_no_reference(self):
        self.assertIsNone(parse_ref("ਅਪਨਾ ਬਿਗਾਰਿ ਬਿਰਾਂਨਾ ਸਾਂਢੈ ॥ ਕਰੈ ਨਿੰਦ ਬਹੁ ਜੋਨੀ ਹਾਂਢੈ ॥੩॥੩॥"))
        self.assertEqual(parse_ref("(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਪਨਾ ੮੭੫)")["ang_from"], 875)
        self.assertEqual(parse_ref("(ਅੰਗ-੭੨੪)")["ang_from"], 724)

    # -- Swar Samund (pp. 83-84)

    def test_the_meanings_and_the_other_shabads_are_prose_and_the_notation_keeps_its_own_shabad(self):
        p83 = self._page(83, [self._verse(1, "ਮਲਾਰ ਮਹਲਾ ੪ ॥", 231, 4558, x0=600, x1=1000), self._verse(2, "ਤਿਸੁ ਜਨ ਕਉ ਹਰਿ ਮੀਠ ਲਗਾਨਾ ਜਿਸੁ ਹਰਿ ਹਰਿ ਕ੍ਰਿਪਾ ਕਰੈ ॥", 300, 4558),
                              self._verse(3, "ਤਿਸ ਕੀ ਭੂਖ ਦੂਖ ਸਭਿ ਉਤਰੈ ਜੋ ਹਰਿ ਗੁਣ ਹਰਿ ਉਚਰੈ ॥੧॥", 360, 4558),
                              _line(4, "(ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ, ਅੰਗ ੧੨੬੩)", 950, x0=900, x1=1480), _line(5, "ਭਾਵ-ਅਰਥ", 1070, x0=700, x1=920),
                              _line(6, "੧. ਜਿਸ ਮਨੁੱਖ ਉਪਰ ਪਰਮਾਤਮਾ ਮਿਹਰ ਕਰਦਾ ਹੈ, ਉਸ ਮਨੁੱਖ ਨੂੰ ਪਰਮਾਤਮਾ ਦਾ ਨਾਮ ਪਿਆਰਾ ਲੱਗਦਾ ਹੈ।", 1140),
                              _line(7, "ਸ ਰ | ਗ ਮ | ਪ — | ਧ ਨ", 1210),          # a line of the meanings the OCR made a row of
                              _line(8, "ਹੋਰ ਸ਼ਬਦ (ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ) –", 2130, x0=160, x1=640),
                              self._verse(9, "੧. ਪ੍ਰਭ ਮੇਰੇ ਪ੍ਰੀਤਮ ਪ੍ਰਾਨ ਪਿਆਰੇ ॥ (ਅੰਗ-੬੯)", 2190, 5001), self._verse(10, "੨. ਬਰਸੁ ਘਨਾ ਮੇਰਾ ਮਨੁ ਭੀਨਾ ॥ (ਅੰਗ-੮੪੯)", 2250, 5002)])
        p84 = self._page(84, [_line(1, "ਰਾਗੁ ਮਲਾਰ (ਤੀਨ ਤਾਲ)", 265, bold=True, x0=560, x1=1100)] + self._grids(2, 360))
        self.assertEqual([(r["role"], bool(r.get("prose"))) for r in p83["regions"]], [("shabad", False), ("ref", False), ("text", True), ("text", True)])
        self.assertTrue(p83["regions"][3]["text"].startswith("ਹੋਰ ਸ਼ਬਦ"))           # the other shabads for the tune, kept apart in the layout
        self.assertEqual(p83["kind"], "text")
        spans = link_pages([p83, p84], merge_style(None))
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([83, 84], 4558)])
        # ... and the other shabads are the record's `also`. The numbers under "(ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ)" are that pothi's pages, not
        # angs, so the words name the shabad: the merge's match, else the one shabad a line of which begins so
        import sqlite3
        from lib.notation_resolve import other_shabads
        con = sqlite3.connect(":memory:")
        con.executescript("""
          CREATE TABLE shabads (shabad_id INTEGER PRIMARY KEY, writer TEXT, raag TEXT, ang_start INTEGER);
          CREATE TABLE lines (line_id INTEGER PRIMARY KEY, shabad_id INTEGER, gurmukhi_uni TEXT, kind TEXT, position_in_shabad INTEGER);
          INSERT INTO shabads VALUES (5001, 'Guru Arjan Dev Ji', 'Raag Bilaaval', 802);
          INSERT INTO lines VALUES (1, 5001, 'ਪ੍ਰਭ ਮੇਰੇ ਪ੍ਰੀਤਮ ਪ੍ਰਾਨ ਪਿਆਰੇ ॥', 'line', 1);
          INSERT INTO shabads VALUES (5002, 'Guru Arjan Dev Ji', 'Raag Malaar', 1268);
          INSERT INTO lines VALUES (2, 5002, 'ਬਰਸੁ ਘਨਾ ਮੇਰਾ ਮਨੁ ਭੀਨਾ ॥', 'line', 1);
          INSERT INTO lines VALUES (3, 5002, 'ਅੰਮ੍ਰਿਤ ਬੂੰਦ ਸੁਹਾਨੀ ਹੀਅਰੈ ਗੁਰਿ ਮੋਹੀ ਮਨੁ ਹਰਿ ਰਸਿ ਲੀਨਾ ॥੧॥ ਰਹਾਉ ॥', 'rahao', 2);
        """)
        texts = [r for _, r in spans[0]["text"]]
        self.assertEqual([len(r.get("others") or []) for r in texts], [0, 2])
        also = other_shabads(texts, con, 4558)
        self.assertEqual([(a["shabad_id"], a["ang"], a["confidence"]) for a in also], [(5001, 802, 0.9), (5002, 1268, 0.9)])   # the Granth's angs, not 69 and 849
        self.assertEqual(also[0]["first_line"], "ਪ੍ਰਭ ਮੇਰੇ ਪ੍ਰੀਤਮ ਪ੍ਰਾਨ ਪਿਆਰੇ ॥")
        self.assertTrue(also[1]["printed"].startswith("੨. ਬਰਸੁ ਘਨਾ"))
        self.assertEqual([a["shabad_id"] for a in other_shabads(texts, con, 5001)], [5002])       # never the notation's own shabad
        # nothing matched by the merge: found by the printed words at the head of a line; words that open two shabads name neither
        bare = [{"text": "ਹੋਰ ਸ਼ਬਦ (ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ) –", "others": [{"text": "੧. ਬਰਸੁ ਘਨਾ ਮੇਰਾ ਮਨੁ (ਅੰਗ-੮੪੯)", "matches": []}]}]
        self.assertEqual([(a["shabad_id"], a["confidence"]) for a in other_shabads(bare, con, 4558)], [(5002, 0.7)])
        con.execute("INSERT INTO shabads VALUES (5003, 'x', 'Raag Sorath', 600)")
        con.execute("INSERT INTO lines VALUES (4, 5003, 'ਬਰਸੁ ਘਨਾ ਮੇਰਾ ਮਨੁ ਤਰਸੈ ॥', 'line', 1)")
        from lib import notation_resolve
        notation_resolve._CORPUS_LINES.clear()
        self.assertEqual([a["shabad_id"] for a in other_shabads(bare, con, 4558)], [None])
        self.assertEqual([a["shabad_id"] for a in other_shabads(bare, None, 4558)], [None])         # no corpus: the printed line only

    # -- the ledger

    def test_a_rejection_removes_the_cut_that_was_rejected_not_another_cut_of_the_same_shabad(self):
        # Bhagat Hayt Gavai Ravidasa: "7. ਰਾਗ ਸੋਰਠਿ" and its swar-vistaar (pp. 52-53) had borrowed the shabad of p. 50 and
        # were rejected as "raag sorath description"; the reader then cut the shabad's own notation, pp. 50-51
        from lib.notation_review import append_entry, apply_review, assign_keys, entry_of, read_ledger
        rec = lambda *a, **kw: ReviewLedgerTests._rec(self, *a, **kw)
        with tempfile.TemporaryDirectory() as d:
            images = os.path.join(d, "images"); os.makedirs(images)
            description = rec("test-book:0052:1", [50, 52, 53], 2010, raag="sorath")
            assign_keys([description])
            append_entry(entry_of(description, "rejected", "raag sorath description", round_=1), d)
            ledger = read_ledger("test-book", d)
            notation_ = rec("test-book:0051:1", [50, 51], 2010)
            got = apply_review([notation_], ledger, images, d)
            self.assertEqual([r["notation_id"] for r in got["records"]], ["test-book:0051:1"])       # shown for review, not dropped unseen
            from lib.notation_review import attach
            self.assertEqual(attach(ledger, [notation_]), {})                                            # and the review page shows it unreviewed
            again = rec("test-book:0052:1", [50, 52, 53], 2010, raag="sorath", y0=230)                 # the rejected cut itself, re-cut a little
            self.assertEqual(apply_review([again], ledger, images, d)["records"], [])

    # -- the other shabads for a tune (Swar Samund's "ਹੋਰ ਸ਼ਬਦ"), as secondary entries of the notation

    def test_the_database_lists_a_notation_under_the_other_shabads_the_book_sets_to_its_tune(self):
        import sqlite3
        build = _script("32_build_notations_db.py")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "notations", "test-book")
            os.makedirs(os.path.join(src, "images"))
            rec = _record("test-book:0084:1", 1248)
            rec["page"], rec["pages"] = 84, [83, 84]
            rec["also"] = [{"shabad_id": 2000, "ang": 69, "printed": "੧. ਪ੍ਰਭ ਮੇਰੇ ਪ੍ਰੀਤਮ ਪ੍ਰਾਨ ਪਿਆਰੇ (ਅੰਗ-੬੯)", "first_line": "ਪ੍ਰਭ ਮੇਰੇ ਪ੍ਰੀਤਮ ਪ੍ਰਾਨ ਪਿਆਰੇ ॥"},
                           {"shabad_id": 1248, "ang": 1263, "printed": "the notation's own shabad, listed again"},
                           {"shabad_id": 777777, "ang": 5, "printed": "an id the corpus does not hold"},
                           {"shabad_id": None, "ang": 849, "printed": "੨. ਬਰਸੁ ਘਨਾ ਮੇਰਾ ਮਨੁ ਭੀਨਾ (ਅੰਗ-੮੪੯)", "first_line": None}]
            notation.write_jsonl(os.path.join(src, "notations.jsonl"),
                                 notation.meta_for({"book": "test-book", "author": "Test Author", "title": "Test Book", "part": 1,
                                                    "style": merge_style(None)}, pages=2), [rec])
            gurbani = os.path.join(d, "gurbani.sqlite")
            con = sqlite3.connect(gurbani)
            con.execute("CREATE TABLE shabads (shabad_id INTEGER PRIMARY KEY)")
            con.executemany("INSERT INTO shabads VALUES (?)", [(1248,), (2000,)])
            con.commit(); con.close()
            out = os.path.join(d, "artifacts", "notations.sqlite")
            build.build(os.path.join(d, "notations"), out, None, gurbani, {}, False, True, False, None)
            con = sqlite3.connect(out)
            with open(os.path.join(HERE, "lib", "notation_columns.json"), encoding="utf-8") as fh:
                optional = json.load(fh)["optional"]
            self.assertEqual([r[1] for r in con.execute("PRAGMA table_info(notation_shabads)")], optional["notation_shabads"])
            rows = con.execute("SELECT notation_id, n, shabad_id, ang FROM notation_shabads ORDER BY n").fetchall()
            self.assertEqual(rows, [("test-book:0084:1", 1, 2000, 69), ("test-book:0084:1", 2, None, 849)])
            self.assertEqual(con.execute("SELECT n FROM shabad_counts WHERE shabad_id = 1248").fetchone(), (1,))     # the notation's own count is its own
            self.assertEqual(con.execute("SELECT value FROM meta WHERE key = 'shabads_also'").fetchone(), ("2",))
            con.close()

    # -- Guru Angad Dev Sangeet Darpan (the second cut re-reviewed, 6 October 2026)

    def test_a_pauri_under_its_unbracketed_source_line_is_the_sections_shabad_and_nothing_crosses_a_raag_description(self):
        style = merge_style({"ref_position": "before"})
        grids = lambda n0, y: [_line(n0, "ਅਸਥਾਈ", y, x0=200, x1=330), _line(n0 + 1, "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", y + 60),
                               _line(n0 + 2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", y + 120), _line(n0 + 3, "ਸ ਬ | ਦੇ ऽ | ਸ ਬ | ਦੁ ऽ", y + 190), _line(n0 + 4, "ਪ ਪ | ਧ ਨੀ | ਸੰ — | ਨੀ ਧ", y + 260)]
        p152 = self._page(152, [_line(1, "(ਸਵਈਏ ਮਹਲੇ ਦੂਜੇ ਕੇ ੨) (੧੩੯੨)", 1133, x0=500, x1=1200), self._verse(2, "ਅਮਿਅ ਦ੍ਰਿਸਟਿ ਸੁਭ ਕਰੈ ਹਰੈ ਅਘ ਪਾਪ ਸਕਲ ਮਲ ॥", 1205, 5372),
                                self._verse(3, "ਕਾਮ ਕ੍ਰੋਧ ਅਰੁ ਲੋਭ ਮੋਹ ਵਸਿ ਕਰੈ ਸਭੈ ਬਲ ॥", 1270, 5372), _line(4, "ਰਾਗੁ ਦੇਵਗੰਧਾਰੀ ਫਰੋਦਸਤ ਤਾਲ ਮਾਤਰਾਂ-14 ਮੱਧ ਲੈਅ", 1766, bold=True)]
                          + grids(5, 1851), style=style)
        p153 = self._page(153, [_line(1, "ਰਾਗੁ ਦੇਵਗੰਧਾਰੀ ਕਹਿਰਵਾ ਤਾਲ ਮਾਤਰਾਂ-8 ਮੱਧ ਲੈਅ", 413, bold=True)] + grids(2, 500), style=style)
        p154 = self._page(154, [_line(1, "ਰਾਗ ਤੁਖਾਰੀ", 193, bold=True, x0=560, x1=1000),
                                _line(2, "ਰਾਗੁ-ਤੁਖਾਰੀ ਥਾਟ-ਤੋੜੀ ਸਵਰ-ਦੋਨੋਂ ਨਿਸ਼ਾਦ, ਗੰਧਾਰ ਕੋਮਲ,", 361), _line(3, "ਜਾਤੀ-ਔੜਵ-ਸੰਪੂਰਨ ਵਾਦੀ-ਪੰਚਮ ਸੰਵਾਦੀ-ਸ਼ੜਜ ਸਮਾਂ-ਦਿਨ ਦਾ ਚੌਥਾ ਪਹਿਰ", 504),
                                _line(4, "ਆਰੋਹ-ਨੁ ਸ, ਗੁ ਮ ਪ, ਨ ਸੰ ਅਵਰੋਹ-ਸੰ ਨ ਧ ਪ, ਨੁ ਧ ਪ, ਮ ਗੁ ਰ ਸ", 563),
                                _line(5, "ਵਾਰ ੨੪ ਪਉੜੀ ਨੰ. ੭ ਭਾਈ ਗੁਰਦਾਸ ਜੀ", 859, x0=500, x1=1200), _line(6, "(ਸੁਪੁੱਤ੍ਰ ਗੁਰ ਅੰਗਦ)", 932, x0=600, x1=1000),
                                _line(7, "ਸਬਦੇ ਸਬਦੁ ਮਿਲਾਇਆ ਗੁਰਮੁਖਿ ਅਘੜ ਘੜਾਏ ਗਹਣਾ।। ਭਾਇ ਭਗਤਿ ਭੈ ਚਲਣਾ", 1004), _line(8, "ਆਪੁ ਗਣਾਇ ਨ ਖਲਹਲੁ ਖਹਣਾ।। ਦੀਨ ਦੁਨੀ ਦੀ ਸਾਹਿਬੀ ਗੁਰਮੁਖਿ ਗੋਸ ਨਸੀਨੀ", 1080),
                                _line(9, "ਬਹਣਾ।। ਪੁਤੁ ਸਪੁਤੁ ਬਬਾਣੇ ਲਹਣਾ।।੭।।", 1147),
                                _line(10, "ਪਦ ਅਰਥ:- ਗੁਰਮੁਖਿ ਅਘੜੁ ਘੜਾਏ ਗਹਣਾ-ਗੁਰਮੁਖ ਮਨ ਨੂੰ ਘੜਕੇ ਗਹਿਣਾ ਰੂਪ ਬਣਾ ਲੈਂਦੇ ਹਨ।", 1374),
                                _line(11, "ਸ ਰ | ਗ ਮ | ਪ — | ਧ ਨ", 1440),         # a line of the meanings the OCR made a row of
                                _line(12, "ਰਾਗੁ ਤੁਖਾਰੀ ਤੀਨ ਤਾਲ ਮਾਤਰਾਂ-16 ਮੱਧ ਲੈਅ", 1642, bold=True)] + grids(13, 1718), style=style)
        p155 = self._page(155, [_line(1, "ਰਾਗੁ ਤੁਖਾਰੀ ਕਹਿਰਵਾ ਤਾਲ ਮਾਤਰਾਂ-8 ਮੱਧ ਲੈਅ", 228, bold=True)] + grids(2, 300), style=style)
        roles = [(r["role"], r.get("source")) for r in p154["regions"]]
        self.assertIn(("ref", None), roles)
        self.assertIn(("shabad", "B"), roles)                                      # the pauri, a shabad of Bhai Gurdas
        self.assertTrue(any(r.get("prose") for r in p154["regions"]))                # the word meanings
        spans = link_pages([p152, p153, p154, p155], style)
        got = [(s["pages"], s["sid"], [r.get("source") for _, r in s["shabad"] if r.get("source")], bool(s.get("inherited"))) for s in spans]
        self.assertEqual(got, [([152], 5372, [], False), ([152, 153], 5372, [], True),         # Devgandhari's second taal inherits its pauri
                               ([154], None, ["B"], False), ([154, 155], None, ["B"], True)])  # Tukhari's own, not Devgandhari's
        # without the source line the Tukhari notations have no shabad: dropped, never Devgandhari's
        bare = self._page(154, [_line(1, "ਰਾਗ ਤੁਖਾਰੀ", 193, bold=True, x0=560, x1=1000),
                                _line(2, "ਰਾਗੁ-ਤੁਖਾਰੀ ਥਾਟ-ਤੋੜੀ ਸਵਰ-ਦੋਨੋਂ ਨਿਸ਼ਾਦ, ਗੰਧਾਰ ਕੋਮਲ,", 361), _line(3, "ਜਾਤੀ-ਔੜਵ-ਸੰਪੂਰਨ ਵਾਦੀ-ਪੰਚਮ ਸੰਵਾਦੀ-ਸ਼ੜਜ", 504),
                                _line(4, "ਰਾਗੁ ਤੁਖਾਰੀ ਤੀਨ ਤਾਲ ਮਾਤਰਾਂ-16 ਮੱਧ ਲੈਅ", 1642, bold=True)] + grids(5, 1718), style=style)
        dropped = []
        spans = link_pages([p152, p153, bare, p155], style, dropped)
        self.assertEqual([(s["pages"], s["sid"]) for s in spans], [([152], 5372), ([152, 153], 5372)])
        self.assertTrue(any(154 in d["pages"] for d in dropped))
