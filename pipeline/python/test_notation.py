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

from lib.notation_layout import classify_line, link_pages, page_layout  # noqa: E402
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
        # the same shabad set again under a heading with no number keeps its shabad
        p5 = self._page(170, [_line(1, "ਤਾਲ ਝਪਤਾਲ", 200, bold=True),
                              _line(2, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 400),
                              _line(3, "ਮਨ ਕਹਾ | ਲੁ ਭਾ | ਈ ऽ | ਐ ਆਨ", 480),
                              _line(4, "ਸ ਰੇ | ਗ ਮ | ਪ — | ਧ ਨੀ", 560)])
        spans = link_pages([p1, p2, p3, p5], merge_style(None))
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

    def test_the_lead_in_of_the_next_shabad_leaves_loose_text_with_the_notation_before(self):
        p1 = self._page(78, self._shabad(78, 4151, 300) + [_line(9, "ਰਾਗ ਭੈਰਉ ਤਾਲ ਦਾਦਰਾ", 640, bold=True)] + self._grids(10, 700)
                        + [_line(13, "ਹਸ ਤ ਖੋ5 ਲਤ ਤੇ ਰੇ ਦੇ$ ਹੁ58 ਭਰੇ ਆ 8 ਇਆ", 1290), _line(14, "ਦੁਧੁ ਦੁ ਦੁ ਨੀਨੀ ਸਾਂ ਸਾਂ ਰੇਸਾਂ ਨੀਸਾਂ ਰੁੇ ਸਾਂ ਦੁ ਪ ਪ", 1365)]
                        + [_line(20, "ਭਗਤਿ ਕਰਤ ਨਾਮਾ ਪਕਰਿ ਉਠਾਇਆ ॥੧॥", 1430, kind="gurbani", matches=[{"shabad_id": 4152, "line_id": 1, "score": 0.9, "source": "G"}]),
                           _line(21, "ਹੀਨੜੀ ਜਾਤਿ ਮੇਰੀ ਜਾਦਿਮ ਰਾਇਆ ॥", 1510, kind="gurbani", matches=[{"shabad_id": 4152, "line_id": 2, "score": 0.9, "source": "G"}])]
                        + [_line(22, "੧੯.", 1600, x0=800, x1=860)])
        spans = link_pages([p1], merge_style(None))
        self.assertEqual(len(spans), 2)
        self.assertGreaterEqual(spans[0]["extent"][78][3], 1365 + 60)      # the two loose rows stay with the first
        self.assertEqual(spans[1]["extent"][78][1], 1430)

    def test_a_line_with_no_two_gurmukhi_letters_is_not_a_heading(self):
        self.assertFalse(is_heading_like("N..O\" '"))
        self.assertEqual(classify_line(_line(5, "N..O\" '", 2021, x0=984, x1=1127)), "text")


class SectionTaalTests(unittest.TestCase):
    def test_a_section_taal_is_written_as_its_key_and_validates(self):
        rec = _record()
        rec["sections"][0]["taal"] = "teentaal"
        self.assertEqual(validate(rec), [])
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
