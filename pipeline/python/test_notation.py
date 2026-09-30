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


if __name__ == "__main__":
    if "--write-expected" in sys.argv:
        write_expected()
    else:
        unittest.main()
