"""
Which shabad a notation is of.

Three witnesses, weighed together: the corpus matches the merge already
made on the printed shabad text (lib/ocr_match through 22_ocr_merge.py),
the reference the book prints beside it ("(ਗਉੜੀ ਕੀ ਵਾਰ ਮ: ੫, ਪੰਨਾ ੩੨੦)"),
and, later, the syllables of the bol row. The first is the strongest --
every matched line votes for its shabad with its score -- and the printed
ang either agrees, which raises the confidence, or disagrees, which is a
flag for the reviewer rather than a reason to overrule the text. A block
whose lines match nothing in any source and carries no reference is not
Gurbani (a thumri, a dharna, a composition of another poet) and is kept as
such.

    resolved   >= 0.80   the notation links to the shabad
    weak       0.50-0.80 linked, flagged weak-shabad for review
    unresolved <  0.50   shabad_id null, flagged unresolved-shabad
"""
from __future__ import annotations
import re
import sqlite3
from collections import defaultdict

from lib.notation_vocab import raag_key_from_corpus

RESOLVED, WEAK = 0.80, 0.50


def votes_from_lines(merged_lines: list[dict]) -> tuple[dict[int, float], dict[int, list[int]], list[str]]:
    """{shabad_id: weight}, {shabad_id: [line_ids]}, [sources] over the merge's matches on these lines."""
    weight: dict[int, float] = defaultdict(float)
    line_ids: dict[int, list[int]] = defaultdict(list)
    sources: list[str] = []
    for line in merged_lines:
        for m in line.get("matches") or []:
            sid = m.get("shabad_id")
            if sid is None:
                continue
            weight[sid] += float(m.get("score") or 0.0)
            if m.get("line_id") is not None and m["line_id"] not in line_ids[sid]:
                line_ids[sid].append(m["line_id"])
            sources.append(m.get("source") or "G")
    return dict(weight), dict(line_ids), sources


_NOT_LETTERS = re.compile(r"[^\u0a05-\u0a39\u0a3e-\u0a4d\u0a59-\u0a5e\u0a70-\u0a75]")


def by_printed_ang(merged_lines: list[dict], ref: dict, con: sqlite3.Connection) -> tuple[int, float, list[int]] | None:
    """
    No line of the block matched the corpus, but the book prints the ang: a salok of a vaar set in
    half-lines ("ਸਾਚੁ ਸੀਲ ਸਚੁ ਸੰਜਮੀ / ਸਾ ਪੂਰੀ ਪਰਵਾਰਿ॥") matches no corpus line by any one of its own
    (Guru Nanak Dev Raag Ratnaavlee, the Maru Vaar; third cut, 5 October 2026). The block read as one
    text, against the lines of that ang and its neighbours: the shabad two or more of whose lines are
    found in it, and more of them than any other's. (shabad_id, score, [line_ids]) or None.
    """
    from rapidfuzz import fuzz
    block = _NOT_LETTERS.sub("", " ".join((l.get("text") or "") for l in merged_lines))
    if len(block) < 12:
        return None
    # by the shabad's first ang (both corpora have it; corpus.sqlite's lines carry no ang of their own)
    try:
        rows = con.execute("SELECT l.shabad_id, l.line_id, l.gurmukhi_uni FROM lines l JOIN shabads s ON s.shabad_id = l.shabad_id "
                           "WHERE s.ang_start BETWEEN ? AND ? AND l.kind IN ('line','rahao')",
                           (ref["ang_from"] - 1, (ref.get("ang_to") or ref["ang_from"]) + 1)).fetchall()
    except sqlite3.Error:
        return None
    found: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for sid, line_id, text in rows:
        line = _NOT_LETTERS.sub("", text or "")
        if len(line) >= 8:
            score = fuzz.partial_ratio(line, block)
            if score >= 85:
                found[sid].append((line_id, score / 100.0))
    ranked = sorted(found.items(), key=lambda kv: (-len(kv[1]), -sum(sc for _, sc in kv[1])))
    if not ranked or len(ranked[0][1]) < 2 or (len(ranked) > 1 and len(ranked[1][1]) == len(ranked[0][1])):
        return None
    sid, hits = ranked[0]
    return sid, sum(sc for _, sc in hits) / len(hits), [lid for lid, _ in hits]


_LIST_NUMBER = re.compile(r"^\s*[0-9੦-੯]{1,2}\s*[.)]\s*")


_CORPUS_LINES: dict[int, list[tuple[int, str]]] = {}


def _corpus_lines(con: sqlite3.Connection) -> list[tuple[int, str]]:
    """Every line of the corpus as (shabad_id, its letters), read once a connection."""
    key = id(con)
    if key not in _CORPUS_LINES:
        try:
            rows = con.execute("SELECT shabad_id, gurmukhi_uni FROM lines WHERE kind IN ('line','rahao')").fetchall()
        except sqlite3.Error:
            rows = []
        _CORPUS_LINES[key] = [(sid, _NOT_LETTERS.sub("", text or "")) for sid, text in rows]
    return _CORPUS_LINES[key]


def other_shabads(regions: list[dict], con: sqlite3.Connection | None, own_sid: int | None) -> list[dict]:
    """
    The record's `also`: the other shabads a book prints beside a notation as sung to the same tune
    (Swar Samund: "ਹੋਰ ਸ਼ਬਦ (ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ) –", then "੧. ਮੈ ਅੰਧੁਲੇ ਕੀ ਟੇਕ ਤੇਰਾ (ਅੰਗ-੧੭੪)" and one more), from the
    layout's `others` lines. The number printed there is a page of the Amrit Kirtan pothi the label names,
    not an ang of the Granth, so the words name the shabad: the corpus line the merge matched, else the one
    shabad a line of which begins with the printed words; a line that opens several shabads, or none, keeps
    its printed text and no id. [{"shabad_id", "ang", "printed", "first_line", "confidence"}], the ang the Granth's.
    """
    from rapidfuzz import fuzz
    from lib.notation_text import parse_ref
    out: list[dict] = []
    for region in regions:
        label = (region.get("text") or "").split("\n")[0]
        of_granth = "ਅੰਮ੍ਰਿਤ ਕੀਰਤਨ" not in label          # else the numbers are that pothi's pages
        for item in region.get("others") or []:
            printed = (item.get("text") or "").strip()
            ref = (parse_ref(printed) or {}) if of_granth else {}
            ang = ref.get("ang_from") if ref.get("source", "G") == "G" else None
            words = _NOT_LETTERS.sub("", _LIST_NUMBER.sub("", printed.split("(")[0]))
            if len(words) < 8:
                continue                                   # the label's own tail, a stray mark
            sid, confidence = None, 0.0
            for m in sorted(item.get("matches") or [], key=lambda m: -(m.get("score") or 0)):
                if m.get("shabad_id") is None or (m.get("source") or "G") != "G" or (m.get("score") or 0) < 0.9:
                    continue
                at = corpus_facts(con, m["shabad_id"]).get("ang") if con is not None else None
                if ang is None or (at is not None and abs(at - ang) <= 1):
                    sid, confidence = m["shabad_id"], 0.9
                    break
            if sid is None and con is not None:
                # the printed words are the head of a line (its first, most often): the shabads a line of which begins so
                best: dict[int, float] = {}
                for cand, line in _corpus_lines(con):
                    if len(line) >= len(words) and line[0] == words[0]:
                        score = fuzz.ratio(words, line[:len(words)])
                        if score >= 90:
                            best[cand] = max(best.get(cand, 0), score)
                if ang is not None:
                    best = {c: sc for c, sc in best.items() if abs((corpus_facts(con, c).get("ang") or -9) - ang) <= 1}
                top = max(best.values(), default=0)
                near = [c for c, sc in best.items() if sc >= top - 2]
                if len(near) == 1:
                    sid, confidence = near[0], 0.7
            if sid is not None and sid == own_sid:
                continue
            facts = corpus_facts(con, sid) if (con is not None and sid is not None) else {}
            out.append({"shabad_id": sid, "ang": facts.get("ang") if sid is not None else ang, "printed": printed,
                        "first_line": facts.get("first_line"), "confidence": confidence})
    return out


def resolve_shabad(merged_lines: list[dict], ref: dict | None, con: sqlite3.Connection | None,
                   bol_match: dict | None = None, scriptures: sqlite3.Connection | None = None) -> dict:
    """
    @param merged_lines  the merged lines of the shabad block (with `matches`)
    @param ref           parse_ref() of the printed reference, or None
    @param con           the corpus (gurbani.sqlite or corpus.sqlite) for the shabad's facts
    @param bol_match     a match_text_all() hit on the bol row, if the caller made one
    @param scriptures    the store of the other scriptures (data/scriptures-full.sqlite, D, B, K), for the facts of
                         a shabad the merge matched there; their ids never collide with the Granth's
    @returns the record's "shabad" object plus "flags" and "kind_hint"
    """
    weight, line_ids, sources = votes_from_lines(merged_lines)
    n_lines = sum(1 for l in merged_lines if l.get("matches"))
    windowed = None
    if not weight and con is not None and ref and ref.get("source", "G") == "G" and ref.get("ang_from") and merged_lines:
        windowed = by_printed_ang(merged_lines, ref, con)
        if windowed:
            weight, line_ids, sources = {windowed[0]: windowed[1]}, {windowed[0]: windowed[2]}, ["G"]
    flags: list[str] = []
    total = sum(weight.values())
    winner, share = None, 0.0
    if weight:
        winner = max(weight, key=weight.get)
        share = weight[winner] / total if total else 0.0
    matched_lines = len(line_ids.get(winner, [])) if winner else 0
    top_score = max((max(float(m.get("score") or 0) for m in l.get("matches") or [{}]) for l in merged_lines
                     if l.get("matches") and any(m.get("shabad_id") == winner for m in l["matches"])), default=0.0)
    text_ok = winner is not None and share >= 0.5 and (matched_lines >= 2 or (matched_lines == 1 and top_score >= 0.9))

    # the winner's own scripture: the source its matched lines came from (the Granth unless the merge found it in the other store)
    winner_source = "G"
    if winner is not None:
        of_winner = [m.get("source") or "G" for l in merged_lines for m in (l.get("matches") or []) if m.get("shabad_id") == winner]
        winner_source = max(set(of_winner), key=of_winner.count) if of_winner else "G"
    facts = corpus_facts(con, winner, winner_source, scriptures) if winner is not None else {}
    ang = facts.get("ang")
    ref_state = None                    # None: no ref; True: agrees; False: disagrees
    if winner_source != "G":
        # a vaar or a bani is cited by its number, not an ang: the printed source agreeing is the witness
        if ref and ref.get("source"):
            ref_state = ref["source"] == winner_source
            if not ref_state:
                flags.append("ref-conflict")
    elif ref and ref.get("ang_from") and ang:
        ref_state = (ref["ang_from"] - 1) <= ang <= (ref["ang_to"] + 1)
        if not ref_state:
            flags.append("ref-conflict")
    bol_state = None
    if bol_match and winner is not None:
        bol_state = bol_match.get("shabad_id") == winner

    method = []
    if text_ok:
        method.append("stream")
    if ref_state:
        method.append("ref")
    if bol_state:
        method.append("bol")
    confidence = 0.0
    if winner is not None:
        confidence = 0.5 * share * (1.0 if text_ok else 0.6)
        confidence += 0.3 if ref_state else (0.15 if ref_state is None else 0.0)
        confidence += 0.2 if bol_state else (0.1 if bol_state is None else 0.0)
    if not text_ok and bol_match and bol_match.get("score", 0) >= 0.85 and winner is None:
        winner = bol_match["shabad_id"]
        facts = corpus_facts(con, winner) if con is not None else {}
        winner_source = "G"
        confidence = 0.6
        method = ["bol"]
    if windowed:
        # found by the printed ang, not by a line the merge matched: linked, and weak for the reviewer
        method, confidence = ["ref"], min(confidence, 0.7)
    confidence = round(min(1.0, confidence), 3)
    source = None
    if winner is not None:
        source = winner_source
    elif sources:
        source = max(set(sources), key=sources.count)
    if ref and ref.get("source") and ref["source"] != "G" and source in (None, "G") and not text_ok:
        source = ref["source"]

    shabad_id = winner if confidence >= WEAK else None
    # a shabad of another scripture is not a Granth shabad_id: the record keeps its source and its facts, and the
    # match itself under `scripture_match`, for the link pass (37_link_notation_scriptures.py) that writes the
    # notation's verse into the store's scripture_links -- the Granth's id space stays the Granth's
    scripture_match = None
    if shabad_id is not None and winner_source != "G":
        scripture_match = {"source": winner_source, "shabad_id": shabad_id, "line_ids": list(line_ids.get(winner, [])),
                           "confidence": confidence, "method": "+".join(method) or "stream"}
        shabad_id = None
    if shabad_id is None and scripture_match is None:
        flags.append("unresolved-shabad")
        method_str = "none"
    elif shabad_id is None:
        method_str = "none"
    else:
        if confidence < RESOLVED:
            flags.append("weak-shabad")
        method_str = "+".join(method) if method else "ref-window"
        if method_str == "ref":
            method_str = "ref-window"
    kind_hint = "notation"
    if shabad_id is None and not weight and (ref is None or ref.get("source") == "G") and n_lines == 0:
        kind_hint = "non-gurbani" if merged_lines else "partial"
    out = {
        "shabad_id": shabad_id, "source": source if shabad_id is not None or ref else None,
        "confidence": confidence, "method": method_str,
        "line_ids": line_ids.get(winner, []) if shabad_id is not None else [],
        "ang": facts.get("ang") if shabad_id is not None else (ref or {}).get("ang_from"),
        "writer": facts.get("writer") if shabad_id is not None else None,
        "first_line": facts.get("first_line") if shabad_id is not None else None,
        "raag": facts.get("raag") if shabad_id is not None else None,
        "raag_key": raag_key_from_corpus(facts.get("raag")) if shabad_id is not None else None,
        "printed": [(l.get("text") or "").strip() for l in merged_lines if (l.get("text") or "").strip()],
        "ref": ref, "verified": False,
        "votes": {str(k): round(v, 3) for k, v in sorted(weight.items(), key=lambda kv: -kv[1])[:5]},
    }
    if scripture_match:
        out.update({"scripture_match": scripture_match, "source": winner_source, "ang": facts.get("ang"), "writer": facts.get("writer"),
                    "first_line": facts.get("first_line"), "raag": facts.get("raag"), "raag_key": None, "confidence": confidence})
        kind_hint = "notation"
    return {"shabad": out, "flags": flags, "kind_hint": kind_hint}


def corpus_facts(con: sqlite3.Connection | None, shabad_id: int, source: str = "G",
                 scriptures: sqlite3.Connection | None = None) -> dict:
    """
    raag, writer, ang, first line and rahao line of a shabad, from gurbani.sqlite or corpus.sqlite; of a shabad
    of another scripture (source D, B, K) from the scriptures store, where its "raag" is the bani or vaar
    (`section`) and its ang the source's own page (the vaar's number, the kabit's). {} without the store.
    """
    if source != "G":
        if scriptures is None:
            return {}
        try:
            row = scriptures.execute("SELECT section, writer, ang_start FROM shabads WHERE source=? AND shabad_id=?",
                                     (source, shabad_id)).fetchone()
            if not row:
                return {}
            first = scriptures.execute("SELECT gurmukhi_uni FROM lines WHERE source=? AND shabad_id=? AND kind IN ('line','rahao') "
                                       "ORDER BY position_in_shabad LIMIT 1", (source, shabad_id)).fetchone()
        except sqlite3.Error:
            return {}
        return {"raag": row[0], "writer": row[1], "ang": row[2], "first_line": first[0] if first else None, "rahao_line": None}
    if con is None:
        return {}
    row = con.execute("SELECT raag, writer, ang_start FROM shabads WHERE shabad_id=?", (shabad_id,)).fetchone()
    if not row:
        return {}
    raag, writer, ang = row
    first = con.execute("SELECT gurmukhi_uni FROM lines WHERE shabad_id=? AND kind IN ('line','rahao') "
                        "ORDER BY position_in_shabad LIMIT 1", (shabad_id,)).fetchone()
    rahao = con.execute("SELECT gurmukhi_uni FROM lines WHERE shabad_id=? AND kind='rahao' "
                        "ORDER BY position_in_shabad LIMIT 1", (shabad_id,)).fetchone()
    return {"raag": raag, "writer": writer, "ang": ang,
            "first_line": first[0] if first else None, "rahao_line": rahao[0] if rahao else None}
