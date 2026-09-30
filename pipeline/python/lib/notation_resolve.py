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


def resolve_shabad(merged_lines: list[dict], ref: dict | None, con: sqlite3.Connection | None,
                   bol_match: dict | None = None) -> dict:
    """
    @param merged_lines  the merged lines of the shabad block (with `matches`)
    @param ref           parse_ref() of the printed reference, or None
    @param con           the corpus (gurbani.sqlite or corpus.sqlite) for the shabad's facts
    @param bol_match     a match_text_all() hit on the bol row, if the caller made one
    @returns the record's "shabad" object plus "flags" and "kind_hint"
    """
    weight, line_ids, sources = votes_from_lines(merged_lines)
    n_lines = sum(1 for l in merged_lines if l.get("matches"))
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

    facts = corpus_facts(con, winner) if (con is not None and winner is not None) else {}
    ang = facts.get("ang")
    ref_state = None                    # None: no ref; True: agrees; False: disagrees
    if ref and ref.get("ang_from") and ang:
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
        confidence = 0.6
        method = ["bol"]
    confidence = round(min(1.0, confidence), 3)
    source = None
    if sources:
        source = max(set(sources), key=sources.count)
    if ref and ref.get("source") and ref["source"] != "G" and source in (None, "G") and not text_ok:
        source = ref["source"]

    shabad_id = winner if confidence >= WEAK else None
    if shabad_id is None:
        flags.append("unresolved-shabad")
        method_str = "none"
    else:
        if confidence < RESOLVED:
            flags.append("weak-shabad")
        method_str = "+".join(method) if method else "ref-window"
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
    return {"shabad": out, "flags": flags, "kind_hint": kind_hint}


def corpus_facts(con: sqlite3.Connection, shabad_id: int) -> dict:
    """raag, writer, ang, first line and rahao line of a shabad, from gurbani.sqlite or corpus.sqlite."""
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
