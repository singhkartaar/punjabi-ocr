"""
Gurmukhi text preparation for embedding.

A line as stored carries structure that is not meaning: the verse counter
(॥੧॥, ॥੪॥੨॥), the pause marker (॥ ਰਹਾਉ ॥, ॥ ਰਹਾਉ ਦੂਜਾ ॥) and the dandas
themselves. Fed to a sentence model they would make every line with the same
counter look a little alike. They are removed here and only here; the database
keeps the line exactly as published.

The marker is recognised as a whole danda-delimited segment, never as a
substring -- ਰਹਾਉ is also an ordinary verb ("I dwell / I remain") in six lines
of SGGS, and those words must stay.
"""
from __future__ import annotations
import re

_DANDA = re.compile(r"[॥।]")
# A line that closes a stanza (antara) ends with its counter: ॥੧॥, or ॥੪॥੨॥
# (stanza and running shabad count). The rahao line also carries a counter
# but is classified separately (kind='rahao') and never counts as a stanza.
STANZA_END = re.compile(r"॥\s*[੦-੯]+(?:\s*॥\s*[੦-੯]+)*\s*॥\s*$")
_DIGITS = re.compile(r"^[੦-੯0-9\s]+$")
_MARKER = re.compile(r"^\s*ਰਹਾਉ(\s+ਦੂਜਾ)?\s*$")


def clean_gurmukhi(text: str) -> str:
    """'ਕੋਈ ਨ ਜਾਣੈ ਤੇਰਾ ਕੇਤਾ ਕੇਵਡੁ ਚੀਰਾ ॥੧॥ ਰਹਾਉ ॥' -> 'ਕੋਈ ਨ ਜਾਣੈ ਤੇਰਾ ਕੇਤਾ ਕੇਵਡੁ ਚੀਰਾ'"""
    kept = []
    for seg in _DANDA.split(text):
        if not seg.strip() or _DIGITS.match(seg) or _MARKER.match(seg):
            continue
        kept.append(seg)
    return " ".join(" ".join(kept).split())


def is_stanza_end(text: str) -> bool:
    """'ਹੁਕਮਿ ਰਜਾਈ ਚਲਣਾ ਨਾਨਕ ਲਿਖਿਆ ਨਾਲਿ ॥੧॥' -> True"""
    return bool(STANZA_END.search(text))
