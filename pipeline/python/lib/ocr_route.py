"""
When a line is worth paying for, and how much paying is allowed.

The Vertex translation run cost $13K because nothing stood between a loop and
the bill. Every paid engine in this pipeline goes through Budget: before a
request it asks whether the month's spend plus this request stays under the
cap, and after a successful request it appends a ledger row. The ledger is a
JSONL file under data/ocr/, so what was spent is readable without a console,
and --dry-run in 21_ocr_run.py prints the estimate and writes nothing.

Routing thresholds (needs_arbiter) are calibrated on the ground truth in
24_ocr_eval.py: they start at the values below and the docstring is updated
when the measurement says otherwise. Cross-engine agreement, not an engine's own
confidence, is the primary signal: Tesseract's word conf is 100 + 5 x certainty
and is known to sit at 0 on correctly read complex-script words.
"""
from __future__ import annotations
import datetime as _dt
import json
import os
import uuid

AGREEMENT_LOW = 0.85       # share of grapheme columns every engine agreed on
OOV_HIGH = 0.25            # out-of-vocabulary words / words, on lines of >= OOV_MIN_WORDS
OOV_MIN_WORDS = 4
CONF_LOW = 0.5
PAGE_MIN_LINES = 3         # a page is routed whole (Vision bills per image) when
PAGE_MIN_SHARE = 0.15      # this many, or this share of, its body lines need it


# The English lexicon is the three Gurbani translations: ordinary English, but
# not the proper nouns and Sikh terms an English book about the Gurus is made
# of, so an out-of-vocabulary rate that flags a Punjabi line flags every
# English one. Measured on Ten Masters: 10% of lines routed for OOV, none of
# them wrong, while Tesseract reads that book at 99.1% word accuracy.
OOV_HIGH_BY_LANG = {"pa": OOV_HIGH, "hi": OOV_HIGH, "en": 0.6}


def needs_arbiter(line: dict, lang: str = "pa") -> str | None:
    """Why this merged line should be shown to a paid engine, or None."""
    if line.get("kind") == "gurbani":
        return None                                   # corpus text: nothing to arbitrate
    if line.get("kind") == "gurbani-unmatched":
        return "unmatched-bold"
    if line.get("agreement") is not None and line["agreement"] < AGREEMENT_LOW:
        return "low-agreement"
    oov_high = OOV_HIGH_BY_LANG.get(lang, OOV_HIGH)
    if (line.get("oov") or 0.0) > oov_high and line.get("n_words", OOV_MIN_WORDS) >= OOV_MIN_WORDS:
        return "oov"
    if line.get("conf") is not None and line["conf"] < CONF_LOW:
        return "low-conf"
    return None


def page_needs_arbiter(lines: list[dict], lang: str = "pa") -> dict | None:
    body = [ln for ln in lines if ln.get("zone", "body") == "body"]
    reasons: dict = {}
    for ln in body:
        why = needs_arbiter(ln, lang)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
    # a bold verse the corpus does not hold is a missing SOURCE (Dasam Bani,
    # Bhai Gurdas, a heading), not a reading problem: it does not send the
    # page to a paid engine on its own
    n = sum(v for k, v in reasons.items() if k != "unmatched-bold")
    if n >= PAGE_MIN_LINES or (body and n / len(body) >= PAGE_MIN_SHARE):
        return {"lines": n, "reasons": reasons}
    return None


class Budget:
    """
    A monthly cap over an append-only ledger.

      b = Budget(ledger_path, cap_usd=5.0)
      if b.allows(cost): ...call...; b.charge("vision", book, pages, cost)
    """

    def __init__(self, ledger_path: str, cap_usd: float, run_id: str | None = None):
        self.path = ledger_path
        self.cap = float(cap_usd)
        self.run_id = run_id or uuid.uuid4().hex[:8]

    def rows(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        out.append(json.loads(raw))
                    except ValueError:
                        continue
        return out

    def spent_this_month(self, now: _dt.datetime | None = None) -> float:
        now = now or _dt.datetime.now(_dt.timezone.utc)
        month = now.strftime("%Y-%m")
        return round(sum(float(r.get("usd", 0.0)) for r in self.rows()
                         if str(r.get("ts", "")).startswith(month)), 6)

    def allows(self, usd: float) -> bool:
        return self.spent_this_month() + float(usd) <= self.cap + 1e-9

    def charge(self, engine: str, book: str, pages: int, usd: float, note: str = "") -> dict:
        row = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "engine": engine, "book": book, "pages": int(pages), "usd": round(float(usd), 6),
               "run_id": self.run_id}
        if note:
            row["note"] = note
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
