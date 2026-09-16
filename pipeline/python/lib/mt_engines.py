# -*- coding: utf-8 -*-
"""
Translating the exchange format (docs/darpan-translation.md).

Everything about a translation run that is not the command line lives here: the
prompt, the shape of a batch, the checks an answer has to pass, and the engine
that does the work. The CLI (11_translate_mt.py) only schedules.

Two things are deliberate.

The prompt rules are not written here. They are READ out of
docs/darpan-translation.md, which is the contract the pipeline already
publishes to anyone running a translation by hand. One source, so the rules a
model is given and the rules a person is given cannot drift apart.

Nothing is ever silently dropped or silently altered. An answer that comes back
with Gurmukhi in it, or with a pad-arth headword the engine has changed, is
rejected and reported -- the importer would refuse it anyway, and a translation
that quietly rewrites Sahib Singh is worse than a missing one.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import ROOT                                            # noqa: E402

GURMUKHI = re.compile(u"[਀-੿]")
LATIN = re.compile(u"[A-Za-z]")
# Sahib Singh ends a clause on a danda; it is where a long arth can be cut.
DANDA = re.compile(u"(?<=[।॥])\\s*")

RULES_DOC = os.path.join(ROOT, "docs", "darpan-translation.md")

PREAMBLE = (
    "You are translating Prof. Sahib Singh's Guru Granth Darpan, a Punjabi "
    "commentary on Sri Guru Granth Sahib, into English.\n\n"
    "Follow these rules exactly:\n"
)
FORMAT_NOTE = (
    "\nAnswer with a JSON array and nothing else. For each record you are "
    "given, return one object.\n"
    "  a record with \"src\"   -> {\"id\": <the id>, \"en\": \"<the English>\"}\n"
    "  a record with \"pairs\" -> {\"id\": <the id>, \"pairs\": [{\"w\": \"<the headword, copied back "
    "unchanged>\", \"en\": \"<the English gloss>\"}, ...]}\n"
    "Return every id you were given, in the same order, and no others."
)


def load_rules(path: str = RULES_DOC) -> str:
    """The numbered 'Translation rules' list, read from the contract document."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"^## Translation rules\s*\n(.*?)(?=^## )", text, re.S | re.M)
    if not m:
        raise RuntimeError("could not find '## Translation rules' in %s" % path)
    body = m.group(1).strip()
    if not body:
        raise RuntimeError("'## Translation rules' is empty in %s" % path)
    return body


def system_instruction(path: str = RULES_DOC) -> str:
    return PREAMBLE + load_rules(path) + FORMAT_NOTE


def read_jsonl(path: str):
    """Records of a JSONL file, skipping blanks and the _meta header."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and "_meta" in rec:
                continue
            yield rec


def done_ids(path: str) -> set:
    """Ids already translated in an output file, so a run resumes where it stopped."""
    out = set()
    for rec in read_jsonl(path):
        rid = str(rec.get("id", ""))
        if rid:
            # a chunked record is only done when rejoined, so ignore the pieces
            out.add(rid.split("#c")[0] if "#c" in rid else rid)
    return out


def chunk_src(src: str, max_chars: int):
    """
    Cut a very long arth into pieces at clause boundaries.

    41 stanza records run past any sane request size. Splitting at the danda
    keeps whole clauses together, which is the smallest unit that still reads
    as a sentence; the pieces are translated separately and rejoined in order.
    """
    text = str(src or "").strip()
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for clause in DANDA.split(text):
        if not clause:
            continue
        if current and len(current) + len(clause) > max_chars:
            pieces.append(current.strip())
            current = clause
        else:
            current += clause
    if current.strip():
        pieces.append(current.strip())
    return pieces or [text]


def to_payload(rec: dict) -> dict:
    """What the engine is shown: the id, the verse as context, and the text."""
    out = {"id": rec["id"]}
    if rec.get("gurmukhi"):
        out["gurmukhi"] = rec["gurmukhi"]
    if isinstance(rec.get("pairs"), list):
        out["pairs"] = [{"w": p.get("w", ""), "src": p.get("src", "")} for p in rec["pairs"]]
    else:
        out["src"] = rec.get("src", "")
    return out


# How much payload one request may carry. Measured, not guessed: a batch of 40
# arth records is 25,844 characters and Vertex closes the connection without
# answering, which sends every record in it down the one-at-a-time retry path.
# An arth paragraph ranges from a line to a page, so a fixed record count says
# nothing about request size; the character budget is what actually binds.
MAX_BATCH_CHARS = 8000


def batches(records, size: int, max_chars: int = MAX_BATCH_CHARS, sizeof=None):
    """
    Group records into requests, closing a batch on whichever limit comes first.

    A single record over the budget still goes on its own rather than being
    dropped -- chunking long records is chunk_src's job, not this one's.
    """
    sizeof = sizeof or (lambda rec: len(json.dumps(rec, ensure_ascii=False)))
    batch, chars = [], 0
    for rec in records:
        weight = sizeof(rec)
        if batch and (len(batch) >= size or (max_chars and chars + weight > max_chars)):
            yield batch
            batch, chars = [], 0
        batch.append(rec)
        chars += weight
    if batch:
        yield batch


def check(answer: dict, sent: dict):
    """
    Is this answer usable? Returns (record, None) or (None, why).

    The same checks the importer applies, made here so a bad record can be
    retried once against the engine rather than discovered hours later.
    """
    if not isinstance(answer, dict):
        return None, "not an object"
    rid = str(answer.get("id", ""))
    if rid != str(sent.get("id", "")):
        return None, "id does not match the record it answers"

    if "pairs" in sent:
        pairs = answer.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            return None, "pad-arth without pairs"
        want = [str(p.get("w", "")) for p in sent["pairs"]]
        got = [str(p.get("w", "")) for p in pairs]
        if got != want:
            return None, "headwords changed"
        for p in pairs:
            gloss = str(p.get("en", "")).strip()
            if not gloss:
                return None, "empty gloss"
            if GURMUKHI.search(gloss):
                return None, "gloss is not English"
        return {"id": rid, "pairs": [{"w": str(p["w"]), "en": str(p["en"]).strip()} for p in pairs]}, None

    en = " ".join(str(answer.get("en", "")).split())
    if not en:
        return None, "empty en"
    if GURMUKHI.search(en):
        return None, "en contains Gurmukhi"
    if not LATIN.search(en):
        return None, "en has no Latin letters"
    return {"id": rid, "en": en}, None


def parse_answer(text: str):
    """The JSON array an engine returned, tolerating a code fence around it."""
    body = str(text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in the answer")
    return json.loads(body[start:end + 1])


class VertexEngine:
    """
    Gemini through Vertex AI, authenticated by Application Default Credentials.

    The project id is never defaulted in code: it comes from --project or
    GCP_PROJECT, so a checkout of this repository cannot bill anyone.
    """

    name = "vertex"

    def __init__(self, project: str, location: str = "us-central1",
                 model: str = "gemini-2.5-flash", temperature: float = 0.0,
                 rules_doc: str = RULES_DOC, attempts: int = 5):
        if not project:
            raise SystemExit(
                "no GCP project: pass --project or set GCP_PROJECT.\n"
                "Authenticate first with: gcloud auth application-default login")
        from google import genai                       # imported lazily: only this engine needs it
        from google.genai import types
        self._types = types
        self.model = model
        self.temperature = temperature
        self.attempts = attempts
        self.system = system_instruction(rules_doc)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def translate(self, payload: list) -> str:
        """One batch in, the model's raw text out. Retries on rate limits."""
        config = self._types.GenerateContentConfig(
            system_instruction=self.system,
            response_mime_type="application/json",
            temperature=self.temperature,
        )
        body = json.dumps(payload, ensure_ascii=False)
        last = None
        for attempt in range(1, self.attempts + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=body, config=config)
                return resp.text
            except Exception as err:                   # noqa: BLE001 - the SDK raises many types
                last = err
                if attempt == self.attempts:
                    break
                # 429 and 5xx are worth waiting out; anything else fails fast
                if not re.search(r"429|RESOURCE_EXHAUSTED|50\d|UNAVAILABLE|DEADLINE",
                                 str(err), re.I):
                    break
                time.sleep(min(60, 2 ** attempt) * (0.5 + os.urandom(1)[0] / 512.0))
        raise RuntimeError("vertex failed after %d attempts: %s" % (self.attempts, last))


ENGINES = {"vertex": VertexEngine}
