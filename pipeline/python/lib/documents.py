"""
Long texts and whole-shabad documents for embedding.

`embed_long` is the one place a text longer than the model's window is
handled: it is split on sentence marks into chunks that fit, the chunks are
embedded, and their vectors averaged. Short texts (nearly all of them) go
straight through. Used by 01_embed.py for every source, so a long Darpan
paragraph never silently loses its tail to truncation.

`shabad_documents` / `embed_documents` build the whole-shabad experiment: a
shabad vector as the mean of its line vectors is a blurred centroid, so this
embeds the shabad as ONE text with the rahao first -- the thesis, and the
part that must survive if the text has to be chunked. Measured and not
adopted (README), kept so it can be re-tested.
"""
from __future__ import annotations
import re
import numpy as np
from lib.embedder import l2_normalize
from lib.sources import load_texts, source_lang, parse_source

SEPARATOR = {"pa": " ॥ ", "en": " "}
_SENTENCE = re.compile(r"(?<=[।॥\.\?!])\s+")


def _token_len(emb, text: str) -> int:
    return len(emb.tokenizer.encode(text, add_special_tokens=False).ids)


def split_to_fit(emb, text: str, max_tokens: int) -> list[str]:
    """Chunks of whole sentences, each within max_tokens (a single sentence over the limit stands alone)."""
    if _token_len(emb, text) <= max_tokens:
        return [text]
    chunks, cur, used = [], [], 0
    for sent in _SENTENCE.split(text):
        if not sent.strip():
            continue
        n = _token_len(emb, sent)
        if cur and used + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, used = [], 0
        cur.append(sent)
        used += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def embed_long(emb, texts: list[str], max_tokens: int, batch_size: int = 64, progress=None):
    """
    Embed texts as documents; a text over the window is chunked and its chunk
    vectors averaged, then re-normalised. Returns (vectors, chunked_count).
    """
    pieces, owner = [], []
    chunked = 0
    for i, t in enumerate(texts):
        parts = split_to_fit(emb, t, max_tokens)
        if len(parts) > 1:
            chunked += 1
        for p in parts:
            pieces.append(p)
            owner.append(i)
    out = np.zeros((len(texts), emb.dim), dtype=np.float32)
    acc = np.zeros((len(texts), emb.dim), dtype=np.float32)
    cnt = np.zeros(len(texts), dtype=np.int32)
    B = 4000
    for s in range(0, len(pieces), B):
        V = emb.encode(pieces[s:s + B], batch_size=batch_size)
        for j, v in enumerate(V):
            o = owner[s + j]
            acc[o] += v
            cnt[o] += 1
        if progress:
            progress(min(s + B, len(pieces)), len(pieces))
    ok = cnt > 0
    out[ok] = l2_normalize(acc[ok] / cnt[ok, None])
    return out, chunked


def shabad_documents(con, source: str):
    """{shabad_id: [text, ...]} -- rahao lines first, then the rest in order, from the given source."""
    rows = con.execute("""SELECT line_id, shabad_id, kind FROM lines
                          WHERE kind IN ('line','rahao') ORDER BY shabad_id, position_in_shabad""").fetchall()
    by_source = load_texts(con, source)
    # one text per line: the first source in the list that has it
    texts = {}
    for part in parse_source(source):
        for lid, t in by_source[part].items():
            texts.setdefault(lid, t)
    docs = {}
    for lid, sid, kind in rows:
        t = texts.get(lid)
        if not t:
            continue
        rahao, other = docs.setdefault(sid, ([], []))
        (rahao if kind == "rahao" else other).append(t)
    return {sid: r + o for sid, (r, o) in docs.items() if r or o}


def embed_documents(emb, docs: dict, source: str, max_tokens: int = 500, batch_size: int = 16):
    """
    Embed each document; a document longer than the model's window is split
    into consecutive chunks of whole lines and the chunk vectors averaged.
    Returns (ids, vectors, chunk_count) with ids ascending.
    """
    sep = SEPARATOR[source_lang(source)]
    ids = sorted(docs)
    chunk_texts, owner = [], []
    for sid in ids:
        lines = docs[sid]
        chunk, used = [], 0
        for line in lines:
            n = _token_len(emb, line)
            if chunk and used + n > max_tokens:
                chunk_texts.append(sep.join(chunk)); owner.append(sid)
                chunk, used = [], 0
            chunk.append(line); used += n
        if chunk:
            chunk_texts.append(sep.join(chunk)); owner.append(sid)
    V = emb.encode(chunk_texts, batch_size=batch_size)
    acc = {}
    for sid, v in zip(owner, V):
        acc.setdefault(sid, []).append(v)
    vectors = np.vstack([l2_normalize(np.mean(acc[sid], axis=0, keepdims=True))[0] for sid in ids]).astype(np.float32)
    return np.array(ids, dtype=np.int32), vectors, len(chunk_texts)
