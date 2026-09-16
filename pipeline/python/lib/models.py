"""
Embedding model profiles.

Each semantic index is built with ONE model and queried with THE SAME model, so
a query lands in the space its documents were indexed in. The profile is copied
into the index's manifest.json; the server reads it back to construct the
matching query encoder. Nothing about a model is hardcoded anywhere else.

Both models ship as int8 dynamic-quantized ONNX (Xenova exports), which is what
runs on the device too -- embedding with fp32 at build time and int8 on the
phone would put query and document on different sides of a quantization gap.
"""
from __future__ import annotations
import os
from lib.paths import ROOT

MODELS_ROOT = os.path.join(ROOT, "vendor", "models")

PROFILES = {
    # English translations -> English query. BGE v1.5 uses CLS pooling and a
    # retrieval prefix on queries only (see 1_Pooling/config.json upstream).
    "bge-small-en-v1.5": {
        "model": "Xenova/bge-small-en-v1.5 (onnx/model_quantized.onnx)",
        "model_dir": "bge-small-en-v1.5",
        "tokenizer": "wordpiece",
        "lowercase": True,
        "strip_accents": True,
        "pooling": "cls",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "pad_token": "[PAD]",
        "pad_id": 0,
        "embed_dim": 384,
        "max_len": 160,
        "text_source": None,
    },
    # Gurmukhi text -> Gurmukhi query. Multilingual-E5 uses mean pooling and
    # asymmetric "query: " / "passage: " prefixes (both required -- the model
    # was trained with them). XLM-R SentencePiece vocabulary, Punjabi included.
    "multilingual-e5-small": {
        "model": "Xenova/multilingual-e5-small (onnx/model_quantized.onnx)",
        "model_dir": "multilingual-e5-small",
        "tokenizer": "unigram",
        "lowercase": False,
        "strip_accents": False,
        "pooling": "mean",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "pad_token": "<pad>",
        "pad_id": 1,
        "embed_dim": 384,
        "max_len": 160,
        "text_source": "gurmukhi_uni",
    },
    # The same model after contrastive fine-tuning on Gurbani (see
    # 06_make_training_pairs.py and finetune_e5.ipynb). Identical architecture,
    # tokenizer and settings -- only the weights differ -- so it drops into the
    # same pipeline and the same JS encoder.
    "multilingual-e5-small-gurbani": {
        "model": "multilingual-e5-small fine-tuned on Gurbani (onnx/model_quantized.onnx)",
        "model_dir": "multilingual-e5-small-gurbani",
        "tokenizer": "unigram",
        "lowercase": False,
        "strip_accents": False,
        "pooling": "mean",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "pad_token": "<pad>",
        "pad_id": 1,
        "embed_dim": 384,
        "max_len": 160,
        "text_source": "gurmukhi_uni",
    },
    # The same fine-tuned weights with the embedding table cut down to the
    # pieces this corpus and its readers actually emit (16_trim_vocab.py):
    # 250,002 vocabulary rows to ~35,000, 118MB to 35MB, for the phone. Every
    # kept row is byte-identical -- the table is per-tensor quantized, so rows
    # are sliced, never requantized -- and the trimmer refuses to write unless
    # every indexed text still tokenizes to the same pieces. So the vectors
    # built with the parent stay valid and no index is rebuilt; only
    # reference_tokens.json is regenerated, because the ids renumber.
    "multilingual-e5-small-gurbani-trim": {
        "model": "multilingual-e5-small fine-tuned on Gurbani, vocabulary trimmed (16_trim_vocab.py)",
        "model_dir": "multilingual-e5-small-gurbani-trim",
        "tokenizer": "unigram",
        "lowercase": False,
        "strip_accents": False,
        "pooling": "mean",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "pad_token": "<pad>",
        "pad_id": 1,
        "embed_dim": 384,
        "max_len": 160,
        "text_source": "gurmukhi_uni",
    },
    # MuRIL: Google's Indic BERT, 768-dim, WordPiece over a vocabulary that
    # includes Gurmukhi. A lab candidate only (see docs/semantic-indexing.md):
    # it is base-size, so ~240 MB int8 against a 1 GB VM, and its tokenizer
    # MUST NOT lowercase or strip accents -- Gurmukhi matras are combining
    # marks and stripping them destroys the word. 02_reduce_quantize.py brings
    # 768 down to the same 256 dimensions as everything else.
    "muril-base-cased": {
        "model": "google/muril-base-cased (onnx/model_quantized.onnx)",
        "model_dir": "muril-base-cased",
        "tokenizer": "wordpiece",
        "lowercase": False,
        "strip_accents": False,
        "pooling": "mean",
        "query_prefix": "",
        "doc_prefix": "",
        "pad_token": "[PAD]",
        "pad_id": 0,
        "embed_dim": 768,
        "max_len": 160,
        "text_source": "gurmukhi_uni",
    },
    # The same after contrastive fine-tuning on our pairs (06_make_training_pairs.py).
    "muril-base-cased-gurbani": {
        "model": "muril-base-cased fine-tuned on Gurbani (onnx/model_quantized.onnx)",
        "model_dir": "muril-base-cased-gurbani",
        "tokenizer": "wordpiece",
        "lowercase": False,
        "strip_accents": False,
        "pooling": "mean",
        "query_prefix": "",
        "doc_prefix": "",
        "pad_token": "[PAD]",
        "pad_id": 0,
        "embed_dim": 768,
        "max_len": 160,
        "text_source": "gurmukhi_uni",
    },
}


def profile_by_dir(model_dir: str) -> dict:
    """The profile whose model_dir an index manifest recorded."""
    for name, p in PROFILES.items():
        if p["model_dir"] == model_dir:
            return profile(name)
    raise KeyError(f"no model profile has model_dir {model_dir!r}")


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise KeyError(f"unknown model profile {name!r}; known: {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["name"] = name
    p["path"] = os.path.join(MODELS_ROOT, p["model_dir"])
    return p


def is_installed(name: str) -> bool:
    """Whether the profile's ONNX file is under vendor/models."""
    return os.path.exists(os.path.join(profile(name)["path"], "model_quantized.onnx"))


def first_available(names) -> str | None:
    """The first of these profiles whose weights are on this machine, or None.
    The writings corpus prefers the Gurbani fine-tune and falls back to the
    base multilingual model, which is what a machine without the fine-tune
    (any public checkout) has; the manifest records which one embedded it."""
    for n in names:
        if n in PROFILES and is_installed(n):
            return n
    return None
