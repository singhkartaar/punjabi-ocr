"""
Sentence embedding via ONNX Runtime.

Deliberately uses the SAME quantized .onnx file that will ship on-device, so a
query embedded at runtime lands in exactly the vector space the corpus was
indexed in. Embedding with fp32 here and int8 on the phone would introduce a
quantization gap between query and document vectors for no reason.

Pooling and prefixes come from the model profile (lib/models.py): BGE v1.5 is
CLS-pooled with a query-only prefix; multilingual-E5 is mean-pooled with
"query: " / "passage: " on both sides.
"""
from __future__ import annotations
import os
import numpy as np
from lib.models import profile as load_profile

DIM = 384
MAX_LEN = 160          # Gurbani lines and their translations are short; see token-length report
# BGE's documented retrieval prefix -- kept as a module constant for callers that
# import it directly; the profile is the source of truth.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model: str = "bge-small-en-v1.5", max_len: int | None = None,
                 threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.profile = load_profile(model)
        model_dir = self.profile["path"]
        if not os.path.exists(os.path.join(model_dir, "model_quantized.onnx")):
            raise FileNotFoundError(
                f"{model_dir}/model_quantized.onnx is missing -- download the model first "
                f"(see README, 'Rebuilding the index'). There is no fallback embedder on purpose.")
        self.max_len = max_len or self.profile["max_len"]
        self.pooling = self.profile["pooling"]
        self.query_prefix = self.profile["query_prefix"]
        self.doc_prefix = self.profile["doc_prefix"]
        self.dim = self.profile["embed_dim"]

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_len)
        self.tokenizer.enable_padding(length=None, pad_id=self.profile["pad_id"],
                                      pad_token=self.profile["pad_token"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model_quantized.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 64, prefix: str | None = None) -> np.ndarray:
        """Embed texts as DOCUMENTS (doc_prefix applied unless `prefix` overrides it)."""
        pre = self.doc_prefix if prefix is None else prefix
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = [pre + t for t in texts[start:start + batch_size]]
            encs = self.tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]      # (B, T, DIM)
            out[start:start + len(chunk)] = l2_normalize(pool(hidden, mask, self.pooling))
        return out

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], prefix=self.query_prefix)[0]


def pool(hidden: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    if how == "cls":
        return hidden[:, 0, :]
    if how == "mean":
        # Masked mean over real tokens only; padding contributes nothing.
        m = mask[:, :, None].astype(np.float32)
        return (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
    raise ValueError(f"unknown pooling {how!r}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (x / norms).astype(np.float32)
