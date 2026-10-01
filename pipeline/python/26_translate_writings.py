"""
Translate a Punjabi or Hindi book's commentary into English, locally, so Ask can answer from it.

  26_translate_writings.py --work santhya                       sarvam-translate on the GPU
  26_translate_writings.py --work santhya --engine indictrans2   AI4Bharat IndicTrans2 (MIT, 1B, sentence-level)
  26_translate_writings.py --work santhya --engine vertex        Gemini through Vertex (metered)
  26_translate_writings.py --work santhya --limit 40 --print 5   a look before the run
  26_translate_writings.py --work h --src-lang hi                a Hindi book (default: the work's language)

Reads data/writings/<work>.jsonl and writes data/writings/<work>.en.jsonl:
one record per translated paragraph {"unit_id", "en", "engine", "model"},
resumable by unit_id (lib/mt_engines.done_ids). Gurbani paragraphs (style
"quote") are NOT translated: their line_ids already point at the corpus's
English translations, and a machine rendering of scripture would be shown
beside the real ones. Headings and body prose are.

Local engines, all with open weights, compared by 28_translate_bench.py:

  sarvam       sarvamai/sarvam-translate (Gemma-3-4B fine-tune, GPL-3.0 weights):
               a paragraph at a time, forgiving of OCR noise; 0.58 s a paragraph
               here in batches of 8. Its trained prompt names only the target
               language, so --src-lang does not change what it is asked.
  indictrans2  ai4bharat/indictrans2-indic-en-1B (MIT, seq2seq): a sentence at a
               time, so a paragraph is split on dandas and full stops, translated
               in batches of 32 sentences and joined back. Gated on Hugging Face
               (a click-through); the engine prints the login steps if refused.
               Its model code targets transformers 4: run it from .venv-dots.
  madlad       google/madlad400-3b-mt (Apache-2.0, T5): sentence-level like the above.
               transformers 5 refuses to tie its embeddings ("both are present in the
               checkpoints with different values") and the output is digits; under
               transformers 4.51 the same weights translate, so run it from .venv-dots
  nllb         facebook/nllb-200-distilled-1.3B (CC-BY-NC-4.0): a reference, not a default
  llama        any instruction model behind llama-server (Gemma 3, Qwen3, Sarvam-M
               as a GGUF): --model-path is the server URL; a paragraph at a time

The Vertex run that translated the Darpan cost $13K, so the Vertex engine
stays a measured spot-check (--engine vertex --limit N, or the benchmark)
through the same Budget as the OCR arbiters.

Every answer is checked the way 11_translate_mt.py checks its own: empty, the
source script in the English, no Latin letters at all, far too long for its
source, or looping; it is rejected and counted, never written.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.mt_engines import done_ids, read_jsonl
from lib.paths import OCR_COSTS, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WRITINGS = os.path.join(ROOT, "data", "writings")
GURMUKHI = re.compile("[਀-੿]")
DEVANAGARI = re.compile("[ऀ-ॿ]")
SOURCE_SCRIPT = {"pa": GURMUKHI, "hi": DEVANAGARI}
LANG_NAME = {"pa": "Punjabi", "hi": "Hindi"}
LATIN = re.compile("[A-Za-z]")
SARVAM = "sarvamai/sarvam-translate"
INDICTRANS2 = "ai4bharat/indictrans2-indic-en-1B"
MADLAD = "google/madlad400-3b-mt"
NLLB = "facebook/nllb-200-distilled-1.3B"
LLAMA_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = {"sarvam": SARVAM, "indictrans2": INDICTRANS2, "madlad": MADLAD, "nllb": NLLB, "llama": LLAMA_URL}
THINK = re.compile(r"<think>.*?</think>\s*", re.S)
VERTEX_USD_PER_1K_CHARS = 0.02          # order of magnitude for a Flash-class model; the ledger records the estimate
SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """A paragraph as sentences, on a danda or Latin stop followed by space."""
    return [s for s in (p.strip() for p in SENTENCE_END.split(" ".join((text or "").split()))) if s]


def _load_or_explain(load, model_path: str):
    """Run a from_pretrained-style loader; a Hugging Face refusal becomes the login steps."""
    from lib.ocr_engines import HF_GATE_HELP, is_gate_error
    try:
        return load()
    except Exception as e:                                # the hub raises several types
        if is_gate_error(e):
            raise SystemExit(HF_GATE_HELP % (model_path, model_path)) from e
        raise


class SarvamTranslateEngine:
    """sarvam-translate through transformers; one paragraph per generation."""
    name = "sarvam-translate"

    def __init__(self, model_path: str = SARVAM, max_new_tokens: int = 768, device: str = "cuda", src: str = "pa"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_path = model_path
        self.tok = _load_or_explain(lambda: AutoTokenizer.from_pretrained(model_path), model_path)
        self.model = _load_or_explain(lambda: AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device), model_path)
        self.max_new_tokens = int(max_new_tokens)

    def model_desc(self) -> str:
        return self.model_path

    instruct = None           # text -> system message (--prompt rules); None: the model's trained one

    def _prompt(self, text: str, tgt: str) -> str:
        system = self.instruct(text) if self.instruct else "Translate the text below to %s." % tgt
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def translate(self, text: str, src: str = "Punjabi", tgt: str = "English") -> str:
        return self.translate_many([text], src, tgt)[0]

    def translate_many(self, texts: list[str], src: str = "Punjabi", tgt: str = "English") -> list[str]:
        """A batch in one generate() call, prompts padded on the left so every
        answer starts at the same position; a decoder-only model's padding
        must sit before the prompt or the last tokens it attends to are pad."""
        self.tok.padding_side = "left"
        prompts = [self._prompt(t, tgt) for t in texts]
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        # English runs to about twice the Gurmukhi's token count; a cap tied
        # to the longest input stops a looping answer early instead of at 768
        longest = int(inputs["attention_mask"].sum(dim=1).max())
        cap = min(self.max_new_tokens, 48 + 2 * longest)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        start = inputs["input_ids"].shape[1]
        return [self.tok.decode(row[start:], skip_special_tokens=True).strip() for row in out]

    def cost_usd(self, chars: int) -> float:
        return 0.0


class IndicTrans2Engine:
    """AI4Bharat IndicTrans2 (seq2seq, sentence-level) through transformers + IndicTransToolkit."""
    name = "indictrans2"
    TAGS = {"pa": "pan_Guru", "hi": "hin_Deva"}
    SENTENCES = 32            # sentences per generate() call
    MAX_TOKENS = 256          # the model's window; a sentence longer than this is cut, not looped

    def __init__(self, model_path: str = INDICTRANS2, device: str = "cuda", src: str = "pa", beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit import IndicProcessor
        except ImportError:
            # the PyPI package is a Cython source distribution that needs a C++
            # compiler; lib/indictrans_processor.py is the same code without it
            from lib.indictrans_processor import IndicProcessor
        if src not in self.TAGS:
            raise SystemExit("indictrans2: no language tag for %r (one of %s)" % (src, ", ".join(self.TAGS)))
        self.torch, self.device, self.beams = torch, device, int(beams)
        self.model_path, self.src_tag = model_path, self.TAGS[src]
        self.tok = _load_or_explain(lambda: AutoTokenizer.from_pretrained(model_path, trust_remote_code=True), model_path)
        # fp32 on purpose: in fp16 this model's generation ran away (thirty
        # paragraphs still not done after 25 minutes at 100% GPU); at 1B
        # parameters fp32 is 4.4 GB and fits
        self.model = _load_or_explain(lambda: AutoModelForSeq2SeqLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32).to(device).eval(), model_path)
        self.ip = IndicProcessor(inference=True)

    def model_desc(self) -> str:
        return self.model_path

    def translate(self, text: str, src: str = "Punjabi", tgt: str = "English") -> str:
        return self.translate_many([text], src, tgt)[0]

    def _sentences(self, sents: list[str]) -> list[str]:
        out: list[str] = []
        t0 = time.time()
        for b in range(0, len(sents), self.SENTENCES):
            chunk = sents[b:b + self.SENTENCES]
            batch = self.ip.preprocess_batch(chunk, src_lang=self.src_tag, tgt_lang="eng_Latn")
            inputs = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                              max_length=self.MAX_TOKENS).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**inputs, use_cache=True, min_length=0, max_new_tokens=self.MAX_TOKENS,
                                          num_beams=self.beams, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang="eng_Latn"))
            print("    indictrans2 %d/%d sentences, %.1fs" % (min(b + self.SENTENCES, len(sents)), len(sents), time.time() - t0),
                  file=sys.stderr, flush=True)
        return out

    def translate_many(self, texts: list[str], src: str = "Punjabi", tgt: str = "English") -> list[str]:
        """Every paragraph split into sentences, all sentences translated in
        batches, each paragraph joined back from its own sentences."""
        sents, owner = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t):
                sents.append(s)
                owner.append(i)
        joined: list[list[str]] = [[] for _ in texts]
        for i, en in zip(owner, self._sentences(sents) if sents else []):
            joined[i].append(en.strip())
        return [" ".join(j) for j in joined]

    def cost_usd(self, chars: int) -> float:
        return 0.0


class Seq2SeqEngine:
    """
    A Hugging Face encoder-decoder translator, a sentence at a time, for the
    models that differ only in how they are told the languages:

      madlad   google/madlad400-3b-mt (T5; Apache-2.0): the target is a prefix
               token on the source, "<2en> ..."; the source language is not stated
      nllb     facebook/nllb-200-distilled-1.3B (M2M-100; CC-BY-NC-4.0): the
               tokenizer's src_lang and a forced first token name the languages

    Paragraphs are split into sentences (split_sentences), translated in
    batches and joined back, as IndicTrans2Engine does.
    """
    name = "seq2seq"
    NLLB_TAGS = {"pa": "pan_Guru", "hi": "hin_Deva"}
    SENTENCES = 16
    MAX_TOKENS = 256

    def __init__(self, model_path: str, kind: str = "madlad", device: str = "cuda", src: str = "pa", beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch, self.device, self.beams, self.kind = torch, device, int(beams), kind
        self.model_path, self.src = model_path, src
        self.name = kind
        self.tok = _load_or_explain(lambda: AutoTokenizer.from_pretrained(model_path), model_path)
        dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if kind == "madlad" else torch.float16)
        self.model = _load_or_explain(lambda: AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype=dtype).to(device).eval(), model_path)
        self.forced = None
        if kind == "nllb":
            self.tok.src_lang = self.NLLB_TAGS[src]
            self.forced = self.tok.convert_tokens_to_ids("eng_Latn")

    def model_desc(self) -> str:
        return self.model_path

    def translate(self, text: str, src: str = "Punjabi", tgt: str = "English") -> str:
        return self.translate_many([text], src, tgt)[0]

    def _sentences(self, sents: list[str]) -> list[str]:
        out: list[str] = []
        for b in range(0, len(sents), self.SENTENCES):
            chunk = sents[b:b + self.SENTENCES]
            if self.kind == "madlad":
                chunk = ["<2en> " + s for s in chunk]
            inputs = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.MAX_TOKENS).to(self.device)
            kw = {"forced_bos_token_id": self.forced} if self.forced is not None else {}
            with self.torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=self.MAX_TOKENS, num_beams=self.beams, **kw)
            out.extend(s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def translate_many(self, texts: list[str], src: str = "Punjabi", tgt: str = "English") -> list[str]:
        sents, owner = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t):
                sents.append(s)
                owner.append(i)
        joined: list[list[str]] = [[] for _ in texts]
        for i, en in zip(owner, self._sentences(sents) if sents else []):
            joined[i].append(en)
        return [" ".join(j) for j in joined]

    def cost_usd(self, chars: int) -> float:
        return 0.0


def strip_think(text: str) -> str:
    """A reasoning model's <think> block is not part of the translation."""
    return THINK.sub("", text or "").strip()


class LlamaServerEngine:
    """
    Any instruction model served by llama.cpp's llama-server (or another
    OpenAI-compatible endpoint), a paragraph at a time: Gemma 3, Qwen3,
    Sarvam-M, and so on, from a GGUF. Start the server yourself, e.g.

      vendor/llama.cpp/llama-server.exe -m vendor/models/gguf/gemma-3-12b-it-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080

    The prompt names the source language and asks for the translation only;
    a <think> block, when the model emits one, is removed.
    """
    name = "llama"

    def __init__(self, url: str = LLAMA_URL, label: str | None = None, src: str = "pa", timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.label = label
        import urllib.request
        # the server answers 503 while the model is still loading; give it a while
        deadline = time.time() + 180
        last = None
        while True:
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=10) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.model = (data.get("data") or [{}])[0].get("id") or "unknown"
                break
            except Exception as e:                        # noqa: BLE001
                last = e
                if time.time() > deadline:
                    raise SystemExit("no llama-server at %s (%s); start it first, see LlamaServerEngine" % (self.url, last))
                time.sleep(3)

    def model_desc(self) -> str:
        return self.label or os.path.basename(str(self.model))

    def translate_many(self, texts: list[str], src: str = "Punjabi", tgt: str = "English") -> list[str]:
        """One request per paragraph; the server holds one slot."""
        return [self.translate(t, src, tgt) for t in texts]

    instruct = None

    def translate(self, text: str, src: str = "Punjabi", tgt: str = "English") -> str:
        import urllib.request
        system = self.instruct(text) if self.instruct else (
            "Translate the %s text you are given into plain %s. Keep names and technical terms; do not add or "
            "omit sentences; do not explain. Answer with the translation only." % (src, tgt))
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0, "max_tokens": 48 + 3 * max(1, len(text) // 3),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return strip_think(data["choices"][0]["message"]["content"])

    def cost_usd(self, chars: int) -> float:
        return 0.0


class VertexTranslateEngine:
    """Gemini through Vertex, metered by the OCR budget ledger."""
    name = "vertex"

    def __init__(self, project: str | None = None, model: str = "gemini-2.5-flash", budget=None):
        from google import genai
        from google.genai import types
        project = project or os.environ.get("GCP_PROJECT")
        if not project:
            raise SystemExit("no GCP project: set GCP_PROJECT (gcloud auth application-default login first)")
        if budget is None:
            raise SystemExit("vertex: a Budget is required (--budget-usd)")
        self.client = genai.Client(vertexai=True, project=project, location="us-central1")
        self.types = types
        self.model = model
        self.budget = budget

    def model_desc(self) -> str:
        return "vertex " + self.model

    def cost_usd(self, chars: int) -> float:
        return round(chars / 1000.0 * VERTEX_USD_PER_1K_CHARS, 6)

    instruct = None

    def translate(self, text: str, src: str = "Punjabi", tgt: str = "English") -> str:
        cost = self.cost_usd(len(text))
        if not self.budget.allows(cost):
            raise SystemExit("vertex: the monthly cap would be exceeded (spent %.4f)" % self.budget.spent_this_month())
        config = self.types.GenerateContentConfig(
            system_instruction=self.instruct(text) if self.instruct else (
                "Translate the %s commentary you are given into plain %s. Keep names and technical terms; "
                "do not add or omit sentences. Answer with the translation only." % (src, tgt)),
            temperature=0.0)
        resp = self.client.models.generate_content(model=self.model, contents=text, config=config)
        self.budget.charge("vertex-translate", "writings", 0, cost, note="%d chars" % len(text))
        return (resp.text or "").strip()


MAX_RATIO = 4.0          # English chars per Punjabi char beyond which the answer is not a translation
REPEAT_WINDOW = 6        # a run of this many words seen three times is the model looping


def repeats(en: str, window: int = REPEAT_WINDOW, times: int = 3) -> bool:
    ws = en.split()
    if len(ws) < window * times:
        return False
    seen: dict[tuple, int] = {}
    for i in range(len(ws) - window + 1):
        k = tuple(w.lower() for w in ws[i:i + window])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def check(en: str, src: str = "", lang: str = "pa", glossary: dict | None = None) -> str | None:
    """Why an answer is not a translation, or None. A 4B model given a
    fragment of OCR noise sometimes answers with its own boilerplate
    ("This is the translation of the given Punjabi text to English:")
    repeated until the token cap; length and repetition catch that. With a
    glossary (lib/mt_glossary), a term the source uses that the English
    renders by none of its accepted forms is a reason too: "term: bhog"."""
    en = " ".join((en or "").split())
    if not en:
        return "empty"
    if SOURCE_SCRIPT.get(lang, GURMUKHI).search(en):
        return "%s in the English" % LANG_NAME.get(lang, lang)
    if not LATIN.search(en):
        return "no Latin letters"
    if src and len(en) > MAX_RATIO * len(src) + 40:
        return "%.0fx longer than the %s" % (len(en) / max(1, len(src)), LANG_NAME.get(lang, "source"))
    if repeats(en):
        return "repeats itself"
    if glossary and src:
        from lib.mt_glossary import missing
        gone = missing(glossary, src, en)
        if gone:
            return "term: %s" % gone[0]["en"]
    return None


PROMPTS = ("stock", "rules", "terms")
ARBITERS = ("none", "self", "llama", "vertex")
# Measured by 28_translate_bench.py on 30 paragraphs of the Sant Attar Singh
# biography against references written from the Punjabi (2026-09-30):
#   stock  chrF 53.9, 27 of 30 accepted, 57 of 61 Sikh terms rendered
#   rules  chrF 55.8 on the 19 it answered; 10 copied back in Punjabi, 44 terms
#   terms  chrF 55.4, 27 accepted, 58 terms; with --arbiter self all 30, chrF 55.0
# Without a glossary terms is stock, so it is the default for every work.
DEFAULT_PROMPT = "terms"


def rules_prompt(glossary: dict, text: str, tgt: str = "English", pairs: bool = True) -> str:
    """
    --prompt rules: the model's own instruction, the glossary's rule, the
    terms THIS paragraph uses (lib/mt_glossary.prompt_lines; a small model
    given forty pairs drifts), and the one thing a 4B model had to be told
    twice: no explanations and no brackets of its own.

    pairs=False is one line of English: sarvam-translate is tuned on a single
    system line, and given the rule and "ਭੋਗ = bhog" pairs on lines of their
    own it translated the instruction itself into Punjabi (29 of 30 answers
    on the Sant Attar Singh bench). An instruction-following engine (llama,
    vertex) gets the pairs.
    """
    from lib.mt_glossary import present, prompt_lines
    terms = present(glossary, text)
    if not pairs:
        line = "Translate the text below to %s." % tgt
        if terms:
            line += " Keep these Sikh terms as they are: %s." % ", ".join(t["en"] for t in terms)
        return line + " Do not add explanations or brackets."
    parts = ["Translate the text below to %s." % tgt]
    if glossary.get("rule"):
        parts.append(glossary["rule"])
    lines = prompt_lines(terms, text)
    if lines:
        parts.append("Terms:\n" + "\n".join(lines))
    parts.append("Do not add explanations, notes or brackets. Answer with the translation only.")
    return "\n".join(parts)


def prepare(prompt: str, glossary: dict, text: str) -> str:
    """
    The text the engine is given: the paragraph, or with --prompt terms the
    paragraph with its glossary terms written in English inside it
    (lib/mt_glossary.inline_terms). A tuned translator like sarvam-translate
    follows no instruction beyond its trained one -- given more it copied the
    Punjabi back or translated the instruction -- but keeps a Latin word.
    """
    if prompt == "terms" and glossary.get("terms"):
        from lib.mt_glossary import inline_terms
        return inline_terms(glossary, text)[0]
    return text


def arbiter_prompt(engine, prompt: str) -> str:
    """What the arbiter asks with: a prompt that engine follows, other than the one that failed."""
    if isinstance(engine, SarvamTranslateEngine):
        return "stock" if prompt == "terms" else "terms"
    return "rules"


def set_prompt(engine, prompt: str, glossary: dict, tgt: str = "English") -> None:
    """Point an engine at the rules prompt, or back at its own (stock, terms)."""
    if prompt == "terms":
        prompt = "stock"                                  # the instruction is the model's own
    if hasattr(engine, "instruct"):
        pairs = not isinstance(engine, SarvamTranslateEngine)
        engine.instruct = (lambda text: rules_prompt(glossary, text, tgt, pairs)) if prompt == "rules" else None
    elif prompt == "rules":
        raise SystemExit("%s takes no instruction: --prompt rules needs sarvam, llama or vertex" % engine.name)


def answer(en: str | None, src: str, lang: str, glossary: dict) -> tuple[str | None, str | None, int]:
    """(cleaned English, why it is refused or None, glosses stripped)."""
    if en is None:
        return None, "out of memory alone", 0
    stripped = 0
    if glossary.get("terms"):
        from lib.mt_glossary import present, strip_glosses
        en, stripped = strip_glosses(en, src, present(glossary, src))
    en = " ".join(en.split())
    return en, check(en, src, lang, glossary), stripped


def arbitrate(arbiter, recs: list[dict], lang: str, glossary: dict, tgt: str,
              prompt: str = "rules") -> list[tuple[dict, str | None, str | None]]:
    """
    A second engine, or the same one asked another way (arbiter_prompt), for
    the paragraphs the first answers were refused for: [(record, English or
    None, why)]. What it cannot fix either is refused twice and not written.
    """
    set_prompt(arbiter, prompt, glossary, tgt)
    out = []
    for rec in recs:
        try:
            raw = translate_batch(arbiter, [prepare(prompt, glossary, rec["text"])], tgt)[0]
        except SystemExit:
            raise
        except Exception as err:                  # noqa: BLE001
            out.append((rec, None, "arbiter: %s" % err))
            continue
        en, why, _ = answer(raw, rec["text"], lang, glossary)
        out.append((rec, en, why))
    return out


def work_meta(path: str) -> dict:
    """The _meta of a writings file, or {} where it has none."""
    with open(path, encoding="utf-8") as fh:
        head = fh.readline()
    try:
        return json.loads(head).get("_meta") or {}
    except json.JSONDecodeError:
        return {}


def work_language(path: str) -> str | None:
    """The _meta.language of a writings file, if it states one."""
    return work_meta(path).get("language")


def local_weights(hf_id: str) -> str:
    """
    vendor/models/<name> when the weights are already on disk there, else the
    Hugging Face id (which the hub would fetch, 8 GB for sarvam-translate).
    The runbook puts the weights under vendor/models by the id's last part.
    """
    local = os.path.join(ROOT, "vendor", "models", hf_id.rsplit("/", 1)[-1])
    return local if os.path.isdir(local) else hf_id


def translate_batch(engine, texts: list[str], tgt: str) -> list[str | None]:
    """
    One batch through the engine; when the card runs out of memory on it
    (a batch of long paragraphs on an 8 GB card with the weights split to the
    host), the batch is retried one paragraph at a time after the cache is
    dropped, and only a paragraph that fails alone comes back as None.
    Before this a whole batch of eight was rejected for one allocation.
    """
    if not hasattr(engine, "translate_many"):
        return [engine.translate(t, tgt) for t in texts]
    try:
        return engine.translate_many(texts, tgt)
    except Exception as err:                     # noqa: BLE001
        if "out of memory" not in str(err).lower() or len(texts) == 1:
            raise
    torch = getattr(engine, "torch", None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    out: list[str | None] = []
    for t in texts:
        try:
            out.append(engine.translate_many([t], tgt)[0])
        except Exception as err:                 # noqa: BLE001
            if "out of memory" not in str(err).lower():
                raise
            out.append(None)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def make_engine(name: str, model_path: str | None = None, src: str = "pa", budget_usd: float = 2.0,
                device: str = "auto"):
    """
    One of the translation engines by its --engine name. `device` is where a
    local causal model loads: "cuda", "cpu", or "auto" (accelerate splits the
    weights between the card and the host memory: sarvam-translate's 8.1 GB in
    bf16 do not fit an 8 GB card whole).
    """
    if name == "vertex":
        from lib.ocr_route import Budget
        return VertexTranslateEngine(budget=Budget(OCR_COSTS, budget_usd))
    if name == "indictrans2":
        return IndicTrans2Engine(model_path or local_weights(INDICTRANS2), src=src)
    if name == "sarvam":
        return SarvamTranslateEngine(model_path or local_weights(SARVAM), src=src, device=device)
    if name in ("madlad", "nllb"):
        # MT_DEVICE=cpu runs a seq2seq model in fp32 on the CPU: MADLAD 3B in
        # bf16 on the GPU ran away (dots to the token cap, invented sentences),
        # and its 11.8 GB of fp32 weights do not fit a 12 GB card
        return Seq2SeqEngine(model_path or DEFAULT_MODEL[name], kind=name, src=src,
                             device=os.environ.get("MT_DEVICE", "cuda"))
    if name == "llama":
        return LlamaServerEngine(model_path or LLAMA_URL, src=src)
    raise SystemExit("unknown engine %r" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--engine", default="sarvam", choices=["sarvam", "indictrans2", "madlad", "nllb", "llama", "vertex"])
    ap.add_argument("--model-path", help="local weights or HF id (llama: the server URL); default the engine's (%s)"
                    % ", ".join("%s: %s" % kv for kv in DEFAULT_MODEL.items()))
    ap.add_argument("--src-lang", choices=sorted(LANG_NAME), help="default: the work's _meta.language, else pa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", help="comma-separated unit_ids to (re)translate, whatever is done")
    ap.add_argument("--print", dest="show", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--styles", default="body,heading,footnote")
    ap.add_argument("--batch", type=int, default=8, help="paragraphs per generate() call (local engines)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="where sarvam loads: auto fills the card and puts the rest in host memory (8.1 GB of "
                         "weights on an 8 GB card ran at 0.35 s a paragraph); cuda insists on the card")
    ap.add_argument("--glossary", help="the book's terms (lib/mt_glossary); 27 passes the manifest's `glossary`")
    ap.add_argument("--prompt", choices=PROMPTS, default=DEFAULT_PROMPT,
                    help="stock: the model's trained instruction; rules: the glossary's rule and the paragraph's "
                         "terms as well (llama, vertex); terms: the stock instruction with the paragraph's terms "
                         "written in English inside the Punjabi (sarvam) (default %s, from 28's measurement)"
                         % DEFAULT_PROMPT)
    ap.add_argument("--arbiter", choices=ARBITERS, default="none",
                    help="who retranslates a paragraph the checks refused, asked another way (arbiter_prompt): self "
                         "(the same local model), llama (a llama-server), vertex (metered by --arbiter-budget-usd)")
    ap.add_argument("--arbiter-budget-usd", type=float, default=1.0)
    ap.add_argument("--src", default=WRITINGS,
                    help="the folder of <work>.jsonl (default data/writings; a manifest's `corpus` puts it elsewhere)")
    ap.add_argument("--out", help="write here instead of <src>/<work>.en.jsonl (a comparison run)")
    args = ap.parse_args()

    folder = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    src = os.path.join(folder, args.work + ".jsonl")
    dst = args.out or os.path.join(folder, args.work + ".en.jsonl")
    if not os.path.exists(src):
        sys.exit("no %s (12_ingest_writings.py)" % src)
    meta = work_meta(src)
    lang = args.src_lang or meta.get("language") or "pa"
    if meta.get("translate") is False:
        # asked for by name, so done; but the manifest meant this work for its
        # own language only, and 27's plan will not carry the English on
        print("note: %s's manifest says translate: false (a work for display in %s); translating as asked"
              % (args.work, LANG_NAME.get(lang, lang)))
    if lang not in LANG_NAME:
        sys.exit("%s is in %r; this translates %s" % (args.work, lang, " or ".join(LANG_NAME.values())))
    from lib.mt_glossary import load as load_glossary
    glossary = load_glossary(args.glossary)
    if args.prompt == "rules" and not glossary["terms"]:
        print("note: --prompt rules without --glossary names no terms; only the no-brackets instruction is added")
    styles = set(args.styles.split(","))
    records = [r for r in read_jsonl(src) if r.get("style") in styles and r.get("lang", lang) != "en"]
    done = done_ids(dst)
    if args.only:
        want = {u.strip() for u in args.only.split(",")}
        todo = [r for r in records if r["unit_id"] in want]
    else:
        todo = [r for r in records if r["unit_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("%s (%s): %d paragraphs, %d done, %d to translate; prompt %s, glossary %d terms, arbiter %s"
          % (args.work, LANG_NAME[lang], len(records), len(done), len(todo), args.prompt, len(glossary["terms"]),
             args.arbiter))
    if not todo:
        return

    engine = make_engine(args.engine, args.model_path, lang, args.budget_usd, device=args.device)
    set_prompt(engine, args.prompt, glossary, LANG_NAME[lang])
    print("engine:", engine.model_desc())

    why_count: dict[str, int] = {}
    refused: list[dict] = []
    stripped = shown = written = 0
    t0 = time.time()

    def write(fh, rec, en, eng, prompt, arbiter=False):
        nonlocal shown, written
        fh.write(json.dumps({"id": rec["unit_id"], "unit_id": rec["unit_id"], "en": en, "engine": eng.name,
                             "model": eng.model_desc(), "prompt": prompt, **({"arbiter": True} if arbiter else {})},
                            ensure_ascii=False) + "\n")
        written += 1
        if shown < args.show:
            shown += 1
            print("  %s:" % lang.upper(), rec["text"][:160])
            print("  EN:", en[:200])

    with open(dst, "a", encoding="utf-8", newline="\n") as fh:
        if fh.tell() == 0:                             # a new file: its header first
            fh.write(json.dumps({"_meta": {"work": args.work, "engine": engine.name, "model": engine.model_desc(),
                                           "prompt": args.prompt, "glossary": os.path.basename(args.glossary or "") or None,
                                           "arbiter": args.arbiter}}, ensure_ascii=False) + "\n")
        # paragraphs of a similar length share a batch so little of it is padding
        size = max(1, args.batch) if hasattr(engine, "translate_many") else 1
        order = sorted(range(len(todo)), key=lambda i: len(todo[i]["text"]))
        n = 0
        for b in range(0, len(order), size):
            batch = [todo[i] for i in order[b:b + size]]
            try:
                answers = translate_batch(engine, [prepare(args.prompt, glossary, r["text"]) for r in batch],
                                          LANG_NAME[lang])
            except SystemExit:
                raise
            except Exception as err:                     # noqa: BLE001
                print("  %s: %s" % (batch[0]["unit_id"], err))
                for rec in batch:
                    why_count["error"] = why_count.get("error", 0) + 1
                    refused.append(rec)
                continue
            for rec, raw in zip(batch, answers):
                n += 1
                en, why, k = answer(raw, rec["text"], lang, glossary)
                stripped += k
                if why:
                    key = why.split(":")[0] if why.startswith("term") else why
                    why_count[key] = why_count.get(key, 0) + 1
                    print("  refused %s: %s" % (rec["unit_id"], why))
                    refused.append(rec)
                    continue
                write(fh, rec, en, engine, args.prompt)
            fh.flush()
            if (b // size) % 10 == 0:
                print("  %d/%d (%.2fs each)" % (n, len(todo), (time.time() - t0) / max(n, 1)), flush=True)

        twice = 0
        if refused and args.arbiter != "none":
            if args.arbiter == "self":
                arbiter = engine
            elif args.arbiter == "vertex":
                arbiter = make_engine("vertex", None, lang, args.arbiter_budget_usd)
            else:
                arbiter = make_engine("llama", None, lang)
            again = arbiter_prompt(arbiter, args.prompt)
            print("arbiter %s: %d refused paragraphs, prompt %s" % (arbiter.model_desc(), len(refused), again))
            for rec, en, why in arbitrate(arbiter, refused, lang, glossary, LANG_NAME[lang], again):
                if why:
                    twice += 1
                    print("  refused twice %s: %s" % (rec["unit_id"], why))
                    continue
                write(fh, rec, en, arbiter, again, arbiter=True)
            fh.flush()
    print("done: %d written, %d refused (%s), %d refused twice, %d glosses stripped, %.0fs -> %s"
          % (written, len(refused), ", ".join("%s %d" % kv for kv in sorted(why_count.items())) or "none",
             twice, stripped, time.time() - t0, dst))


if __name__ == "__main__":
    main()
