"""
One interface over every OCR engine, local or paid.

An engine takes a page image and returns lines:

  {"n": 1, "bbox": [x0, y0, x1, y1], "text": "...", "conf": 0.0-1.0 | None,
   "words": [{"bbox", "text", "conf"}], "block": int, "par": int, "zone": str | None}

in pixels of the image it was given. Everything else -- normalisation, zones,
matching, voting -- happens later on these records, so an engine can be added
or swapped without touching the merge. "conf" is the engine's own number
rescaled to [0, 1]; None where the engine has none (dots.ocr), in which case
cross-engine agreement is the only signal, and that is the primary one anyway.

The engines, and why each is here (see the plan's engine table):

  tesseract  CPU, tessdata_best pan/eng/hin, word confidences, the baseline
             every other engine is measured against; gurmukhifix afterwards
             because the LSTM sometimes emits a sihari in glyph order
  surya      Surya 2, GPU, line confidences, the English first choice
  indicocr   Bodhan / AI4Bharat IndicOCR, 0.8B, the only model with a published
             Punjabi printed-text number; block confidences and a layout class
  dotsocr    dots.ocr, the best open score on Gurmukhi in the one benchmark that
             isolates the script; layout classes, no confidences
  vision     Google Cloud Vision DOCUMENT_TEXT_DETECTION, per-symbol
             confidence, Gurmukhi supported, $1.50 per 1,000 pages, whole pages
             only (it bills per image). PAID: every call goes through Budget.
  gemini     reserved for routed line crops in M3; refuses whole pages.

Imports are lazy and per engine, exactly as lib/mt_engines.py does for the
Vertex SDK: a machine without torch can still run Tesseract.
"""
from __future__ import annotations
import base64
import json
import os
import re
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import TESSDATA                                         # noqa: E402

VISION_USD_PER_PAGE = 0.0015
VISION_BATCH = 16
# where the Tesseract binary usually is when it is not on PATH
TESSERACT_EXES = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", "/usr/bin/tesseract",
                  "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
HF_GATE_HELP = """%s is a gated model on Hugging Face. To use it:
  1. sign in at https://huggingface.co and open https://huggingface.co/%s
  2. click "Agree and access repository" (it asks for your contact details)
  3. make a Read token at https://huggingface.co/settings/tokens
  4. in this venv run  hf auth login  (or  huggingface-cli login) and paste it,
     or set HF_TOKEN for one shell
then run this again. docs/ocr-runbook.md has the same steps."""


def is_gate_error(e: BaseException) -> bool:
    """A Hugging Face refusal: not logged in, or the licence not accepted."""
    if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"):
        return True
    text = str(e)
    return any(m in text for m in ("401", "403", "gated", "Access to model", "authenticated"))


def image_size(path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.width, im.height


def split_block(text: str, bbox: list, conf, zone=None, block: int = 0) -> list[dict]:
    """A multi-line block's text as line records, the box shared out by height."""
    parts = [p for p in (text or "").split("\n") if p.strip()]
    if not parts:
        return []
    x0, y0, x1, y1 = bbox
    step = (y1 - y0) / float(len(parts))
    out = []
    for i, p in enumerate(parts):
        out.append({"bbox": [int(x0), int(y0 + i * step), int(x1), int(y0 + (i + 1) * step)],
                    "text": p.strip(), "conf": conf, "words": [], "block": block, "par": 0, "zone": zone})
    return out


def number(lines: list[dict]) -> list[dict]:
    """Reading order top to bottom, left to right within a band; 1-based n."""
    lines = sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for i, ln in enumerate(lines, start=1):
        ln["n"] = i
    return lines


class Engine:
    name = "base"
    langs = {"pa", "en", "hi"}
    paid = False
    gpu = False

    def __init__(self, **opts):
        self.opts = opts

    def model_desc(self) -> str:
        return self.name

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        raise NotImplementedError

    def recognise_many(self, paths: list[str], lang: str) -> list[list[dict]]:
        return [self.recognise(p, lang) for p in paths]

    def cost_usd(self, pages: int) -> float:
        return 0.0


# --------------------------------------------------------------------------- tesseract

_GURMUKHIFIX = None


def gurmukhi_fix(text: str) -> str:
    """
    gurmukhifix's character corrector on one line (its language key is
    "punjabi"; "pa" has no config). Returns the line unchanged when the
    package is absent or raises -- a fixer must never lose a line.
    """
    global _GURMUKHIFIX
    if _GURMUKHIFIX is None:
        try:
            from gurmukhifix import CharacterCorrector
            _GURMUKHIFIX = CharacterCorrector("punjabi")
        except Exception:                               # noqa: BLE001
            _GURMUKHIFIX = False
    if not _GURMUKHIFIX:
        return text
    try:
        out = _GURMUKHIFIX.correct(text)
        fixed = out[0] if isinstance(out, tuple) else out
        return fixed if isinstance(fixed, str) and fixed.strip() else text
    except Exception:                                   # noqa: BLE001
        return text


class TesseractEngine(Engine):
    name = "tesseract"
    LANGS = {"pa": "pan+eng", "en": "eng", "hi": "hin+eng"}

    def __init__(self, psm: int = 3, oem: int = 1, tessdata: str | None = None,
                 tess_lang: str | None = None, **opts):
        super().__init__(**opts)
        import pytesseract
        if not shutil.which("tesseract"):
            found = next((p for p in TESSERACT_EXES if os.path.exists(p)), None)
            if found is None:
                raise SystemExit("tesseract is not on PATH and not at any of: %s" % ", ".join(TESSERACT_EXES))
            pytesseract.pytesseract.tesseract_cmd = found
        self.pt = pytesseract
        self.psm, self.oem = int(psm), int(oem)
        self.tess_lang = tess_lang                       # e.g. "pan" or "script/Gurmukhi", a bake-off variant
        self.tessdata = tessdata or TESSDATA
        # pytesseract passes config verbatim, so a quoted path reaches Tesseract
        # with its quotes; the environment variable is the reliable channel.
        os.environ["TESSDATA_PREFIX"] = self.tessdata

    def model_desc(self) -> str:
        return "tesseract %s psm%d oem%d %s%s" % (self.pt.get_tesseract_version(), self.psm, self.oem,
                                                  os.path.basename(self.tessdata.rstrip("\\/")),
                                                  (" " + self.tess_lang) if self.tess_lang else "")

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        from PIL import Image
        config = "--oem %d --psm %d" % (self.oem, self.psm)
        with Image.open(png_path) as im:
            d = self.pt.image_to_data(im, lang=self.tess_lang or self.LANGS[lang], config=config,
                                      output_type=self.pt.Output.DICT)
        return self.lines_from_data(d, lang)

    def recognise_region(self, img, bbox: list, lang: str, psm: int = 7, whitelist: str | None = None,
                         pad: int = 6) -> list[dict]:
        """
        One region of a page image (a numpy array or a PIL image) through
        Tesseract: lines with their boxes offset back into page coordinates.
        psm 7 reads the region as one line, 6 as a block, 8 as one word;
        `whitelist` limits the characters (the LSTM only partly honours it,
        so callers still filter what comes back). The notation grid reader
        uses it for a row strip or a single cell.
        """
        from PIL import Image
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if hasattr(img, "shape"):
            h, w = img.shape[:2]
            x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                return []
            im = Image.fromarray(img[y0:y1, x0:x1])
        else:
            w, h = img.size
            x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                return []
            im = img.crop((x0, y0, x1, y1))
        config = "--oem %d --psm %d" % (self.oem, int(psm))
        if whitelist:
            config += " -c tessedit_char_whitelist=" + whitelist
        d = self.pt.image_to_data(im, lang=self.tess_lang or self.LANGS[lang], config=config,
                                  output_type=self.pt.Output.DICT)
        lines = self.lines_from_data(d, lang)
        for ln in lines:
            ln["bbox"] = [ln["bbox"][0] + x0, ln["bbox"][1] + y0, ln["bbox"][2] + x0, ln["bbox"][3] + y0]
            for wd in ln.get("words", []):
                wd["bbox"] = [wd["bbox"][0] + x0, wd["bbox"][1] + y0, wd["bbox"][2] + x0, wd["bbox"][3] + y0]
        return lines

    @staticmethod
    def lines_from_data(d: dict, lang: str) -> list[dict]:
        """image_to_data's DICT -> line records; words grouped by (block, par, line)."""
        groups: dict = {}
        order: list = []
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            conf = float(d["conf"][i])
            if not txt or conf < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            if key not in groups:
                groups[key] = []
                order.append(key)
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            groups[key].append({"bbox": [x, y, x + w, y + h], "text": txt, "conf": round(conf / 100.0, 3)})
        lines = []
        for key in order:
            ws = groups[key]
            text = " ".join(w["text"] for w in ws)
            if lang == "pa":
                text = gurmukhi_fix(text)
            lines.append({
                "bbox": [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
                         max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)],
                "text": text, "conf": round(statistics.mean(w["conf"] for w in ws), 3),
                "words": ws, "block": int(key[0]), "par": int(key[1]), "zone": None})
        return number(lines)


# --------------------------------------------------------------------------- surya

class SuryaEngine(Engine):
    """Surya 2 when its inference manager exists, else the 0.x predictors."""
    name = "surya"
    gpu = True
    ZONES = {"PageHeader": "header", "Page-header": "header", "PageFooter": "stamp",
             "Page-footer": "stamp", "Footnote": "footnote"}

    def __init__(self, **opts):
        super().__init__(**opts)
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
            self.rec = RecognitionPredictor(SuryaInferenceManager())
            self.det = None
            self.v2 = True
        except ImportError:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            self.det = DetectionPredictor()
            self.rec = RecognitionPredictor()
            self.v2 = False

    def model_desc(self) -> str:
        try:
            import surya
            v = getattr(surya, "__version__", "?")
        except Exception:                                # noqa: BLE001
            v = "?"
        return "surya %s (%s)" % (v, "v2" if self.v2 else "v1")

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        if self.v2:
            result = self.rec([im])[0]
        else:
            result = self.rec([im], [[lang]], self.det)[0]
        lines: list[dict] = []
        text_lines = getattr(result, "text_lines", None)
        if text_lines:
            for i, tl in enumerate(text_lines):
                bbox = [int(v) for v in tl.bbox]
                lines.append({"bbox": bbox, "text": _strip_html(tl.text), "conf": _conf(tl.confidence),
                              "words": [], "block": i, "par": 0, "zone": None})
        else:
            for i, blk in enumerate(getattr(result, "blocks", []) or []):
                text = getattr(blk, "text", None) or _strip_html(getattr(blk, "html", "") or "")
                bbox = [int(v) for v in blk.bbox]
                zone = self.ZONES.get(str(getattr(blk, "label", "")))
                lines.extend(split_block(text, bbox, _conf(getattr(blk, "confidence", None)), zone, i))
        return number(lines)


def _conf(v):
    if v is None:
        return None
    v = float(v)
    return round(v / 100.0 if v > 1.0 else v, 3)


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


# --------------------------------------------------------------------------- indicocr

class IndicOCREngine(Engine):
    name = "indicocr"
    gpu = True
    ZONES = {"PageHeader": "header", "PageFooter": "stamp", "Footnote": "footnote"}

    def __init__(self, repo: str = "bodhan-ai/indic-ocr", **opts):
        super().__init__(**opts)
        self.repo = repo
        # the model code ships inside the Hugging Face repo beside its weights
        # (indic_ocr.py, idp_*.py); a local path is used as it is, a repo id is
        # fetched into the hub cache (1.9 GB) and put on sys.path
        try:
            local = repo if os.path.isdir(repo) else self._snapshot(repo)
            if local not in sys.path:
                sys.path.insert(0, local)
            from indic_ocr import IndicOCR
            self.parser = IndicOCR.from_pretrained(local)
        except SystemExit:
            raise
        except Exception as e:                        # the hub raises several types; the text says which
            if is_gate_error(e):
                raise SystemExit(HF_GATE_HELP % (repo, repo)) from e
            raise

    @staticmethod
    def _snapshot(repo: str) -> str:
        from huggingface_hub import snapshot_download
        try:
            return snapshot_download(repo, ignore_patterns=["assets/*"])
        except Exception as e:                        # noqa: BLE001
            if is_gate_error(e):
                raise SystemExit(HF_GATE_HELP % (repo, repo)) from e
            raise

    def model_desc(self) -> str:
        return "indicocr %s" % self.repo

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        page = self.parser.parse(png_path)
        lines: list[dict] = []
        for blk in sorted(page.get("blocks", []), key=lambda b: b.get("order", 0)):
            text = blk.get("text") or ""
            if not text.strip():
                continue
            bbox = [int(v) for v in blk.get("bbox_xyxy", [0, 0, 0, 0])]
            zone = self.ZONES.get(str(blk.get("type", "")))
            lines.extend(split_block(text, bbox, _conf(blk.get("conf")), zone, int(blk.get("order", 0))))
        return number(lines)


# --------------------------------------------------------------------------- dots.ocr

DOTS_CATEGORIES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
                   "Picture", "Section-header", "Table", "Text", "Title"]
DOTS_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's bbox, "
    "its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are %s.\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object.\n" % json.dumps(DOTS_CATEGORIES)
)


def smart_resize(w: int, h: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 11289600):
    """Qwen2-VL's resize rule, so a bbox the model returns can be mapped back."""
    import math
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return wb, hb


class DotsOCREngine(Engine):
    name = "dotsocr"
    gpu = True
    ZONES = {"Page-header": "header", "Page-footer": "stamp", "Footnote": "footnote"}

    # A 2130x2979 page at the model's default ceiling of 11.3 MP becomes ~14K
    # image tokens and the attention alone asks for 46 GB; 2.6 MP (~195 dpi
    # for these scans) fits a 12 GB card and keeps the glyphs legible.
    MAX_PIXELS = 2_600_000

    def __init__(self, backend: str = "transformers", model_path: str = "rednote-hilab/dots.ocr",
                 http: str | None = None, attn: str = "sdpa", max_new_tokens: int = 16000,
                 max_pixels: int | None = None, **opts):
        super().__init__(**opts)
        self.backend = "http" if http else backend
        self.http = http
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.max_pixels = int(max_pixels or self.MAX_PIXELS)
        try:
            from dots_ocr.utils.prompts import dict_promptmode_to_prompt
            self.prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]
        except Exception:                                # noqa: BLE001
            self.prompt = DOTS_PROMPT
        if self.backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.torch = torch
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation=attn, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def model_desc(self) -> str:
        return "dots.ocr %s via %s" % (self.model_path, self.backend)

    def _generate_transformers(self, im) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": im},
                                                 {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        # newer transformers processors emit a key the model's own code predates
        inputs.pop("mm_token_type_ids", None)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _generate_http(self, png_bytes: bytes) -> str:
        import urllib.request
        url = self.http.rstrip("/") + "/chat/completions"
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        body = {"model": self.opts.get("http_model", "model"), "temperature": 0.0,
                "max_tokens": self.max_new_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self.prompt}]}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        w0, h0 = im.size
        w1, h1 = smart_resize(w0, h0, max_pixels=self.max_pixels)
        im_r = im.resize((w1, h1), Image.LANCZOS) if (w1, h1) != (w0, h0) else im
        if self.backend == "http":
            import io
            buf = io.BytesIO()
            im_r.save(buf, format="PNG")
            raw = self._generate_http(buf.getvalue())
        else:
            raw = self._generate_transformers(im_r)
        elements = _parse_json_list(raw)
        sx, sy = w0 / float(w1), h0 / float(h1)
        lines: list[dict] = []
        for i, el in enumerate(elements):
            if not isinstance(el, dict) or "bbox" not in el:
                continue
            text = el.get("text") or ""
            cat = str(el.get("category", ""))
            if cat == "Picture" or not text.strip():
                continue
            b = el["bbox"]
            bbox = [int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)]
            lines.extend(split_block(_unmarkdown(text), bbox, None, self.ZONES.get(cat), i))
        return number(lines)


def _parse_json_list(raw: str) -> list:
    body = (raw or "").strip()
    body = re.sub(r"^```[a-z]*\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        return json.loads(body[start:end + 1])
    except ValueError:
        return []


def _unmarkdown(s: str) -> str:
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", s)
    return s.strip()


# --------------------------------------------------------------------------- google vision

class VisionEngine(Engine):
    """
    DOCUMENT_TEXT_DETECTION with a language hint; lines rebuilt from symbols.

    Whole pages only: Vision bills per image, so a page cut into line crops
    costs a page per crop. Budget is mandatory; without one the engine refuses
    to construct, so no code path can call it unmetered.
    """
    name = "vision"
    paid = True
    HINTS = {"pa": "pa", "en": "en", "hi": "hi"}
    BREAKS_EOL = {3, 5}       # EOL_SURE_SPACE, LINE_BREAK

    def __init__(self, budget=None, book: str = "", **opts):
        super().__init__(**opts)
        if budget is None:
            raise SystemExit("vision: a Budget is required (--budget-usd)")
        from google.cloud import vision
        self.vision = vision
        # the project that pays: without it the client bills the credentials'
        # default quota project, which may be another project entirely (and
        # one where the API is not enabled)
        project = os.environ.get("GCP_PROJECT")
        if project:
            from google.api_core.client_options import ClientOptions
            self.client = vision.ImageAnnotatorClient(client_options=ClientOptions(quota_project_id=project))
        else:
            self.client = vision.ImageAnnotatorClient()
        self.budget = budget
        self.book = book

    def model_desc(self) -> str:
        return "google-cloud-vision DOCUMENT_TEXT_DETECTION"

    def cost_usd(self, pages: int) -> float:
        return round(pages * VISION_USD_PER_PAGE, 6)

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        return self.recognise_many([png_path], lang)[0]

    def recognise_many(self, paths: list[str], lang: str) -> list[list[dict]]:
        out: list[list[dict]] = []
        for i in range(0, len(paths), VISION_BATCH):
            chunk = paths[i:i + VISION_BATCH]
            cost = self.cost_usd(len(chunk))
            if not self.budget.allows(cost):
                raise SystemExit("vision: %.4f USD would exceed the monthly cap of %.2f (spent %.4f)"
                                 % (cost, self.budget.cap, self.budget.spent_this_month()))
            reqs = []
            for p in chunk:
                with open(p, "rb") as fh:
                    content = fh.read()
                reqs.append(self.vision.AnnotateImageRequest(
                    image=self.vision.Image(content=content),
                    features=[self.vision.Feature(type_=self.vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
                    image_context=self.vision.ImageContext(language_hints=[self.HINTS.get(lang, lang)])))
            resp = self.client.batch_annotate_images(requests=reqs)
            self.budget.charge(self.name, self.book, len(chunk), cost)
            for r in resp.responses:
                if r.error and r.error.message:
                    raise RuntimeError("vision: %s" % r.error.message)
                out.append(self._lines(r))
        return out

    def _lines(self, resp) -> list[dict]:
        lines: list[dict] = []
        fta = resp.full_text_annotation
        if not fta or not fta.pages:
            return lines
        bi = 0
        for page in fta.pages:
            for block in page.blocks:
                bi += 1
                for pi, para in enumerate(block.paragraphs):
                    words: list[dict] = []
                    for word in para.words:
                        txt = "".join(s.text for s in word.symbols)
                        xs = [v.x for v in word.bounding_box.vertices]
                        ys = [v.y for v in word.bounding_box.vertices]
                        confs = [s.confidence for s in word.symbols if s.confidence is not None]
                        words.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": txt,
                                      "conf": round(statistics.mean(confs), 3) if confs else None})
                        last = word.symbols[-1] if word.symbols else None
                        brk = last.property.detected_break.type_ if last and last.property else 0
                        if int(brk) in self.BREAKS_EOL:
                            lines.append(self._line(words, bi, pi))
                            words = []
                    if words:
                        lines.append(self._line(words, bi, pi))
        return number([ln for ln in lines if ln["text"]])

    @staticmethod
    def _line(words: list[dict], block: int, par: int) -> dict:
        confs = [w["conf"] for w in words if w["conf"] is not None]
        return {"bbox": [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words),
                         max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)],
                "text": " ".join(w["text"] for w in words),
                "conf": round(statistics.mean(confs), 3) if confs else None,
                "words": words, "block": block, "par": par, "zone": None}


# --------------------------------------------------------------------------- the PDF's own text layer

class PdfTextEngine(Engine):
    """
    The OCR layer a scan already carries (lib/ocr_pages.page_text_layer), as an
    engine: no confidence, no cost, one more independent reading for the vote.
    Its boxes are in the rendered page's pixels but were laid out on the
    UNDESKEWED page; at the small angles measured here that is within a line.
    """
    name = "pdftext"

    def __init__(self, pages: dict | None = None, **opts):
        super().__init__(**opts)
        self.by_file = {p["file"]: p for p in (pages or {}).get("pages", [])}

    def model_desc(self) -> str:
        return "pdf text layer"

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        rec = self.by_file.get(os.path.basename(png_path))
        if not rec:
            return []
        lines = [{"bbox": list(t["bbox"]), "text": t["text"], "conf": None, "words": [],
                  "block": i, "par": 0, "zone": None} for i, t in enumerate(rec.get("text_layer", []))]
        return number(lines)


# --------------------------------------------------------------------------- gemini (routed only)

class GeminiEngine(Engine):
    name = "gemini"
    paid = True

    def __init__(self, budget=None, **opts):
        super().__init__(**opts)
        if budget is None:
            raise SystemExit("gemini: a Budget is required (--budget-usd)")
        self.budget = budget

    def recognise(self, png_path: str, lang: str) -> list[dict]:
        raise SystemExit("gemini reads packed line crops only (22_ocr_merge.py --routed, M3); "
                         "it never takes whole pages")


ENGINES = {"tesseract": TesseractEngine, "surya": SuryaEngine, "indicocr": IndicOCREngine,
           "dotsocr": DotsOCREngine, "vision": VisionEngine, "gemini": GeminiEngine,
           "pdftext": PdfTextEngine}


def load_engine(name: str, **opts) -> Engine:
    if name not in ENGINES:
        raise SystemExit("unknown engine %r; one of %s" % (name, ", ".join(sorted(ENGINES))))
    return ENGINES[name](**opts)


def write_page(path: str, meta: dict, lines: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for ln in lines:
            fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_page(path: str):
    """(meta, lines) of one per-engine page file."""
    meta, lines = {}, []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if "_meta" in rec:
                meta = rec["_meta"]
            else:
                lines.append(rec)
    return meta, lines


def timed(fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, round(time.time() - t0, 2)
