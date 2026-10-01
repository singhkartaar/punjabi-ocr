import os

ROOT = os.environ.get("ROOT_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CORPUS_DB = os.environ.get("CORPUS_DB") or os.path.join(ROOT, "data", "corpus.sqlite")
# The other protected sources an OCR'd book may quote (Dasam Bani, Bhai Gurdas):
# 25_fetch_granths.py writes it, 22_ocr_merge.py reads it, and both go on without it.
GRANTHS_DB = os.environ.get("GRANTHS_DB") or os.path.join(ROOT, "data", "granths.sqlite")
ARTIFACTS = os.environ.get("ARTIFACTS_DIR") or os.path.join(ROOT, "artifacts")
VECTORS = os.path.join(ROOT, "data", "vectors")
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(ROOT, "vendor", "models", "bge-small-en-v1.5")
MAHANKOSH_DB = os.environ.get("MAHANKOSH_DB") or os.path.join(ROOT, "data", "mahankosh.sqlite")
MAHANKOSH_CACHE = os.path.join(ROOT, "data", "cache", "mahankosh")

# The machine-translation exchange (docs/darpan-translation.md): the Node
# pipeline writes inputs here, 11_translate_mt.py fills the outputs, and
# 08-apply-mt.js imports them.
MT_IN = os.path.join(ROOT, "data", "mt", "in")
MT_OUT = os.path.join(ROOT, "data", "mt", "out")

# The OCR working set (20_ocr_pages.py .. 24_ocr_eval.py): page images, one
# directory of output per engine, ground truth, and the ledger every paid
# engine writes to before and after a request. Gitignored whole.
OCR_DIR = os.environ.get("OCR_DIR") or os.path.join(ROOT, "data", "ocr")
OCR_COSTS = os.path.join(OCR_DIR, "costs.jsonl")
# Tesseract's tessdata_best models. Program Files is not writable without
# elevation, so the models live under vendor/ (gitignored) and are passed to
# Tesseract with --tessdata-dir; TESSDATA_PREFIX overrides.
TESSDATA = os.environ.get("TESSDATA_PREFIX") or os.path.join(ROOT, "vendor", "tessdata")
# Keertan notation books (29_notation_parse.py .. 32_build_notations_db.py):
# one directory per book with notations.jsonl, images.json and the crops.
# Gitignored whole; the crops are published as release assets.
NOTATIONS_DIR = os.environ.get("NOTATIONS_DIR") or os.path.join(ROOT, "data", "notations")
# The review ledger (34_notation_review.py): one append-only file a book of
# the reviewer's verdicts, and the accepted records with their images as
# fixtures. Small, committed to the data repository; the images are re-cut.
REVIEW_DIR = os.environ.get("REVIEW_DIR") or os.path.join(os.path.dirname(NOTATIONS_DIR), "review")

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(VECTORS, exist_ok=True)

# The English index lives at the top level of artifacts/ and data/vectors/ (it
# was there first and the app treats it as the default). Every other index --
# "pa-plain", "pa-enriched", and the chosen "pa" -- gets a subdirectory of each,
# with the same file names inside, so one loader serves all of them.
DEFAULT_INDEX = "en"


def index_paths(name: str | None):
    """(vectors_dir, artifacts_dir) for a named index; both created on demand."""
    if not name or name == DEFAULT_INDEX:
        return VECTORS, ARTIFACTS
    v, a = os.path.join(VECTORS, name), os.path.join(ARTIFACTS, name)
    os.makedirs(v, exist_ok=True)
    os.makedirs(a, exist_ok=True)
    return v, a
