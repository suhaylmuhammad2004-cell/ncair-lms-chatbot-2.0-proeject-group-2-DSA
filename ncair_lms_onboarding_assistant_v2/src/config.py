"""Central configuration. Everything that used to be scattered constants
(model names, thresholds, paths) lives here and can be overridden with
environment variables, so no code edit is needed to switch models.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
INDEX_DIR = os.path.join(ROOT, "index")  # cached embeddings (git-ignored)
EVAL_DIR = os.path.join(ROOT, "evaluation")

PAGES_FILE = os.path.join(DATA_DIR, "portal_pages.json")
MANUAL_FILE = os.path.join(DATA_DIR, "ncair_manual.md")
KB_FILE = os.path.join(DATA_DIR, "ncair_knowledge_base.txt")
ROUTER_EXAMPLES_FILE = os.path.join(DATA_DIR, "router_examples.json")

# ---------------------------------------------------------------- models
# DEFAULT: Llama (llama3.2:3b, served by Ollama) routes and answers everything.
# N-ATLaS is used ONLY to write the answer when the user's language is Hausa,
# Yoruba or Igbo (see MULTILINGUAL_* below). If N-ATLaS is not installed, those
# questions are simply answered by the default model.
# Other profiles are for experiments: "natlas" = N-ATLaS for everything.
PROFILES = {
    "llama": {"router": "llama3.2:3b", "answer": "llama3.2:3b"},
    "natlas": {"router": "natlas", "answer": "natlas"},
    "mixed": {"router": "llama3.2:3b", "answer": "natlas"},
}
PROFILE = os.environ.get("NCAIR_PROFILE", "llama")
MULTILINGUAL_MODEL = os.environ.get("NCAIR_MULTILINGUAL_MODEL", "natlas")  # "" disables it
MULTILINGUAL_LANGS = {"ha", "yo", "ig"}  # Pidgin is not a listed N-ATLaS language, so it stays on the default
ROUTER_MODEL = os.environ.get("NCAIR_ROUTER_MODEL", PROFILES[PROFILE]["router"])
ANSWER_MODEL = os.environ.get("NCAIR_ANSWER_MODEL", PROFILES[PROFILE]["answer"])
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Generation settings. The N-ATLaS model card uses temperature 0.1 and
# repetition penalty 1.12; its context length is ~8k, we keep prompts small.
ROUTER_OPTIONS = {"temperature": 0.0, "num_predict": 200, "num_ctx": 4096}
ANSWER_OPTIONS = {"temperature": 0.1, "repeat_penalty": 1.12, "num_predict": 500, "num_ctx": 4096}
LLM_TIMEOUT_S = float(os.environ.get("NCAIR_LLM_TIMEOUT", "180"))

# ------------------------------------------------------------- retrieval
# Multilingual so that a Hausa/Yoruba/Igbo question can still match English
# passages if the router fails and the raw query is searched directly.
EMBEDDING_MODEL = os.environ.get(
    "NCAIR_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
USE_DENSE = os.environ.get("NCAIR_USE_DENSE", "auto")  # auto | yes | no
TOP_K = int(os.environ.get("NCAIR_TOP_K", "4"))
# Two-level abstention (see retrieval.coverage()):
#  * below MIN_COVERAGE  -> hard abstain: the LLM gets "no relevant information".
#  * below SOFT_COVERAGE -> the passages are passed on but flagged low-confidence,
#    and the LLM must say "not covered" unless they clearly answer the question.
# MIN_COVERAGE was tuned on part of the benchmark (lowest value with 0
# false abstentions). The benchmark has no held-out split, so its scores are not
# independent of this choice. Lexical overlap cannot separate paraphrased in-scope
# questions from out-of-scope ones any better than that. The soft level and the
# LLM's own grounding rule cover the rest. Re-tune if you switch to dense-only.
MIN_COVERAGE = float(os.environ.get("NCAIR_MIN_COVERAGE", "0.25"))
SOFT_COVERAGE = float(os.environ.get("NCAIR_SOFT_COVERAGE", "0.5"))
MAX_CHUNK_CHARS = 700

# ------------------------------------------------------------------- app
MAX_CALLS_PER_TURN = 2
HISTORY_TURNS = 3
SHOW_DEBUG = os.environ.get("NCAIR_DEBUG", "0") == "1"
GRADIO_SHARE = os.environ.get("NCAIR_SHARE", "0") == "1"  # public link is OFF by default

# Required by the N-ATLaS terms of use for public deployments.
ATTRIBUTION = (
    "N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and "
    "Digital Economy, and powered by Awarri Technologies."
)
