# NCAIR LMS Onboarding Assistant (version 2.0)

A local, retrieval-augmented chatbot that helps new interns and NYSC members of **NCAIR (National Center for Artificial Intelligence and Robotics)** register on the NCAIR Learning Management System (LMS). It answers questions about registration steps, programme rules, attendance and portal links in **English, Hausa, Yorùbá, Igbo and Nigerian Pidgin**, using only the official NCAIR onboarding manual and programme guidelines. It runs on your own machine through [Ollama](https://ollama.com); no cloud API is needed.

> **Versions.** *Version 1.0* is the original keyword-routed assistant, kept in `comparison/old_app.py` so the two can be compared. *Version 2.0* is everything else in this repository. (Inside the comparison scripts, 1.0 is called "Old" and 2.0 is called "New"; the function `agentic_rag_orchestrator_v3` in `src/orchestrator.py` is just the historical name of the entry point.)

---

## 1. Main idea

Onboarding at NCAIR is a one-time, in-person process: after orientation, interns go to the 50-Seater Hall in the e-Government Building, are added to the LMS by staff, receive an invitation email, and complete ten numbered registration steps on their phones. Every step is a place a first-time user can get stuck (the email lands in spam, a button will not open, a password is rejected), and the official guidance exists only as an 11-page image-based PDF manual and a text guideline file.

A language model that answers from memory is unreliable for exactly the facts that matter here (attendance thresholds, upload limits, URLs). So the assistant follows three rules:

1. **Decide what the user is asking** (a link, a step in the process, or a policy question) with a language model that can only choose from real tools.
2. **Answer only from retrieved context**, and say so honestly when the documents do not cover the question.
3. **Check the result in code**: any URL the model invents is removed, and any link the answer must contain is added.

## 2. Methodology

```
user message ──► Router (LLM, JSON-schema constrained) ──► Tools ──► Answer LLM (grounded in context) ──► Guardrails ──► reply
                 language + English rewrite + 0-2 calls     get_portal_link        reply in the user's language       only real URLs,
                                                            get_onboarding_step                                       required links present
                                                            search_knowledge_base (hybrid RAG)
```

**Step by step (`src/orchestrator.py`)**

1. **Input.** The message plus the last 3 conversation turns.
2. **Routing (`src/router.py`).** The router model returns one JSON object: the detected language (`en`, `ha`, `yo`, `ig`, `pcm`, `other`), a standalone English rewrite of the question, and 0 to 2 tool calls. Ollama's structured output is given a JSON schema generated from the tool registry, so the model can only name tools, page ids and step numbers that exist. If the output cannot be parsed after one retry, or contains no valid call, the English question is searched in the knowledge base. **There is no keyword routing anywhere.**
3. **Tools (`src/tools.py`).** Tools return *context*, never final text:
   - `get_portal_link(page_id)`: title, URL and description of one LMS/NCAIR page (6 pages in `data/portal_pages.json`).
   - `get_onboarding_step(step)`: the manual text for step 1 to 10, or `all`, with the related page link.
   - `search_knowledge_base(query)`: hybrid retrieval over the manual and guidelines (below).
4. **Answer generation (`src/system_prompt.py`).** The answer model writes the reply in the user's language with strict rules: use only the supplied context, copy URLs exactly, say "I don't have that information" and send the user to staff at the hall when the context does not answer, keep button and page names as written.
5. **Guardrails (`src/orchestrator.py`).** URLs that are not in the page registry or the tool context are removed; up to two required page links the model left out are appended.
6. **Offline fallback.** If no model is reachable, the user receives the retrieved text instead of an error.

**Retrieval (`src/retrieval.py`)**

- *Lexical:* pure-Python BM25 (k1 = 1.5, b = 0.75) over tokens that are lower-cased, diacritic-folded (so Yorùbá/Igbo tone marks never block a match), stop-word filtered and lightly stemmed.
- *Dense (optional):* `paraphrase-multilingual-MiniLM-L12-v2` embeddings, cached on disk in `index/` keyed by a hash of the corpus, fused with BM25 by reciprocal rank fusion (constant 60). Without `sentence-transformers` (or offline on first run) the app uses BM25 only.
- *Chunks:* the manual (23 sections) and guidelines (6 sections) become 29 chunks of at most 700 characters, split on sentence boundaries. The top 4 chunks are returned, plus the sibling chunks of the best hit (up to 1,200 characters) so a rule split across chunks is not lost.
- *Abstention:* `coverage` is the idf-weighted share of the query's terms found in the best passages. Below **0.25** the model is told nothing relevant was found (hard abstain); between **0.25 and 0.5** the passages are passed on but flagged low-confidence, and the model must say the topic is not covered unless they clearly answer the question.

**Models (`src/config.py`)**

| Role | Default |
|---|---|
| Router and answer model | `llama3.2:3b` (Ollama) |
| Hausa / Yorùbá / Igbo answers | **N-ATLaS** (8B, quantized GGUF) when installed; if it is missing or fails, the default model answers. Pidgin is not an N-ATLaS language and stays on the default model. |

Other profiles exist for experiments: `NCAIR_PROFILE=natlas` (N-ATLaS routes and answers everything) and `mixed` (Llama routes, N-ATLaS answers).

**Data-driven design.** Links live in `data/portal_pages.json`; the ten onboarding steps are read from `data/ncair_manual.md`; tool schemas, prompts and the link guardrail are generated from those files. Adding a page or a tool is a data edit, not a code change.

**Evaluation.** `evaluation/benchmark.json` holds 76 questions written by the project team from the manual and guidelines (knowledge base 25, steps 12, multilingual 12, links 10, out of scope 9, small talk 3, multi-tool 3, follow-up 2). Each lists expected tool calls, required facts, expected links, gold source sections and whether the assistant should abstain. `comparison/compare_old_new.py` runs versions 1.0 and 2.0 on the same questions and scores seven metrics: router (right tool), answers (all facts present), links (right page), retrieval (facts in retrieved passages), grounding (answerable questions wrongly refused, lower is better), speed (median seconds per answer, lower is better) and N-ATLaS (Hausa/Yorùbá/Igbo cases written by N-ATLaS with the right tool and link).

### Recorded results (76 questions)

| Metric | Version 1.0 | Version 2.0 |
|---|---|---|
| Router: right tool | 72.4% | 85.5% |
| Answers: all facts present | 61.9% | 76.2% |
| Links: right page | 45.0% | 85.0% |
| Retrieval: facts retrieved | 82.8% | 93.1% |
| Grounding: wrongly refuses answerable (lower is better) | 9.5% | 16.7% |
| Speed: median time per answer (lower is better) | 2.7 s | 6.0 s |
| N-ATLaS: multilingual cases passed (9 cases) | - | 55.6% |

Version 2.0 is better on routing, answers, links and retrieval, but slower and it wrongly refuses more answerable questions. On English-only retrieval (27 questions) both versions score 88.9%. For the 9 Hausa/Yorùbá/Igbo cases: written by N-ATLaS 9/9, right tool 7/9, right link 7/9, language detected 4/9. **Speed depends on the machine** (CPU/GPU, model size and quantization, whether the model is already loaded, other load), so it is a relative indicator, not a constant.

---

## 3. Setup

### 3.1 Prerequisites

| Need | Why |
|---|---|
| **Python 3.10 or newer** (developed on 3.12) | the application |
| **[Ollama](https://ollama.com/download)** | runs the language models locally |
| Free disk/RAM for the models | `llama3.2:3b` is about 2 GB; N-ATLaS Q4_K_M is about 4.9 GB (optional) |
| Internet, **once** | to download the models and the embedding model; afterwards everything runs offline |
| *Optional:* `poppler-utils` and `tesseract-ocr` | only to re-run OCR (`scripts/ingest_pdf.py`) or the version 1.0 comparison |

### 3.2 Install

```bash
# 1. Get the project and open a terminal in its folder
cd DSA_Project_Final

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install every Python dependency
pip install -r requirements.txt

# 4. Download the default model (Ollama must be installed and running)
ollama pull llama3.2:3b

# 5. (Recommended) download the multilingual specialist, registered in Ollama as "natlas"
scripts/setup_natlas.sh Q4_K_M       # needs bash; on Windows use WSL/Git Bash, or see docs/NATLAS_SETUP.md
```

Without step 5 the assistant still works: Hausa, Yorùbá and Igbo questions are then answered by `llama3.2:3b`. See `docs/NATLAS_SETUP.md` for other quantizations, the manual `ollama cp` route, and the N-ATLaS licence terms.

### 3.3 Run

```bash
python src/app.py
```

Gradio prints a local address (by default `http://127.0.0.1:7860`); open it in a browser. The first start with dense retrieval downloads the embedding model and builds `index/` (a few seconds to minutes); later starts reuse the cache. To skip it, or to work fully offline from the start: `NCAIR_USE_DENSE=no python src/app.py` (BM25 only).

Useful switches: `NCAIR_DEBUG=1` shows the routing trace under each reply; `NCAIR_SHARE=1` creates a public Gradio link (**off by default**).

### 3.4 Check that it works

```bash
python scripts/live.py --part retrieval   # retriever only, no model needed
python scripts/live.py --part full        # whole pipeline, needs Ollama
NCAIR_PROFILE=natlas python scripts/check_natlas.py   # N-ATLaS smoke test: JSON routing, multilingual replies, latency
python src/eval_retrieval.py --verbose    # retrieval hit@k / MRR / abstention, no model needed
```

`scripts/live.py --part` also accepts `router` and `tools` to watch one stage at a time.

### 3.5 Configuration (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `NCAIR_PROFILE` | `llama` | `llama`, `natlas` or `mixed` (see Models) |
| `NCAIR_ROUTER_MODEL` / `NCAIR_ANSWER_MODEL` | from profile | override the Ollama model names |
| `NCAIR_MULTILINGUAL_MODEL` | `natlas` | Hausa/Yorùbá/Igbo specialist; set to `""` to disable |
| `OLLAMA_HOST` | `http://localhost:11434` | where Ollama listens |
| `NCAIR_LLM_TIMEOUT` | `180` | seconds allowed per model call |
| `NCAIR_USE_DENSE` | `auto` | `auto`, `yes` or `no` (dense retrieval) |
| `NCAIR_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | dense model |
| `NCAIR_TOP_K` | `4` | passages returned |
| `NCAIR_MIN_COVERAGE` / `NCAIR_SOFT_COVERAGE` | `0.25` / `0.5` | hard and soft abstention thresholds |
| `NCAIR_DEBUG` / `NCAIR_SHARE` | `0` / `0` | routing trace / public link |

Generation settings (temperature, token limits, context size) are in `src/config.py`.

### 3.6 Evaluate

```bash
python src/eval_benchmark.py --stage context    # tools + retrieval ceiling, no model needed
python src/eval_benchmark.py --stage routing    # router vs expected tool calls (needs Ollama)
python src/eval_benchmark.py --stage answers    # full pipeline: facts, links, abstention (needs Ollama)
python comparison/natlas_eval.py                # N-ATLaS on its own, on the Hausa/Yorùbá/Igbo cases
python comparison/compare_old_new.py            # version 1.0 vs 2.0 on the 7 metrics (needs everything above installed)
```

Results are written to `evaluation/results_*.json` and `comparison/results/` (both git-ignored and re-creatable).

### 3.7 Updating the knowledge

Edit the data files; no code change is needed. The embedding cache is keyed by a hash of the corpus, so it rebuilds itself.

- Policies and rules: `data/ncair_knowledge_base.txt` (numbered headings, `1. TITLE`).
- Manual sections and onboarding steps: `data/ncair_manual.md` (`## id | title` headings with `@type`, `@step`, `@link`, `@page` metadata).
- Links: `data/portal_pages.json`. Router examples: `data/router_examples.json`.
- To re-OCR the PDF manual: `scripts/ingest_pdf.py` (needs `poppler-utils` and `tesseract-ocr`), then compare the output with `ncair_manual.md` by hand.

---

## 4. File structure

```
DSA_Project_Final/
├── README.md                  this file
├── requirements.txt           all Python dependencies
├── .gitignore                 ignores __pycache__, index/, models/*.gguf, results, .env
├── src/                       the application
│   ├── app.py                 Gradio chat interface (entry point)
│   ├── config.py              models, thresholds, paths; every setting can be overridden by environment variable
│   ├── orchestrator.py        full turn: route -> tools -> answer -> guardrails; offline fallback
│   ├── router.py              LLM router with JSON-schema constrained output
│   ├── tools.py               tool registry (portal link, onboarding step, knowledge-base search)
│   ├── retrieval.py           BM25 + optional dense retrieval, fusion, abstention signal
│   ├── knowledge.py           loads pages, parses manual and guidelines into chunks
│   ├── system_prompt.py       answer-generation prompts
│   ├── llm.py                 Ollama client (standard library only) and a scripted fake client for tests
│   ├── eval_retrieval.py      retrieval metrics, no model needed
│   └── eval_benchmark.py      context / routing / answers evaluation and scoring helpers
├── data/
│   ├── NCAIR LMS ONBOARDING MANUAL.pdf   official manual (11 pages, image-based)
│   ├── ncair_manual.md        curated, structured copy of the manual (23 sections, incl. the 10 steps)
│   ├── ncair_knowledge_base.txt   programme guidelines (6 sections)
│   ├── portal_pages.json      every link the assistant may give, with a `verified` flag
│   ├── router_examples.json   few-shot examples for the router prompt
│   └── raw/manual_ocr.txt     raw OCR of the manual, kept for reference
├── evaluation/benchmark.json  76 benchmark questions with expected tools, facts, links and sources
├── comparison/
│   ├── old_app.py             version 1.0 (keyword router + FAISS), kept for comparison
│   ├── compare_old_new.py     seven-metric comparison of versions 1.0 and 2.0
│   └── natlas_eval.py         evaluates N-ATLaS on the Hausa/Yorùbá/Igbo questions
├── scripts/
│   ├── setup_natlas.sh        downloads quantized N-ATLaS and registers it in Ollama as "natlas"
│   ├── check_natlas.py        smoke test of the model setup
│   ├── live.py                interactive test of one pipeline stage
│   └── ingest_pdf.py          one-time OCR of the manual PDF
├── docs/NATLAS_SETUP.md       N-ATLaS download, quantizations, profiles, licence notes
└── notebooks/ncair_rag_prototype.ipynb   original Colab prototype (history only)
```

Created automatically and not shipped: `src/__pycache__/` (Python bytecode), `index/` (cached embeddings, `emb_<hash>.npy`), `models/` (downloaded GGUF files), `evaluation/results_*.json`, `comparison/results/`, `.gradio/`.

---

## 5. Limitations

- **Narrow knowledge.** It knows only what is in the manual and guidelines. Anything else is refused by design, and changes to the real onboarding process are not picked up until the data files are edited by hand.
- **Unverified content.** The `courses`, `profile` and `contact` URLs and the support e-mail were carried over from version 1.0, are not in the official manual, and are flagged `verified: false` in `data/portal_pages.json`. The support e-mail is not shown to users until it is marked verified. Check them against the live site before release.
- **Small local models make mistakes.** `llama3.2:3b` sometimes picks the wrong tool (router accuracy is 85.5% on the benchmark), mislabels the language (4 of 9 correct on Hausa/Yorùbá/Igbo cases), or fills in details not in the context. The guardrails fix URLs only, not other facts.
- **Over-cautious abstention.** 16.7% of answerable benchmark questions were wrongly refused. The thresholds (0.25 / 0.5) were tuned on part of the benchmark and lexical overlap cannot fully separate paraphrased in-scope questions from out-of-scope ones.
- **Speed.** Every reply needs at least two model calls, so version 2.0 is slower than version 1.0 (6.0 s versus 2.7 s median in the recorded run). Times depend heavily on the hardware, model size and whether models are already loaded; N-ATLaS (8B) is heavier than Llama 3.2 3B.
- **Evaluation limits.** The 76 questions were written by the project team from the same documents the assistant retrieves from, there is no held-out split, and each system was run once, so scores are in-distribution and small differences may not be stable. Scoring is string matching on facts and URLs; it cannot judge fluency or faithfulness. The Hausa/Yorùbá/Igbo/Pidgin questions are unreviewed drafts (three per language) and need native-speaker review.
- **Language quality.** N-ATLaS's own model card rated Yorùbá lowest (2.69/5 versus 4.21 English, 3.98 Hausa, 3.87 Igbo). Nigerian Pidgin is not a supported N-ATLaS language, and quantization costs some accuracy. Treat non-English answers as needing review.
- **N-ATLaS terms.** Use is capped at 1,000 active end-users per 30 days without a commercial licence, some uses are forbidden, and the attribution shown in the app footer is required. Read the terms on the model page before any public deployment.
- **Scope of the app.** No user accounts, no storage of conversations, only the last 3 turns are used as history, context windows are 4,096 tokens, and the public share link is off by default. It guides users but cannot register anyone or see their LMS account.

---

## 6. Acknowledgements

Developed for **NCAIR, the National Center for Artificial Intelligence and Robotics**, a special purpose vehicle of NITDA under the Federal Ministry of Communications, Innovation and Digital Economy.

Project facilitators: **Stephen Ayuba** and **Victor**.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.
