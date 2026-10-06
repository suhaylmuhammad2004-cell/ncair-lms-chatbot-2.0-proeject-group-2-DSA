#!/usr/bin/env python3
"""Compare the OLD chatbot (app.py) with the NEW one (DSA_Project_Final.zip).

Both are run on the same questions (New's evaluation/benchmark.json, 76 cases) and scored with
New's own scoring helpers (src/eval_benchmark.py), so the numbers are comparable.

The 7 metrics
  1. Router     right tool chosen             (higher is better)
  2. Answers    all expected facts present    (higher is better)
  3. Links      right page link in the answer (higher is better)
  4. Retrieval  expected facts are in the retrieved passages (higher is better)
  5. Grounding  wrongly refuses answerable questions          (LOWER is better)
  6. Speed      median seconds per answer                     (LOWER is better)
  7. N-ATLaS    Hausa/Yoruba/Igbo cases handled by the N-ATLaS multilingual model (higher is better)
A metric the Old app does not have is shown as "-". A metric that could not be measured on this
machine (e.g. Ollama not running) is shown as "n/a".

Usage
  python compare_old_new.py --old app.py --new DSA_Project_Final.zip
  python compare_old_new.py --old app.py --new DSA_project/ --limit 10      # quick smoke run

Requirements
  * Ollama running with llama3.2:3b (and `natlas` for the N-ATLaS row). Without it the Router (Old),
    Retrieval and the table still work; the LLM-based metrics show n/a.
  * The Old app's own dependencies: langchain, langchain-community, faiss-cpu, sentence-transformers,
    pytesseract + pdf2image (+ tesseract and poppler installed). gradio/IPython are NOT needed here
    (they are stubbed because the UI is never launched).
  * Old app.py reads ../data relative to itself. By default it is given New's data/ folder
    (PDF + ncair_knowledge_base.txt). Use --old-data-dir if Old had a different data folder.

Nothing inside New is modified: this script only imports New's code.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import zipfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DASH, NA = "-", "n/a"
OLD_MODEL = "llama3.2:3b"  # model hard-coded in Old app.py (query_local_llama default)

# What each system actually has. Old has a (keyword) router, FAISS retrieval, links, answers and one LLM.
# It has no N-ATLaS / multilingual model, so that row is a dash.
OLD_HAS = {"router": True, "answers": True, "links": True, "retrieval": True,
           "grounding": True, "speed": True, "natlas": False}

METRICS = [
    ("router", "1. Router: right tool"),
    ("answers", "2. Answers: all facts present"),
    ("links", "3. Links: right page"),
    ("retrieval", "4. Retrieval: facts retrieved"),
    ("grounding", "5. Grounding: wrongly refuses answerable (lower is better)"),
    ("speed", "6. Speed: time per answer (lower is better)"),
    ("natlas", "7. N-ATLaS: multilingual model"),
]


# ------------------------------------------------------------------ loading
def find_new_root(path: str) -> str:
    """Accept the project folder, a folder containing it, or the .zip."""
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        dest = tempfile.mkdtemp(prefix="new_project_")
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        path = dest
    for root, _dirs, files in os.walk(path):
        if os.path.basename(root) == "src" and "config.py" in files and "orchestrator.py" in files:
            return os.path.dirname(root)
    sys.exit(f"Could not find the New project (src/config.py, src/orchestrator.py) under: {path}")


def load_new(root: str):
    sys.path.insert(0, os.path.join(root, "src"))
    import config
    import eval_benchmark as eb
    from llm import OllamaClient
    from orchestrator import Assistant
    from retrieval import Retriever
    from tools import ToolBox
    return config, eb, OllamaClient, Assistant, Retriever, ToolBox


class OldApp:
    """Loads the Old app.py as a module (UI stubbed) and records which tool each question used."""

    TOOL_NAMES = {  # Old function -> equivalent New tool name
        "get_portal_link": "get_portal_link",
        "get_step_guidance": "get_onboarding_step",
        "search_ncair_knowledge_base": "search_knowledge_base",
    }

    def __init__(self, app_path: str, data_dir: str, ollama_host: str):
        self.tmp = tempfile.mkdtemp(prefix="old_app_")
        src = os.path.join(self.tmp, "src")
        os.makedirs(src)
        shutil.copy(app_path, os.path.join(src, "app.py"))
        shutil.copytree(data_dir, os.path.join(self.tmp, "data"))  # Old reads ../data next to itself

        stubbed = ["IPython", "IPython.display", "gradio"]  # UI only; never launched here
        for name in stubbed:
            sys.modules[name] = mock.MagicMock()
        try:
            spec = importlib.util.spec_from_file_location("old_app", os.path.join(src, "app.py"))
            self.mod = importlib.util.module_from_spec(spec)
            print("Loading Old app (runs its OCR + FAISS build, this can take a while)...")
            spec.loader.exec_module(self.mod)
        except ImportError as e:
            sys.exit(f"Old app needs a package that is not installed: {e}\n"
                     "Install: pip install langchain langchain-community langchain-text-splitters faiss-cpu "
                     "sentence-transformers pytesseract pdf2image requests")
        finally:
            for name in stubbed:
                sys.modules.pop(name, None)

        self.mod.OLLAMA_API_URL = ollama_host.rstrip("/") + "/api/generate"
        self.calls = []
        self._search = self.mod.search_ncair_knowledge_base  # untouched, used for the retrieval metric
        for fn, tool in self.TOOL_NAMES.items():
            self._wrap(fn, tool)

    def _wrap(self, fn, tool):
        orig = getattr(self.mod, fn)

        def wrapper(*a, **k):
            self.calls.append(tool)
            return orig(*a, **k)

        setattr(self.mod, fn, wrapper)

    def ask(self, query: str):
        self.calls = []
        t0 = time.perf_counter()
        text = self.mod.agentic_rag_orchestrator(query)
        secs = time.perf_counter() - t0
        return text, list(dict.fromkeys(self.calls)), secs

    def retrieve(self, query: str) -> str:
        return self._search(query, top_k=2)  # exactly what Old's orchestrator uses

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------ scoring
SUPPORT_HINT = re.compile(r"contact (?:the )?support|support@ncair|ask (?:an? )?ncair staff|ask staff", re.I)


def refused(eb, text: str, facts) -> bool:
    """Did the answer refuse / say 'not covered'? Same detector for Old and New.

    New says "not covered / I don't have..." (eb.abstained). Old is instructed to answer
    "contact support at ..." when it cannot answer, so that counts as a refusal too, but only when
    the answer does not actually contain the expected facts."""
    return eb.abstained(text) or bool(SUPPORT_HINT.search(text) and not eb.facts_ok(text, facts))


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_pct(v):
    return NA if v is None else f"{100 * v:.1f}%"


def fmt_secs(v):
    return NA if v is None else f"{v:.1f} s"


def ratio(vals):
    vals = [v for v in vals if v is not None]
    return f"{int(sum(vals))}/{len(vals)}" if vals else "0/0"


# ----------------------------------------------------------------- main run
def evaluate(args):
    root = find_new_root(args.new)
    config, eb, OllamaClient, Assistant, Retriever, ToolBox = load_new(root)
    cases = eb.load_cases()
    if args.limit:
        cases = cases[: args.limit]
    pages = None

    data_dir = args.old_data_dir or os.path.join(root, "data")
    old = OldApp(args.old, data_dir, config.OLLAMA_HOST)
    toolbox = ToolBox(Retriever())
    assistant = Assistant(toolbox=toolbox)
    pages = toolbox.pages

    old_llm_ok = OllamaClient(OLD_MODEL, host=config.OLLAMA_HOST).is_available()
    new_llm_ok = assistant.router_llm.is_available() and assistant.answer_llm.is_available()
    natlas_ok = assistant._multilingual_available()
    print(f"Ollama: Old model '{OLD_MODEL}' {'ready' if old_llm_ok else 'NOT available'}; "
          f"New router '{config.ROUTER_MODEL}'/answer '{config.ANSWER_MODEL}' {'ready' if new_llm_ok else 'NOT available'}; "
          f"N-ATLaS '{config.MULTILINGUAL_MODEL}' {'ready' if natlas_ok else 'NOT available'}")

    # Warm-up so model load time is not counted as answer time.
    if not args.no_warmup:
        if old_llm_ok:
            old.ask("What is the minimum attendance?")
        if new_llm_ok:
            assistant.reply("What is the minimum attendance?")
            if natlas_ok:
                assistant.reply("Ina neman link din shiga LMS na NCAIR.")

    old_rows, new_rows = {}, {}
    for i, c in enumerate(cases, 1):
        q, hist = c["query"], c.get("history")
        text, calls, secs = old.ask(q)  # Old has no history parameter
        old_rows[c["id"]] = {"text": text, "calls": calls, "secs": secs}
        if new_llm_ok:
            r = assistant.reply(q, hist)
            new_rows[c["id"]] = {"text": r.text, "calls": [x["tool"] for x in r.tool_calls],
                                 "secs": r.timings.get("total_s"), "model": r.answer_model,
                                 "language": r.language, "fallback": r.answer_fallback}
        print(f"[{i}/{len(cases)}] {c['id']:5s} old={calls} new={new_rows.get(c['id'], {}).get('calls', NA)}")

    # ---- eligibility (identical for both systems)
    def has_kb(c):
        return any(x["tool"] == "search_knowledge_base" for x in c["expected_calls"])

    answer_cases = [c for c in cases if c["language"] == "en" and c["expected_facts"] and not c["should_abstain"]]
    link_cases = [c for c in cases if c["expected_links"]]
    retr_cases = [c for c in cases if has_kb(c) and c["expected_facts"] and not c["should_abstain"]]
    native_cases = [c for c in cases if c["category"] == "multilingual" and c["language"] in config.MULTILINGUAL_LANGS]

    def tool_ok(c, calls):
        return float(eb.score_routing(c, [{"tool": t} for t in calls])["tool_ok"])

    def score(rows, llm_ok, router_needs_llm):
        s = {}
        s["router"] = mean([tool_ok(c, rows[c["id"]]["calls"]) for c in cases]) if (llm_ok or not router_needs_llm) and rows else None
        if llm_ok:
            s["answers"] = mean([float(eb.facts_ok(rows[c["id"]]["text"], c["expected_facts"])) for c in answer_cases])
            s["links"] = mean([float(eb.links_ok(rows[c["id"]]["text"], c["expected_links"], pages)) for c in link_cases])
            s["grounding"] = mean([float(refused(eb, rows[c["id"]]["text"], c["expected_facts"])) for c in answer_cases])
            times = [rows[c["id"]]["secs"] for c in cases if rows[c["id"]]["secs"] is not None]
            s["speed"] = statistics.median(times) if times else None
            s["speed_mean"] = statistics.mean(times) if times else None
        else:
            s.update(answers=None, links=None, grounding=None, speed=None, speed_mean=None)
        return s

    old_s = score(old_rows, old_llm_ok, router_needs_llm=False)  # Old's router is keyword-based: no LLM needed
    new_s = score(new_rows, new_llm_ok, router_needs_llm=True)

    # ---- retrieval (no LLM involved for either)
    def retr(get_text, only_en=False):
        cs = [c for c in retr_cases if (c["language"] == "en" or not only_en)]
        return mean([float(eb.facts_ok(get_text(c), c["expected_facts"])) for c in cs]), len(cs)

    old_retr, n_retr = retr(lambda c: old.retrieve(c["query"]))  # Old has no query rewriting: raw question
    new_retr, _ = retr(lambda c: toolbox.execute("search_knowledge_base", {"query": c.get("english_query") or c["query"]}).context)
    old_retr_en, n_retr_en = retr(lambda c: old.retrieve(c["query"]), only_en=True)
    new_retr_en, _ = retr(lambda c: toolbox.execute("search_knowledge_base", {"query": c["query"]}).context, only_en=True)
    old_s["retrieval"], new_s["retrieval"] = old_retr, new_retr

    # ---- N-ATLaS (New only)
    if new_llm_ok:
        by_natlas = [float(new_rows[c["id"]].get("model") == config.MULTILINGUAL_MODEL) for c in native_cases]
        route_ok = [tool_ok(c, new_rows[c["id"]]["calls"]) for c in native_cases]
        link_ok = [float(eb.links_ok(new_rows[c["id"]]["text"], c["expected_links"], pages)) if c["expected_links"] else 1.0
                   for c in native_cases]
        lang_ok = [float(new_rows[c["id"]]["language"] == c["language"]) for c in native_cases]
        passed = [float(a and b and l) for a, b, l in zip(by_natlas, route_ok, link_ok)]
        new_s["natlas"] = mean(passed)
        natlas_detail = (f"{len(native_cases)} Hausa/Yoruba/Igbo cases: written by N-ATLaS {ratio(by_natlas)}, "
                         f"right tool {ratio(route_ok)}, right link {ratio(link_ok)}, language detected {ratio(lang_ok)}; "
                         f"pass = all three of N-ATLaS + tool + link")
    else:
        new_s["natlas"] = None
        natlas_detail = "not measured (Ollama/models unavailable)"
    old_s["natlas"] = None

    # ---- table
    def cell(system_has, value, kind):
        if not system_has:
            return DASH
        return fmt_secs(value) if kind == "speed" else fmt_pct(value)

    lines = ["| Metric | Old | New |", "|---|---|---|"]
    table = {}
    for key, label in METRICS:
        o = cell(OLD_HAS[key], old_s.get(key), key)
        n = cell(True, new_s.get(key), key)
        lines.append(f"| {label} | {o} | {n} |")
        table[key] = {"old": o, "new": n}
    md = "\n".join(lines)

    details = [
        f"Cases: {len(cases)} total | answers/grounding {len(answer_cases)} (English, facts expected) | "
        f"links {len(link_cases)} | retrieval {n_retr} (knowledge-base questions with facts)",
        f"Retrieval, English questions only ({n_retr_en}): Old {fmt_pct(old_retr_en)} vs New {fmt_pct(new_retr_en)} "
        f"(Old searches the raw question with top_k=2; New uses the router's English rewrite with top_k={config.TOP_K}; "
        f"the table value uses each system's own input)",
        f"Speed: median / mean seconds per answer: Old {fmt_secs(old_s.get('speed'))} / {fmt_secs(old_s.get('speed_mean'))}, "
        f"New {fmt_secs(new_s.get('speed'))} / {fmt_secs(new_s.get('speed_mean'))}",
        f"N-ATLaS: {natlas_detail}",
        "Old has no conversation-history input, so follow-up questions are sent to it without history.",
        "Old's router is keyword rules; its 'tool' is derived from which function it called "
        "(get_portal_link / get_step_guidance -> get_onboarding_step / FAISS search -> search_knowledge_base).",
    ]
    if not old_llm_ok:
        details.append(f"Old: model '{OLD_MODEL}' unavailable -> answer-based metrics are n/a (router and retrieval still measured).")
    if not new_llm_ok:
        details.append("New: its router and answer models are unavailable -> all New metrics except Retrieval are n/a.")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n\n" + "\n".join(f"- {d}" for d in details) + "\n")
    with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"table": table, "details": details, "old_rows": old_rows, "new_rows": new_rows},
                  f, indent=2, ensure_ascii=False)

    print("\n" + md + "\n")
    for d in details:
        print("- " + d)
    print(f"\nSaved {os.path.join(out_dir, 'comparison.md')} and comparison.json")
    old.cleanup()


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Compare Old app.py with New (DSA_Project_Final) on 7 metrics.")
    ap.add_argument("--old", default=os.path.join(here, "old_app.py"), help="path to the Old app.py")
    ap.add_argument("--new", default=os.path.join(here, ".."), help="New project: .zip or folder")
    ap.add_argument("--old-data-dir", default="", help="data folder for Old (default: New's data/)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N benchmark cases")
    ap.add_argument("--no-warmup", action="store_true", help="skip the model warm-up calls before timing")
    ap.add_argument("--out", default=os.path.join(here, "results"), help="folder for comparison.md / comparison.json")
    evaluate(ap.parse_args(argv))


if __name__ == "__main__":
    main()
