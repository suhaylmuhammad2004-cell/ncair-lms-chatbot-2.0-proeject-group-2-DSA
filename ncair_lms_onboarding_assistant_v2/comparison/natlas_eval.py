#!/usr/bin/env python3
"""Evaluate N-ATLaS on its own (no Old app, no router, no Llama in the loop).

Each Hausa/Yoruba/Igbo question in New's benchmark is answered by N-ATLaS directly, using the SAME
answer prompt New uses, with the context produced by the GOLD tool calls (so router mistakes and the
language detector cannot affect the result, only N-ATLaS can).

Per question it records: did N-ATLaS answer at all (no error/timeout/empty), seconds, the right link is
present (link questions), expected facts present (numbers/times such as 75% or 5:00 PM), invented URLs,
and a rough flag when the reply looks like English instead of the user's language. It saves every answer
so you can read them. Quality of the language itself needs a native speaker: this script cannot judge it.

Usage (from New's root; this file lives in comparison/):
  python comparison/natlas_eval.py
  python comparison/natlas_eval.py --limit 3              # quick test
  python comparison/natlas_eval.py --also-llama           # same prompts on llama3.2:3b for side-by-side
  python comparison/natlas_eval.py --timeout 900          # seconds allowed per answer (default 900)
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGLISH_HINTS = {"the", "and", "you", "your", "to", "of", "is", "are", "for", "with", "please", "can", "this", "that"}


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and sum(w in ENGLISH_HINTS for w in words) / len(words) > 0.18


def invented_urls(text: str, allowed: set) -> list:
    urls = re.findall(r"(?:https?://)?(?:lms\.|www\.)?ncair\.nitda\.gov\.ng[^\s)\]>\"']*", text, re.I)
    norm = lambda u: re.sub(r"^https?://", "", u.strip().rstrip(".,;:!?)").lower()).rstrip("/")
    return [u for u in urls if norm(u) not in allowed]


def run_model(model, cases, ctx_of, tb, eb, config, OllamaClient, answer_system_prompt, answer_user_prompt, timeout):
    llm = OllamaClient(model, timeout=timeout)
    if not llm.is_available():
        sys.exit(f"Model '{model}' is not available in Ollama (run `ollama list`).")
    allowed = {re.sub(r"^https?://", "", p.url.lower()).rstrip("/") for p in tb.pages.values()}
    rows = []
    for i, c in enumerate(cases, 1):
        eq = c.get("english_query") or c["query"]
        messages = [{"role": "system", "content": answer_system_prompt(tb.pages, c["language"])},
                    {"role": "user", "content": answer_user_prompt(c["query"], eq, ctx_of(c, eq))}]
        t0, text, err = time.perf_counter(), "", ""
        try:
            text = llm.chat(messages, options=config.ANSWER_OPTIONS)
        except Exception as e:  # timeout / connection / model error
            err = str(e)[:200]
        secs = time.perf_counter() - t0
        row = {
            "id": c["id"], "language": c["language"], "query": c["query"], "answer": text, "error": err,
            "answered": bool(text.strip()), "secs": round(secs, 1),
            "links_ok": (float(eb.links_ok(text, c["expected_links"], tb.pages)) if c["expected_links"] and text else
                         (0.0 if c["expected_links"] else None)),
            "facts_ok": (float(eb.facts_ok(text, c["expected_facts"])) if c["expected_facts"] and text else
                         (0.0 if c["expected_facts"] else None)),
            "invented_urls": invented_urls(text, allowed),
            "looks_english": looks_english(text) if text else None,
        }
        rows.append(row)
        print(f"[{model}] [{i}/{len(cases)}] {c['id']} answered={row['answered']} {row['secs']}s "
              f"links={row['links_ok']} facts={row['facts_ok']} english?={row['looks_english']} {err}")
    return rows


def summarize(rows):
    def m(key):
        v = [r[key] for r in rows if r[key] is not None]
        return f"{sum(v):.0f}/{len(v)}" if v else "n/a"
    times = [r["secs"] for r in rows if r["answered"]]
    return {
        "answered": f"{sum(r['answered'] for r in rows)}/{len(rows)}",
        "right link": m("links_ok"), "facts present": m("facts_ok"),
        "invented URLs": str(sum(len(r["invented_urls"]) for r in rows)),
        "looks like English (should be low)": f"{sum(bool(r['looks_english']) for r in rows)}/{sum(r['looks_english'] is not None for r in rows)}",
        "median s per answer": f"{statistics.median(times):.1f}" if times else "n/a",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate N-ATLaS on its own.")
    ap.add_argument("--new", default=os.path.join(HERE, ".."), help="New project folder")
    ap.add_argument("--model", default="", help="model to evaluate (default: New's multilingual model, 'natlas')")
    ap.add_argument("--also-llama", action="store_true", help="also run llama3.2:3b on the same prompts")
    ap.add_argument("--include-pidgin", action="store_true", help="also include the 3 Pidgin questions")
    ap.add_argument("--timeout", type=float, default=900, help="seconds allowed per answer")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", default=os.path.join(HERE, "results"), help="output folder")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.abspath(a.new), "src"))
    import config
    import eval_benchmark as eb
    from llm import OllamaClient
    from retrieval import Retriever
    from system_prompt import answer_system_prompt, answer_user_prompt
    from tools import ToolBox

    tb = ToolBox(Retriever())
    langs = set(config.MULTILINGUAL_LANGS) | ({"pcm"} if a.include_pidgin else set())
    cases = [c for c in eb.load_cases() if c["category"] == "multilingual" and c["language"] in langs]
    if a.limit:
        cases = cases[: a.limit]

    def ctx_of(c, english_query):
        parts = []
        for call in c["expected_calls"]:
            args = dict(call["args"])
            if call["tool"] == "search_knowledge_base":
                args["query"] = english_query
            parts.append(tb.execute(call["tool"], args).context)
        return "\n\n=====\n\n".join(parts)

    models = [a.model or config.MULTILINGUAL_MODEL]
    if a.also_llama:
        models.append("llama3.2:3b")

    results = {}
    for model in models:
        results[model] = run_model(model, cases, ctx_of, tb, eb, config, OllamaClient,
                                   answer_system_prompt, answer_user_prompt, a.timeout)

    lines = [f"# Standalone evaluation, {len(cases)} questions (gold tool context, same answer prompt as New)", ""]
    names = list(results)
    summ = {m: summarize(r) for m, r in results.items()}
    lines += ["| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in summ[names[0]]:
        lines.append(f"| {k} | " + " | ".join(summ[m][k] for m in names) + " |")
    lines += ["", "Language quality is NOT scored here: read the answers below with a native speaker if you can.", ""]
    for i, c in enumerate(cases):
        lines.append(f"## {c['id']} ({c['language']}): {c['query']}")
        for m in names:
            r = results[m][i]
            lines.append(f"**{m}** ({r['secs']}s){' ERROR: ' + r['error'] if r['error'] else ''}\n\n{r['answer']}\n")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "natlas_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(a.out, "natlas_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "\n".join(lines[:3 + len(summ[names[0]]) + 2]))
    print(f"\nSaved {os.path.join(a.out, 'natlas_eval.md')} (all answers) and natlas_eval.json")


if __name__ == "__main__":
    main()
