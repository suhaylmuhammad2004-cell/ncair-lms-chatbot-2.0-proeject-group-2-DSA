"""Score the assistant on evaluation/benchmark.json.

Stages (--stage):
  retrieval  retrieval-only metrics (hit@k, MRR, abstention). No LLM.   -> eval_retrieval.py
  context    "oracle context": run the GOLD tool calls and check the tool context contains the expected
             facts/links. No LLM. This is the ceiling the answer model can reach with the current
             retrieval and tools.
  routing    the router LLM picks tools; compared with the expected calls (needs Ollama).
  answers    full pipeline; checks facts, links and correct abstention in the final text (needs Ollama).
  all        context + routing + answers

Compare models:  NCAIR_PROFILE=natlas python src/eval_benchmark.py --stage routing
                 NCAIR_PROFILE=llama  python src/eval_benchmark.py --stage routing
All benchmark cases are used (there is no dev/test split).
"""
import argparse
import json
import os
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List

import config

ABSTAIN_PHRASES = [
    "not covered", "isn't covered", "is not covered", "not mentioned", "no information", "does not say",
    "doesn't say", "do not have", "don't have", "cannot find", "can't find", "couldn't find", "not available",
    "not in the official", "not provided", "does not provide", "doesn't provide", "does not give", "doesn't give",
    "not stated", "no mention",
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s)


def load_cases() -> List[Dict]:
    with open(os.path.join(config.EVAL_DIR, "benchmark.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


# ------------------------------------------------------------- scoring
def facts_ok(text: str, facts: List[List[str]]) -> bool:
    t = norm(text)
    return all(any(norm(alt) in t for alt in fact) for fact in facts)


def links_ok(text: str, link_ids: List[str], pages) -> bool:
    t = norm(text).replace("https://", "").replace("http://", "")
    return all(norm(pages[p].url).replace("https://", "").rstrip("/") in t for p in link_ids)


def abstained(text: str) -> bool:
    t = norm(text)
    return any(p in t for p in ABSTAIN_PHRASES)


def score_routing(case: Dict, predicted: List[Dict]) -> Dict[str, bool]:
    exp_names = sorted(c["tool"] for c in case["expected_calls"])
    got_names = sorted(c["tool"] for c in predicted)
    ok_sets = [sorted(s) for s in case.get("acceptable_tool_sets", [])]
    tool_ok = got_names == exp_names or got_names in ok_sets
    args_ok = True
    for ec in case["expected_calls"]:
        if ec["tool"] == "search_knowledge_base":
            continue
        match = [p for p in predicted if p["tool"] == ec["tool"]]
        args_ok &= any(all(str(p.get(k)) == str(v) for k, v in ec["args"].items()) for p in match)
    if got_names in ok_sets and got_names != exp_names:
        args_ok = True  # alternative accepted route: arguments not comparable
    return {"tool_ok": tool_ok, "args_ok": bool(args_ok), "exact": tool_ok and bool(args_ok)}


def summarize(rows: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        by_cat["ALL"].append(r)
    out = {}
    for cat, rs in by_cat.items():
        out[cat] = {"n": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            out[cat][k] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def print_table(title: str, summ: Dict, keys: List[str]) -> None:
    print(f"\n== {title} ==")
    print(f"{'category':14s} {'n':>3s} " + " ".join(f"{k:>12s}" for k in keys))
    for cat in sorted(summ, key=lambda c: (c == "ALL", c)):
        row = summ[cat]
        print(f"{cat:14s} {row['n']:3d} " + " ".join(
            f"{'-' if row[k] is None else format(row[k], '.2f'):>12s}" for k in keys))


# -------------------------------------------------------------- stages
def stage_context(cases, toolbox) -> List[Dict]:
    """Run the gold calls; no LLM involved."""
    rows = []
    for c in cases:
        if c["category"] in ("smalltalk",):
            continue
        ctx, retrieved = [], []
        for call in c["expected_calls"]:
            args = dict(call["args"])
            if call["tool"] == "search_knowledge_base":
                args["query"] = c.get("english_query") or c["query"]
            res = toolbox.execute(call["tool"], args)
            ctx.append(res.context)
            retrieved += res.sources
        if not c["expected_calls"]:
            continue
        text = "\n".join(ctx)
        if c["should_abstain"]:
            facts = None
        else:
            facts = facts_ok(text, c["expected_facts"]) if c["expected_facts"] else None
        pages_ok = links_ok(text, c["expected_links"], toolbox.pages) if c["expected_links"] else None
        rows.append({
            "id": c["id"], "category": c["category"],
            "context_has_facts": None if facts is None else float(facts),
            "context_has_links": None if pages_ok is None else float(pages_ok),
            "gold_retrieved": (float(any(s in c["gold_sources"] for s in retrieved))
                               if c["gold_sources"] and any(x["tool"] == "search_knowledge_base" for x in c["expected_calls"]) else None),
            "abstained_correctly": (float("NO RELEVANT INFORMATION" in text or "CONFIDENCE: LOW" in text)
                                    if c["should_abstain"] else None),
        })
    return rows


def stage_routing(cases, assistant) -> List[Dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        d = assistant.router.route(c["query"], c.get("history"))
        s = score_routing(c, d.calls)
        rows.append({
            "id": c["id"], "category": c["category"], "query": c["query"],
            "expected": c["expected_calls"], "predicted": d.calls, "router_fallback": float(d.fallback),
            "language_ok": (float(d.language == c["language"]) if c["category"] == "multilingual" else None),
            **{k: float(v) for k, v in s.items()},
        })
        print(f"[{i}/{len(cases)}] {c['id']} exact={s['exact']} -> {[(p['tool'], {k: v for k, v in p.items() if k != 'tool'}) for p in d.calls]}")
    return rows


def stage_answers(cases, assistant) -> List[Dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        r = assistant.reply(c["query"], c.get("history"))
        s = score_routing(c, r.tool_calls)
        english = c["language"] == "en"
        row = {
            "id": c["id"], "category": c["category"], "query": c["query"], "answer": r.text,
            "exact_route": float(s["exact"]),
            "facts_ok": (float(facts_ok(r.text, c["expected_facts"])) if (c["expected_facts"] and english) else None),
            "links_ok": (float(links_ok(r.text, c["expected_links"], assistant.tb.pages)) if c["expected_links"] else None),
            "abstain_ok": (float(abstained(r.text)) if (c["should_abstain"] and english) else None),
            "false_abstain": (float(abstained(r.text)) if (not c["should_abstain"] and english and c["expected_facts"]) else None),
            "removed_urls": len(r.removed_urls), "guardrail_added_link": float(bool(r.added_links)),
            "total_s": r.timings.get("total_s"),
        }
        rows.append(row)
        print(f"[{i}/{len(cases)}] {c['id']} route={s['exact']} facts={row['facts_ok']} links={row['links_ok']} t={row['total_s']}s")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["retrieval", "context", "routing", "answers", "all"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (quick smoke run)")
    a = ap.parse_args(argv)

    cases = load_cases()
    if a.limit:
        cases = cases[: a.limit]
    results = {"profile": config.PROFILE, "router_model": config.ROUTER_MODEL, "answer_model": config.ANSWER_MODEL,
               "n_cases": len(cases)}

    if a.stage == "retrieval":
        import eval_retrieval
        eval_retrieval.main(["--verbose"])
        return

    from retrieval import Retriever
    from tools import ToolBox
    toolbox = ToolBox(Retriever())

    if a.stage in ("context", "all"):
        rows = stage_context(cases, toolbox)
        keys = ["context_has_facts", "context_has_links", "gold_retrieved", "abstained_correctly"]
        summ = summarize(rows, keys)
        print_table(f"ORACLE CONTEXT (gold tool calls, no LLM) retriever={toolbox.retriever.method}", summ, keys)
        results["context"] = {"rows": rows, "summary": summ}

    if a.stage in ("routing", "answers", "all"):
        from orchestrator import Assistant
        assistant = Assistant(toolbox=toolbox)
        if not assistant.router_llm.is_available():
            sys.exit(f"Model '{config.ROUTER_MODEL}' is not available in Ollama. See docs/NATLAS_SETUP.md.")
        if a.stage in ("routing", "all"):
            rows = stage_routing(cases, assistant)
            keys = ["tool_ok", "args_ok", "exact", "language_ok", "router_fallback"]
            summ = summarize(rows, keys)
            print_table(f"ROUTING router={config.ROUTER_MODEL}", summ, keys)
            results["routing"] = {"rows": rows, "summary": summ}
        if a.stage in ("answers", "all"):
            rows = stage_answers(cases, assistant)
            keys = ["exact_route", "facts_ok", "links_ok", "abstain_ok", "false_abstain", "guardrail_added_link"]
            summ = summarize(rows, keys)
            print_table(f"ANSWERS answer={config.ANSWER_MODEL}", summ, keys)
            times = [r["total_s"] for r in rows if r["total_s"] is not None]
            if times:
                print(f"\nlatency: median {statistics.median(times):.1f}s, max {max(times):.1f}s per question")
            results["answers"] = {"rows": rows, "summary": summ}

    out = os.path.join(config.EVAL_DIR, f"results_{config.PROFILE}_{a.stage}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
