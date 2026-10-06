"""Evaluate retrieval on its own, with no LLM needed.

For every benchmark case routed to search_knowledge_base we ask: did the
retriever surface at least one gold source section, and how high? We also
measure abstention: out-of-scope questions should be flagged as "not covered",
answerable ones should not.

Usage:
    python src/eval_retrieval.py
    python src/eval_retrieval.py --min-coverage 0.4 --dense no
"""
import argparse
import json
import os
import sys

import config
from retrieval import Retriever


def load_cases():
    with open(os.path.join(config.EVAL_DIR, "benchmark.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


def retrieval_query(case) -> str:
    # Multilingual cases carry a gold English rewrite. In production the router
    # produces this rewrite; here we isolate the retriever from the router.
    return case.get("english_query") or case["query"]


def evaluate(retriever: Retriever, cases, k: int, min_coverage: float):
    answerable = [
        c for c in cases
        if not c["should_abstain"] and c["gold_sources"] and c["category"] != "followup"
        and any(call["tool"] == "search_knowledge_base" for call in c["expected_calls"])
    ]
    oos = [c for c in cases if c["should_abstain"]]
    rows, hits_at, rr = [], {1: 0, 3: 0, k: 0}, 0.0
    for c in answerable:
        res = retriever.search(retrieval_query(c), k=max(k, 5), min_coverage=min_coverage)
        ranked = res.section_ids
        pos = next((i + 1 for i, s in enumerate(ranked) if s in c["gold_sources"]), None)
        for n in hits_at:
            if pos is not None and pos <= n:
                hits_at[n] += 1
        rr += 1.0 / pos if pos else 0.0
        rows.append((c["id"], pos, res.abstain, round(res.coverage, 2), ranked[:3]))
    n = max(len(answerable), 1)
    false_abstain = sum(1 for r in rows if r[2])
    oos_rows = []
    correct_abstain = 0
    for c in oos:
        res = retriever.search(retrieval_query(c), k=k, min_coverage=min_coverage)
        correct_abstain += int(res.abstain)
        oos_rows.append((c["id"], res.abstain, round(res.coverage, 2)))
    return {
        "n_answerable": len(answerable),
        "hit@1": hits_at[1] / n,
        "hit@3": hits_at[3] / n,
        f"hit@{k}": hits_at[k] / n,
        "mrr": rr / n,
        "false_abstain_rate": false_abstain / n,
        "n_out_of_scope": len(oos),
        "correct_abstain_rate": correct_abstain / max(len(oos), 1),
        "rows": rows,
        "oos_rows": oos_rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--min-coverage", type=float, default=config.MIN_COVERAGE)
    ap.add_argument("--dense", default=config.USE_DENSE, choices=["auto", "yes", "no"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    retriever = Retriever(dense=a.dense)
    cases = load_cases()
    r = evaluate(retriever, cases, a.k, a.min_coverage)
    print(f"retriever={retriever.method}  k={a.k}  min_coverage={a.min_coverage}")
    print(f"answerable questions: {r['n_answerable']}   out-of-scope questions: {r['n_out_of_scope']}")
    for key in ("hit@1", "hit@3", f"hit@{a.k}", "mrr", "false_abstain_rate", "correct_abstain_rate"):
        print(f"  {key:22s} {r[key]:.3f}")
    if a.verbose:
        print("\nanswerable (id, rank of first gold section, abstained?, coverage, top-3 sections)")
        for row in r["rows"]:
            flag = "  MISS" if row[1] is None else ""
            print("  ", row, flag)
        print("\nout-of-scope (id, abstained?, coverage)")
        for row in r["oos_rows"]:
            print("  ", row)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in r.items()} | {"retriever": retriever.method}, f, indent=2)
    return r


if __name__ == "__main__":
    main()
