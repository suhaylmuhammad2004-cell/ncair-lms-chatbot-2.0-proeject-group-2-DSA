"""Type a question, see one part of the system respond in real time.

  python scripts/live.py --part retrieval   # retriever only: top passages, scores, coverage, abstain (no model)
  python scripts/live.py --part router      # router only: language, English rewrite, tool calls (needs Ollama)
  python scripts/live.py --part tools       # router + tools: the context the answer model would receive (needs Ollama)
  python scripts/live.py --part full        # whole pipeline: route -> tools -> answer -> guardrails (needs Ollama)
Type 'quit' to exit.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from retrieval import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="full", choices=["retrieval", "router", "tools", "full"])
a = ap.parse_args()

retriever = Retriever()
if a.part == "retrieval":
    handler = None
else:
    from orchestrator import Assistant
    from tools import ToolBox
    asst = Assistant(toolbox=ToolBox(retriever))
    print(f"router={config.ROUTER_MODEL} answer={config.ANSWER_MODEL}")

print(f"[{a.part}] retriever={retriever.method}. Ask something ('quit' to exit).")
while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q.lower() in ("quit", "exit"):
        break
    t = time.perf_counter()
    if a.part == "retrieval":
        r = retriever.search(q)
        print(f"coverage={r.coverage:.2f} abstain={r.abstain} low_confidence={r.low_confidence}")
        for h in r.hits:
            print(f"  #{h.rank} {h.chunk.section_id}  score={h.score:.3f}\n      {h.chunk.text[:160]}...")
    elif a.part == "router":
        d = asst.router.route(q)
        print(f"language={d.language}\nenglish_query={d.english_query}\ncalls={d.calls}\nfallback={d.fallback} {d.error}")
    elif a.part == "tools":
        d = asst.router.route(q)
        print(f"calls={d.calls} fallback={d.fallback}")
        for c in d.calls:
            res = asst.tb.execute(c["tool"], {k: v for k, v in c.items() if k != "tool"})
            print(f"\n--- {res.tool} ok={res.ok} abstain={res.abstain} sources={res.sources}\n{res.context}")
    else:
        r = asst.reply(q)
        print(f"calls={r.tool_calls} sources={r.sources}\n\n{r.text}\n\n{r.debug()}")
    print(f"({time.perf_counter() - t:.1f}s)")
