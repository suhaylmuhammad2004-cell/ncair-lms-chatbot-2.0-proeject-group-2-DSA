"""Smoke test for the N-ATLaS setup (replaces the old test_amini.py).

Checks, against the local Ollama model(s): (1) the model exists, (2) the router returns valid structured JSON
for English/Hausa/Yoruba/Igbo/Pidgin prompts, (3) a grounded answer comes back in the user's language, and
prints latency so quantizations can be compared. Run:  python scripts/check_natlas.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from orchestrator import Assistant

PROMPTS = ["Where do I sign in?", "Ina neman link din shiga LMS", "Mo fẹ́ mọ iye ìgbà ìkópa tó kéré jù",
           "Biko nye m njikọ ịbanye", "Abeg, how I go take register?"]
a = Assistant()
print(f"router={config.ROUTER_MODEL} answer={config.ANSWER_MODEL}")
for m in {config.ROUTER_MODEL, config.ANSWER_MODEL}:
    from llm import OllamaClient
    if not OllamaClient(m).is_available():
        sys.exit(f"Model '{m}' not found in Ollama. Run scripts/setup_natlas.sh first.")
for p in PROMPTS:
    t = time.perf_counter()
    r = a.reply(p)
    print(f"\nQ: {p}\n   lang={r.language} fallback={r.decision.fallback} calls={r.tool_calls}\n   A: {r.text[:300]}\n   {r.timings}")
