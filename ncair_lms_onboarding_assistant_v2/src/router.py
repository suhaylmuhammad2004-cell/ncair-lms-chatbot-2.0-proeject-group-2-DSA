"""LLM-driven structured router.

The model sees the user's message (and a little history) and returns ONE JSON
object: detected language, an English rewrite, and 0-2 tool calls. The JSON
schema is generated from the tool registry, so the model can only name real
tools, real page ids and real step numbers. If the model still produces
something unusable, we degrade to "search the knowledge base with the user's
own words" -- never to keyword rules.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from llm import LLMError
from tools import ToolBox

LANGS = ["en", "ha", "yo", "ig", "pcm", "other"]


@dataclass
class RouterDecision:
    language: str = "en"
    english_query: str = ""
    calls: List[Dict[str, Any]] = field(default_factory=list)
    fallback: bool = False      # True when the router output was unusable
    error: str = ""
    raw: str = ""


class Router:
    def __init__(self, llm, toolbox: ToolBox, examples_path: str = config.ROUTER_EXAMPLES_FILE):
        self.llm, self.tb = llm, toolbox
        with open(examples_path, encoding="utf-8") as f:
            self.examples = json.load(f)
        self.schema = {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": LANGS},
                "english_query": {"type": "string"},
                "calls": {"type": "array", "maxItems": config.MAX_CALLS_PER_TURN,
                          "items": toolbox.call_schema()},
            },
            "required": ["language", "english_query", "calls"],
        }
        self.system = self._system_prompt()

    # ------------------------------------------------------------ prompt
    def _system_prompt(self) -> str:
        ex = "\n".join(
            f'User: {e["user"]}\nJSON: {json.dumps(e["output"], ensure_ascii=False)}' for e in self.examples
        )
        return (
            "You are the routing module of the NCAIR LMS onboarding assistant for interns and NYSC members. "
            "Read the user's LAST message (earlier messages are only for resolving words like 'that' or 'and then?') "
            "and reply with one JSON object with exactly these fields:\n"
            "- language: the language of the user's last message: en English, ha Hausa, yo Yoruba, ig Igbo, "
            "pcm Nigerian Pidgin, other.\n"
            "- english_query: the request as a clear, standalone English sentence (copy it if already English).\n"
            "- calls: 0 to 2 tool calls.\n\n"
            "Tools:\n" + self.tb.tool_help() + "\n\n"
            "Rules:\n"
            "- Greetings, thanks and chit-chat need no tool: calls = [].\n"
            "- Link requests (where / open / go to / link) use get_portal_link. 'What do I do' or 'I am stuck at "
            "step N' about the registration process uses get_onboarding_step. Facts, rules, policies, problems, "
            "locations use search_knowledge_base.\n"
            "- If a question might not be covered by the documents, still call search_knowledge_base.\n"
            "- Use two calls only when the user clearly asks for two different things.\n"
            "- Only use page ids and step values listed above. Never invent them.\n\n"
            "Examples:\n" + ex
        )

    def _user_prompt(self, query: str, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return query
        recent = history[-2 * config.HISTORY_TURNS:]
        convo = "\n".join(f'{m["role"]}: {m["content"][:300]}' for m in recent)
        return f"Conversation so far:\n{convo}\n\nLast message: {query}"

    # ------------------------------------------------------------- parse
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _validate_call(self, call: Any, english_query: str) -> Optional[Dict[str, Any]]:
        """Return a clean call dict, or None if it cannot be made valid."""
        if not isinstance(call, dict):
            return None
        name = call.get("tool") or call.get("name")
        spec = self.tb.specs.get(name)
        if spec is None:
            return None
        args: Dict[str, Any] = {}
        for pname, pschema in spec.properties.items():
            val = call.get(pname)
            if val is None and isinstance(call.get("arguments"), dict):
                val = call["arguments"].get(pname)
            if val is None and isinstance(call.get("parameters"), dict):
                val = call["parameters"].get(pname)
            if pname == "query" and not val:
                val = english_query  # a search with no query still has something to search for
            if val is None:
                continue
            val = str(val).strip()
            enum = pschema.get("enum")
            if enum is not None and val not in enum:
                if val.lower() in enum:
                    val = val.lower()
                else:
                    return None
            args[pname] = val
        if any(r not in args for r in spec.required):
            return None
        return {"tool": name, **args}

    # ------------------------------------------------------------- route
    def route(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> RouterDecision:
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": self._user_prompt(query, history)}]
        raw = ""
        for attempt in range(2):
            try:
                raw = self.llm.chat(messages, schema=self.schema, options=config.ROUTER_OPTIONS)
            except LLMError as e:
                return self._fallback(query, f"llm error: {e}", raw)
            data = self._extract_json(raw)
            if isinstance(data, dict) and isinstance(data.get("calls", []), list):
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "That was not valid JSON for the required format. Reply with the JSON object only."})
        else:
            return self._fallback(query, "unparseable router output", raw)

        lang = data.get("language", "en")
        lang = lang if lang in LANGS else "other"
        eq = str(data.get("english_query") or "").strip() or query
        raw_calls = data.get("calls", [])
        calls, seen = [], set()
        for c in raw_calls[: config.MAX_CALLS_PER_TURN]:
            v = self._validate_call(c, eq)
            key = json.dumps(v, sort_keys=True) if v else None
            if v and key not in seen:
                seen.add(key)
                calls.append(v)
        if raw_calls and not calls:
            # the model asked for something, but nothing valid survived: ground it in the KB
            return self._fallback(query, "no valid tool call", raw, language=lang, english_query=eq)
        return RouterDecision(language=lang, english_query=eq, calls=calls, raw=raw)

    def _fallback(self, query: str, error: str, raw: str, language: str = "en",
                  english_query: str = "") -> RouterDecision:
        eq = english_query or query
        return RouterDecision(language=language, english_query=eq,
                              calls=[{"tool": "search_knowledge_base", "query": eq}],
                              fallback=True, error=error, raw=raw)
