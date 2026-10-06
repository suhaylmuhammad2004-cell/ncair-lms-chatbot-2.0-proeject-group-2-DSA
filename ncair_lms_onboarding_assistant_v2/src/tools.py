"""Tool registry.

Each tool is a `ToolSpec` (name, description, JSON-schema parameters, runner).
The router schema, the router prompt and the validation are all generated from
this registry, and the allowed values (page ids, step numbers) are read from
data files at start-up. To add a tool or a page nothing else needs editing.

A tool does NOT write the final answer. It returns `ToolResult` context that the
answer LLM turns into a reply in the user's language, grounded in that context.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import config
from knowledge import Page, load_pages, step_sections
from retrieval import Retriever


@dataclass
class ToolResult:
    tool: str
    args: Dict[str, Any]
    context: str                                   # text handed to the answer LLM
    links: List[Page] = field(default_factory=list)  # pages the answer should include
    sources: List[str] = field(default_factory=list)  # section ids used
    ok: bool = True
    abstain: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: Dict[str, Dict[str, Any]]          # JSON-schema properties
    required: List[str]
    run: Callable[[Dict[str, Any]], ToolResult]

    def schema(self) -> Dict[str, Any]:
        """JSON schema for one call of this tool (used inside the router schema)."""
        props = {"tool": {"type": "string", "enum": [self.name]}}
        props.update(self.properties)
        return {"type": "object", "properties": props, "required": ["tool"] + self.required,
                "additionalProperties": False}


class ToolBox:
    def __init__(self, retriever: Optional[Retriever] = None):
        self.pages: Dict[str, Page] = load_pages()
        self.steps = step_sections()
        self.retriever = retriever or Retriever()
        self.specs: Dict[str, ToolSpec] = {}
        self._register()

    # --------------------------------------------------------- registry
    def _register(self) -> None:
        page_catalog = "; ".join(f"{p.id} = {p.title}" for p in self.pages.values())
        step_catalog = "; ".join(f"{n} = {s['title'].split(':', 1)[-1].strip()}" for n, s in self.steps.items())

        self._add(ToolSpec(
            name="get_portal_link",
            description=(
                "Give the user the link to ONE page of the NCAIR LMS or NCAIR website. "
                "Use when the user wants to go to, open, or find a page. Pages: " + page_catalog + "."
            ),
            properties={"page_id": {"type": "string", "enum": list(self.pages)}},
            required=["page_id"],
            run=self._get_portal_link,
        ))
        self._add(ToolSpec(
            name="get_onboarding_step",
            description=(
                "Explain what to do at a numbered step of the registration/onboarding process, "
                "or all steps (step='all'). Use when the user is doing, stuck on, or asks what comes "
                "next in the process. Steps: " + step_catalog + "."
            ),
            properties={"step": {"type": "string", "enum": ["all"] + list(self.steps)}},
            required=["step"],
            run=self._get_onboarding_step,
        ))
        self._add(ToolSpec(
            name="search_knowledge_base",
            description=(
                "Search the official NCAIR documents (onboarding manual and programme guidelines) for "
                "policies, rules, attendance, courses, SIWES/NYSC differences, troubleshooting, "
                "location, returning interns. Use for any factual question not answered by the two tools "
                "above, including questions that might be outside the documents. "
                "'query' must be a short English search query."
            ),
            properties={"query": {"type": "string"}},
            required=["query"],
            run=self._search,
        ))

    def _add(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    # ------------------------------------------------------------ schema
    def call_schema(self) -> Dict[str, Any]:
        return {"anyOf": [s.schema() for s in self.specs.values()]}

    def tool_help(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.specs.values())

    # --------------------------------------------------------- execution
    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult(name, args, f"Error: unknown tool '{name}'", ok=False)
        missing = [r for r in spec.required if r not in args or args[r] in (None, "")]
        if missing:
            return ToolResult(name, args, f"Error: missing argument(s) {missing}", ok=False)
        try:
            return spec.run(args)
        except Exception as e:  # a tool must never crash the chat
            return ToolResult(name, args, f"Error executing {name}: {e}", ok=False)

    # ------------------------------------------------------------- tools
    def _get_portal_link(self, args) -> ToolResult:
        page = self.pages.get(str(args["page_id"]).strip())
        if page is None:
            return ToolResult("get_portal_link", args, f"Error: unknown page '{args['page_id']}'", ok=False)
        note = "" if page.verified else " (Note: please confirm this link works; it has not been checked against the live site.)"
        ctx = f"PAGE: {page.title}\nURL: {page.url}\nABOUT: {page.description}{note}"
        return ToolResult("get_portal_link", {"page_id": page.id}, ctx, links=[page])

    def _get_onboarding_step(self, args) -> ToolResult:
        which = str(args["step"]).strip().lower()
        if which in ("all", "0", "every", "full"):
            chosen = list(self.steps.items())
        elif which in self.steps:
            chosen = [(which, self.steps[which])]
        else:
            return ToolResult("get_onboarding_step", args,
                              f"Error: no such step '{which}'. Valid steps: {', '.join(self.steps)}", ok=False)
        parts, links, sources = [], [], []
        for n, s in chosen:
            parts.append(f"STEP {n} - {s['title'].split(':', 1)[-1].strip()}\n{s['body']}")
            sources.append(f"manual/{s['id']}")
            pid = s["meta"].get("link")
            if pid in self.pages and self.pages[pid] not in links:
                links.append(self.pages[pid])
        links = links[:2] if which != "all" else []
        return ToolResult("get_onboarding_step", {"step": which}, "\n\n".join(parts) + self._related(links),
                          links=links, sources=sources)

    @staticmethod
    def _related(links: List[Page]) -> str:
        """Expose related page links to the LLM inside the context (so it can cite them)."""
        return "".join(f"\nRELATED PAGE: {p.title} - {p.url}" for p in links)

    def _search(self, args) -> ToolResult:
        res = self.retriever.search(str(args["query"]))
        links: List[Page] = []
        if not res.abstain:
            for h in res.hits[:2]:  # a page attached to a top passage is relevant to the answer
                pid = h.chunk.meta.get("link")
                if pid in self.pages and self.pages[pid] not in links:
                    links.append(self.pages[pid])
        return ToolResult("search_knowledge_base", {"query": args["query"]},
                          Retriever.format_context(res) + self._related(links), links=links,
                          sources=res.section_ids, abstain=res.abstain)
