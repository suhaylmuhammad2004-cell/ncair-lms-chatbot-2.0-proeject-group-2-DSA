"""Loads every knowledge source into plain data objects.

Nothing here is NCAIR-specific logic: page links come from portal_pages.json,
manual sections from ncair_manual.md (including the onboarding steps), and
policy sections from ncair_knowledge_base.txt. Adding or editing content means
editing those files, not code.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config


# ---------------------------------------------------------------- pages
@dataclass
class Page:
    id: str
    title: str
    url: str
    description: str
    verified: bool = True
    source: str = ""


def load_pages(path: str = config.PAGES_FILE) -> Dict[str, Page]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {p["id"]: Page(**{k: p[k] for k in Page.__dataclass_fields__ if k in p}) for p in raw["pages"]}


def load_support_email(path: str = config.PAGES_FILE) -> str:
    """Support e-mail, but only if it has been marked verified in portal_pages.json."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("support_email", "") if raw.get("support_email_verified") else ""


# --------------------------------------------------------------- chunks
@dataclass
class Chunk:
    id: str                  # unique chunk id, e.g. "manual/step-4#0"
    section_id: str          # section id used for retrieval labels, e.g. "manual/step-4"
    doc: str                 # "manual" | "kb"
    title: str
    text: str                # text shown to the LLM (title included)
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        page = self.meta.get("page")
        return f"{self.doc}: {self.title}" + (f" (manual p.{page})" if page else "")


_HEADING = re.compile(r"^## (?P<id>[\w\-]+) \| (?P<title>.+)$")
_META = re.compile(r"^@(?P<k>\w+):\s*(?P<v>.*)$")


def parse_manual(path: str = config.MANUAL_FILE) -> List[Dict]:
    """Parse the structured manual into sections: dicts with id/title/meta/body."""
    sections: List[Dict] = []
    cur: Optional[Dict] = None
    in_comment = False
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if "<!--" in line:
                in_comment = True
            if in_comment:
                if "-->" in line:
                    in_comment = False
                continue
            m = _HEADING.match(line)
            if m:
                cur = {"id": m["id"], "title": m["title"].strip(), "meta": {}, "body": []}
                sections.append(cur)
                continue
            if cur is None:
                continue
            mm = _META.match(line)
            if mm and not cur["body"]:
                cur["meta"][mm["k"]] = mm["v"].strip()
            elif line.strip():
                cur["body"].append(line.strip())
    for s in sections:
        s["body"] = " ".join(s["body"])
    return sections


def parse_kb(path: str = config.KB_FILE) -> List[Dict]:
    """Split the plain-text knowledge base on its numbered headings ("1. TITLE")."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections: List[Dict] = []
    cur: Optional[Dict] = None
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s+([A-Z][A-Z0-9 &(),/\-\.]+)$", line.strip())
        if m:
            cur = {"id": f"kb-{m.group(1)}", "title": m.group(2).title(), "meta": {}, "body": []}
            sections.append(cur)
        elif cur is not None and line.strip():
            cur["body"].append(line.strip())
    return sections


def _split_text(text: str, limit: int) -> List[str]:
    """Split a long section on sentence/bullet boundaries, never mid-sentence."""
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=- )", text)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > limit:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        out.append(buf)
    return out


def build_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc, sections in (("manual", parse_manual()), ("kb", parse_kb())):
        for s in sections:
            sid = f"{doc}/{s['id']}"
            if doc == "kb":
                # KB sections are lists of bullet rules; chunk bullet-by-bullet-ish
                body = " ".join(s["body"])
            else:
                body = s["body"]
            for i, piece in enumerate(_split_text(body, config.MAX_CHUNK_CHARS)):
                chunks.append(
                    Chunk(
                        id=f"{sid}#{i}",
                        section_id=sid,
                        doc=doc,
                        title=s["title"],
                        text=f"{s['title']}. {piece}",
                        meta=dict(s["meta"]),
                    )
                )
    return chunks


def step_sections() -> Dict[str, Dict]:
    """Onboarding steps, keyed by step number as a string, straight from the manual."""
    steps = {}
    for s in parse_manual():
        if s["meta"].get("type") == "step":
            steps[s["meta"]["step"]] = s
    return dict(sorted(steps.items(), key=lambda kv: int(kv[0])))
