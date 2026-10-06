"""Hybrid retrieval over the structured knowledge chunks.

* Lexical: BM25 implemented here in pure Python (no heavy dependency, so the
  retrieval test suite runs anywhere, including offline CI).
* Dense (optional): a multilingual sentence-transformers model. Used
  automatically when installed; embeddings are cached on disk keyed by a hash
  of the corpus, so nothing is recomputed at start-up.
* Fusion: reciprocal-rank fusion when both are available.
* Abstention signal: `coverage` = idf-weighted share of the query's terms that
  the best passages actually contain. Terms the corpus has never seen count
  as maximally informative, so "WiFi password" or "who founded" score low.

This replaces the previous design that ran Tesseract OCR over the PDF every
time the module was imported.
"""
import hashlib
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import config
from knowledge import Chunk, build_chunks

_STOP = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being do does did
    i me my we our you your he she it its they them their this that these those there here what which who whom
    how when where why can could should would will shall may might must have has had not no yes so than then
    about into over any some all each very just also please tell get got""".split()
)


def _fold(s: str) -> str:
    """Lowercase and strip diacritics so Yoruba/Igbo marks never block a match."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ingly", "ings", "ing", "edly", "ied", "ies", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf in ("ied", "ies") else "")
    return w


def tokenize(text: str, keep_stop: bool = False) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", _fold(text))
    return [_stem(t) for t in toks if keep_stop or t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class SearchResult:
    query: str
    hits: List[Hit]
    coverage: float
    abstain: bool
    method: str
    low_confidence: bool = False

    @property
    def section_ids(self) -> List[str]:
        seen, out = set(), []
        for h in self.hits:
            if h.chunk.section_id not in seen:
                seen.add(h.chunk.section_id)
                out.append(h.chunk.section_id)
        return out


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.idf_unseen = math.log(1 + self.N / 0.5)
        self.tf = []
        for d in self.docs:
            c: Dict[str, int] = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            self.tf.append(c)

    def idf(self, t: str) -> float:
        n = self.df.get(t)
        if n is None:
            return self.idf_unseen
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, q: Sequence[str]) -> List[float]:
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in set(q):
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class Retriever:
    def __init__(self, chunks: Optional[List[Chunk]] = None, dense: str = config.USE_DENSE):
        self.chunks = chunks if chunks is not None else build_chunks()
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25(self._tokens)
        self._emb = None
        self._encoder = None
        if dense != "no":
            self._init_dense(required=(dense == "yes"))

    # ------------------------------------------------------------ dense
    def _corpus_hash(self) -> str:
        h = hashlib.sha256(config.EMBEDDING_MODEL.encode())
        for c in self.chunks:
            h.update(c.id.encode())
            h.update(c.text.encode())
        return h.hexdigest()[:16]

    def _init_dense(self, required: bool) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # not installed / offline
            if required:
                raise
            print(f"[retrieval] dense retrieval unavailable ({type(e).__name__}); using BM25 only.")
            return
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        cache = os.path.join(config.INDEX_DIR, f"emb_{self._corpus_hash()}.npy")
        try:
            self._encoder = SentenceTransformer(config.EMBEDDING_MODEL)
            if os.path.exists(cache):
                self._emb = np.load(cache)
            else:
                self._emb = self._encoder.encode(
                    [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
                )
                np.save(cache, self._emb)
        except Exception as e:
            if required:
                raise
            print(f"[retrieval] could not load embedding model ({type(e).__name__}); using BM25 only.")
            self._encoder = self._emb = None

    @property
    def method(self) -> str:
        return "hybrid(bm25+dense)" if self._emb is not None else "bm25"

    # ----------------------------------------------------------- search
    @staticmethod
    def _rank(scores: List[float]) -> List[int]:
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def coverage(self, query: str, idxs: Sequence[int]) -> float:
        q = set(tokenize(query))
        if not q or not idxs:
            return 0.0
        total = sum(self.bm25.idf(t) for t in q)
        best = 0.0
        for i in idxs:
            have = set(self._tokens[i])
            got = sum(self.bm25.idf(t) for t in q if t in have)
            best = max(best, got / total)
        return best

    def search(self, query: str, k: int = config.TOP_K, min_coverage: float = config.MIN_COVERAGE) -> SearchResult:
        q = tokenize(query)
        lex = self.bm25.scores(q)
        lex_rank = self._rank(lex)
        if self._emb is not None:
            qv = self._encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            dense_scores = (self._emb @ qv).tolist()
            dense_rank = self._rank(dense_scores)
            fused: Dict[int, float] = {}
            for ranking in (lex_rank, dense_rank):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)
            order = sorted(fused, key=lambda i: -fused[i])[:k]
            scores = fused
        else:
            order = [i for i in lex_rank[:k] if lex[i] > 0]
            scores = {i: lex[i] for i in order}
        # Section expansion: a long section was split into several chunks, and the sentence that answers
        # the question may be in the sibling of the chunk that matched. Pull in the siblings of the
        # best hit (bounded) so the answer model sees the whole rule.
        if order:
            sid = self.chunks[order[0]].section_id
            sibs = [i for i, c in enumerate(self.chunks) if c.section_id == sid and i not in order]
            if sum(len(self.chunks[i].text) for i in sibs) <= 1200:
                order = order[:1] + sibs + order[1:]
                for i in sibs:
                    scores.setdefault(i, 0.0)
        order = order[: k + 2]
        hits = [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]
        cov = self.coverage(query, order)
        return SearchResult(
            query, hits, cov,
            abstain=(not hits or cov < min_coverage),
            method=self.method,
            low_confidence=cov < config.SOFT_COVERAGE,
        )

    # ----------------------------------------------------------- format
    @staticmethod
    def format_context(result: SearchResult) -> str:
        if result.abstain:
            return (
                "NO RELEVANT INFORMATION FOUND in the official documents for this question. "
                "Do not answer from general knowledge."
            )
        body = "\n\n".join(f"[{h.chunk.label}]\n{h.chunk.text}" for h in result.hits)
        if result.low_confidence:
            body = (
                "RETRIEVAL CONFIDENCE: LOW. These passages may not answer the question. "
                "If they do not clearly contain the answer, say it is not covered.\n\n" + body
            )
        return body
