"""Lit Scout (DESIGN §4 L3) — literature grounding for the Reasoner.

Two cleanly separated halves, per the harness rule "retrieval is a deterministic tool,
extraction is an agent":
  - `ArxivClient` — a plain HTTP client for the arXiv API. No LLM. The network fetcher is
    injectable, so tests run against canned Atom XML with no network.
  - `LitScout(Agent)` — turns the retrieved papers into structured, testable `findings`
    (technique + source + predicted_effect) via the agent harness.

`scout(topic)` returns `(lit_context, lit_priors)`: a prose block the Reasoner reads, and a
list of "technique (source)" priors fed into the search-space analysis. This is what makes
ideas literature-grounded instead of blind local hill-climbing.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from scholarloop.agent import Agent

_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class Paper:
    title: str
    summary: str
    arxiv_id: str
    url: str


class ArxivClient:
    """Deterministic arXiv search. `fetcher(query, max_results) -> atom_xml` is injectable."""

    def __init__(self, fetcher=None, *, timeout: float = 20.0):
        self._fetch = fetcher or self._http_fetch
        self.timeout = timeout

    def _http_fetch(self, query: str, max_results: int) -> str:  # pragma: no cover - network
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": query, "max_results": max_results,
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return r.read().decode("utf-8")

    def search(self, topic: str, max_results: int = 5) -> list[Paper]:
        xml = self._fetch(f"all:{topic}", max_results)
        root = ET.fromstring(xml)
        papers: list[Paper] = []
        for e in root.findall(_ATOM + "entry"):
            id_text = (e.findtext(_ATOM + "id") or "").strip()
            papers.append(Paper(
                title=" ".join((e.findtext(_ATOM + "title") or "").split()),
                summary=" ".join((e.findtext(_ATOM + "summary") or "").split()),
                arxiv_id=id_text.rsplit("/", 1)[-1],
                url=id_text,
            ))
        return papers


FINDINGS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "technique": {"type": "string"},
                    "source": {"type": "string"},
                    "predicted_effect": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["technique", "source", "predicted_effect"],
            },
        },
    },
    "required": ["findings"],
}

_SYSTEM = (
    "You read ML papers and extract concrete, testable techniques (architecture changes, "
    "optimizers, schedules, augmentations) that could improve a model on the given topic. "
    "Cite the arXiv id as the source for each. Do not invent techniques the papers don't support."
)


class LitScout(Agent):
    name = "lit_scout"
    system = _SYSTEM
    schema = FINDINGS_SCHEMA

    def __init__(self, llm, arxiv: ArxivClient | None = None, *, trace=None, max_results: int = 5):
        super().__init__(llm, trace=trace)
        self.arxiv = arxiv or ArxivClient()
        self.max_results = max_results

    def build_prompt(self, ctx: dict) -> str:
        papers: list[Paper] = ctx["papers"]
        body = "\n\n".join(f"[{p.arxiv_id}] {p.title}\n{p.summary[:600]}" for p in papers)
        return (f"Topic: {ctx['topic']}\n\nFrom these papers, extract techniques that could "
                f"improve a model on this topic. For each: technique, source (arXiv id), "
                f"predicted_effect, and a one-line rationale.\n\nPapers:\n{body}")

    def scout(self, topic: str) -> tuple[str, list[str]]:
        """Retrieve + extract. Returns (lit_context prose, lit_priors list)."""
        papers = self.arxiv.search(topic, self.max_results)
        if not papers:
            return "", []
        findings = self.run({"topic": topic, "papers": papers}).get("findings", [])
        lit_context = "\n".join(
            f"- {f['technique']} ({f['source']}): {f['predicted_effect']}"
            + (f" — {f.get('rationale')}" if f.get("rationale") else "")
            for f in findings)
        lit_priors = [f"{f['technique']} ({f['source']})" for f in findings]
        return lit_context, lit_priors
