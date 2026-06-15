"""Lit Scout: deterministic arXiv parsing (fake fetcher) + LLM extraction (MockLLM)."""

from scholarloop.litscout import ArxivClient, LitScout
from scholarloop.llm import MockLLM

ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1608.03983v1</id>
    <title>SGDR: Stochastic Gradient Descent with Warm Restarts</title>
    <summary>We propose a cosine annealing schedule with warm restarts.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1512.03385v1</id>
    <title>Deep Residual Learning for Image Recognition</title>
    <summary>Residual connections ease optimization of deep networks.</summary>
  </entry>
</feed>"""

EMPTY_XML = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def test_arxiv_parsing_from_canned_feed():
    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    papers = arxiv.search("image classification")
    assert [p.arxiv_id for p in papers] == ["1608.03983v1", "1512.03385v1"]
    assert papers[0].title.startswith("SGDR")
    assert "cosine" in papers[0].summary


def test_scout_extracts_findings_and_priors():
    arxiv = ArxivClient(fetcher=lambda q, n: ATOM_XML)
    llm = MockLLM(jsons=[{"findings": [
        {"technique": "cosine schedule", "source": "arXiv:1608.03983",
         "predicted_effect": "lower val error", "rationale": "warm restarts escape minima"},
    ]}])
    lit_context, lit_priors = LitScout(llm, arxiv).scout("image classification")
    assert "cosine schedule" in lit_context
    assert lit_priors == ["cosine schedule (arXiv:1608.03983)"]


def test_scout_handles_no_papers():
    arxiv = ArxivClient(fetcher=lambda q, n: EMPTY_XML)
    # LLM should never be called when there are no papers
    assert LitScout(MockLLM(), arxiv).scout("obscure topic") == ("", [])
