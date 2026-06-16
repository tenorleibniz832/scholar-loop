# Sample run — a complete flow with a real LLM

A **verbatim capture** of one autonomous ScholarLoop run: the real `digits-mlp` torch engine,
with **Claude Opus 4.8** as every agent, taken all the way from *"pick a direction"* to a
written-up, peer-reviewed paper. Produced by [`../run_to_paper.py`](../run_to_paper.py).

> **5 experiments · 24 LLM calls · ≈ $0.27.** The loop read the literature, proposed
> literature-grounded configs, debated them, ran real PyTorch experiments through the
> multi-fidelity funnel, learned from a diverging run, found a config that beats the baseline, and
> wrote it up — then its own reviewer rejected the paper as too marginal. (It's not wrong.)

| file | what it is |
|---|---|
| **[`run.md`](run.md)** | the autonomous run log — director's direction, the real arXiv findings, every experiment (funnel tiers + verdicts + predict-vs-measured), and the lessons the Reflector distilled |
| **[`paper.md`](paper.md)** | the paper the L5 Writer produced + the Reviewer's assessment. Every number is checked against the experiment registry (grounding ✅) |
| **[`experiments.jsonl`](experiments.jsonl)** | the raw ledger — one real torch training run per record |

Reproduce (non-deterministic — a fresh run yields different experiments and a different paper):

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/run_to_paper.py
```
