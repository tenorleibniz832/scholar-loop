# Sample run (second domain) — regression on diabetes

A second **verbatim capture**, on a different domain and a different metric: the real
`diabetes-mlp` torch engine (RMSE regression on sklearn diabetes), with **Claude Opus 4.8** as
every agent. Same orchestrator, funnel, and guards as the [digits run](../sample_run/) — only the
profile + engine pair are new, which is the whole point of pluggable domains. Produced by
[`../run_to_paper.py`](../run_to_paper.py) with `SCHOLARLOOP_PROFILE=diabetes-mlp`.

> **8 experiments · 26 LLM calls · ≈ $0.40.** The must-beat bar here is the **linear-model
> reference (56.5 RMSE)** — OLS/Ridge both land there on this split, and an untuned MLP only ties
> it. The loop found that a *shallow, well-regularized, long-trained* MLP (depth 1, width 64,
> weight_decay 3e-3, 400 epochs) actually edges past linear regression to a confirmed **55.26 RMSE**
> — a real but small win, climbed all the way smoke → verify → full. Adding depth did **not** help
> (56.46, no gain), and predict-then-verify caught the Reasoner's over-optimistic −2.5 prediction
> for it (measured +1.2). The reviewer rejected the paper 3/10 as marginal.

**Two anti-self-deception guards firing live in this run:**

- **predict-then-verify** — for the depth-2 idea the Reasoner predicted a −2.5 RMSE gain; the
  measurement was +1.2 (it got *worse*). The gap is recorded as calibration error, not hidden.
- **number-grounding flagged `1.2`** — the Writer stated the improvement as "about 1.2 RMSE", but
  1.2 is a *derived* difference (56.5 − 55.26), never a directly measured value, so the grounding
  audit marked the draft ⚠️. That is the anti-hallucination gate working as designed: only numbers
  that trace to a recorded measurement pass unflagged.

| file | what it is |
|---|---|
| **[`run.md`](run.md)** | the autonomous run log — direction, the real (arXiv + OpenAlex, citation-ranked) findings, every experiment (funnel tiers + verdicts + predict-vs-measured), and the distilled lessons |
| **[`paper.md`](paper.md)** | the paper the L5 Writer produced + the Reviewer's assessment, with the grounding audit's ⚠️ on the derived `1.2` |
| **[`experiments.jsonl`](experiments.jsonl)** | the raw ledger — one real torch training run per record |

Reproduce (non-deterministic — a fresh run yields different experiments and a different paper):

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  SCHOLARLOOP_PROFILE=diabetes-mlp \
  SCHOLARLOOP_OUT=examples/sample_run_diabetes \
  SCHOLARLOOP_STEPS=5 \
  python examples/run_to_paper.py
```
