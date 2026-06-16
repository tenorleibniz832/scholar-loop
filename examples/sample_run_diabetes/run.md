# ScholarLoop — autonomous run log

A complete run on the real **diabetes-mlp** torch engine, driven by **claude-opus-4-8**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **8** experiments · **26** LLM calls · 35252+9040 tokens · ≈ **$0.402**
- baseline to beat: 56.5 val_rmse

## 🎯 Director — direction
> Establish a strong regularized MLP baseline with proper input normalization, dropout, and weight decay tuning before exploring architectural complexity. On the small Diabetes regression dataset, careful regularization and feature preprocessing typically yield the largest gains and the lowest RMSE.  
> *topic for the Lit Scout:* regularization techniques for multilayer perceptron regression on small tabular datasets dropout weight decay

## 🔭 Lit Scout — grounded findings (real literature: arXiv + OpenAlex, citation-ranked)
- **Denoising autoencoder pretraining / input corruption for tabular features** (doi:10.1613/jair.1.12312) — Reduces overfitting on small datasets by learning robust feature representations, acting as a regularizer for downstream MLP regression
- **Semi-supervised learning leveraging unlabelled tabular data with consistency regularization** (doi:10.1007/s10994-019-05855-6) — Improves generalization of MLP regression when labelled data is scarce by exploiting unlabelled examples
- **Boosted factor (BF) regularization on output/feature weights** (doi:10.1109/tsmc.2022.3213628) — Lowers regression error on small tabular datasets by controlling weight magnitudes more effectively than plain weight decay
- **Skip connections (edRVFL-SC) feeding raw features to later layers** (doi:10.1109/tsmc.2022.3213628) — Eases optimization and improves regression accuracy by preserving original feature information
- **Random skip connections (edRVFL-RSC) as stochastic regularization** (doi:10.1109/tsmc.2022.3213628) — Acts like dropout on connections, reducing overfitting on small datasets
- **Ensemble of deep layers / ensemble skip connection (esc-edRVFL)** (doi:10.1109/tsmc.2022.3213628) — Reduces variance and improves robustness of regression predictions on small data
- **Variational autoencoder-based data imputation/augmentation before MLP training** (doi:10.1613/jair.1.12312) — Mitigates missing-value harm and effectively augments small datasets, improving regression generalization

## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_rmse | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|
| exp_0001 | smoke | 55.3809 | kept | — | Boosted factor (BF) regularization on output/feature weights |
| exp_0002 | verify | 55.3248 | kept | — | Boosted factor (BF) regularization on output/feature weights |
| exp_0003 | full | 55.3809 | kept | — | Boosted factor (BF) regularization on output/feature weights |
| exp_0004 | smoke | 56.5587 | discarded | -2.5→1.2339 | Boosted factor / skip-connection regression study (doi:10.11 |
| exp_0005 | verify | 56.4632 | kept | — | Boosted factor / skip-connection regression study (doi:10.11 |
| exp_0006 | smoke | 55.3161 | kept | -0.6→-0.0087 | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0007 | verify | 55.2618 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |
| exp_0008 | full | 55.3161 | kept | — | L1/L2 weight regularization (weight decay) (doi:10.1007/s429 |

## 🧠 Accumulated skills (self-improvement)

- [capacity-vs-regularization tradeoff, w=0.55] On small tabular datasets (e.g. diabetes), increasing both depth and width while lowering lr and extending epochs did not beat a simpler parent (val_rmse worsened 55.32->56.46). Before adding capacity, verify the baseline is actually underfitting (train vs val gap); if train error is already low, prefer tuning regularization/lr on the existing architecture rather than enlarging the network.
- [hyperparameter-tuning, w=0.55] For small tabular MLPs (e.g. diabetes), pairing a lower LR (~0.01) with moderate weight decay (~1e-3) and extended training (~400 epochs) on a shallow single-layer 64-unit net yields a stable, better-regularized model that beats the baseline; use this as a sensible default starting point and tune incrementally from it.
- [hyperparameter-tuning, w=0.40] On small low-dimensional datasets like diabetes, increasing L2 weight decay (e.g. to 3e-3) on an already-shallow MLP yields only marginal val_rmse gains (~0.01 RMSE). Treat weight-decay sweeps as fine-tuning with low expected payoff; prioritize larger architectural or data changes for meaningful improvements rather than expecting decay alone to break overfitting plateaus.

## 🔁 Agent trace (every call, auditable)

`advisor:3 · critic:Contrarian:3 · critic:Innovator:3 · critic:Pragmatist:3 · director:2 · lit_scout:2 · reasoner:5 · reflector:3`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
