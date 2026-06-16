# ScholarLoop — autonomous run log

A complete run on the real **digits-mlp** torch engine, driven by **claude-opus-4-8**. Every experiment below is a real PyTorch training run; every decision is a real LLM call.

- **5** experiments · **24** LLM calls · 24501+5774 tokens · ≈ **$0.267**
- baseline to beat: 5.0% val_top1_err

## 🎯 Director — direction
> Establish a strong MLP baseline on digit classification using modern training recipes (proper normalization, regularization, and learning-rate scheduling) before exploring architectural variants.  
> *topic for the Lit Scout:* MLP architectures and training tricks for image classification MNIST regularization normalization

## 🔭 Lit Scout — grounded findings (real arXiv)
- **Register Attention Transformer (RATS): decompose the classification token into N learnable register tokens routing patch information through an L->N->N->L compress-communicate-broadcast attention bottleneck** (2606.14701v1) — Encourages discovery of reusable compositional part representations, potentially improving classification robustness and interpretability

## 🪜 Experiments (real torch · multi-fidelity funnel)

| id | tier | val_top1_err | verdict | predicted→measured | grounded source |
|---|---|---|---|---|---|
| exp_0001 | smoke | 7.5556% | discarded | — | Standard deep MLP scaling practice; cf. modern training reci |
| exp_0002 | smoke | 24.4444% | discarded | -2.0→16.8888 | Skill library lesson [training-stability, w=0.60]: lr=0.05 t |
| exp_0003 | smoke | 4.6667% | kept | -1.3→-19.7777 | Goodfellow, Bengio & Courville, Deep Learning (2016), Ch. 7- |
| exp_0004 | verify | 4.8148% | kept | — | Goodfellow, Bengio & Courville, Deep Learning (2016), Ch. 7- |
| exp_0005 | full | 4.6667% | kept | — | Goodfellow, Bengio & Courville, Deep Learning (2016), Ch. 7- |

## 🧠 Accumulated skills (self-improvement)

- [training-duration, w=0.70] When a lowered learning rate underperforms, suspect undertraining before reducing capacity: increase epochs to ensure convergence at that lr (e.g., 100 epochs at lr=0.05) and pair with mild weight decay (~5e-4). Here this dropped val_top1_err from 24.4 to 4.7, confirming prior failures were convergence-limited, not capacity-limited.
- [training-stability, w=0.70] When lowering lr to stabilize a deeper/wider MLP, verify epochs are sufficient for convergence at the new lr; lr=0.01 with only 60 epochs likely undertrained the larger model (err jumped from 7.6 to 24.4). Scale epochs up or add a warmup/decay schedule when reducing lr, and confirm the loss has plateaued before judging capacity benefits.
- [training-stability, w=0.60] When scaling MLP width/depth on small datasets like digits, lr=0.05 may be too high and cause divergence (val err 7.55 > random-ish). Add LR warmup + scheduling and sweep lr (e.g., 0.001-0.01) before increasing capacity; validate that the baseline trains stably at the chosen lr before claiming capacity-driven gains.

## 🔁 Agent trace (every call, auditable)

`advisor:3 · critic:Contrarian:3 · critic:Innovator:3 · critic:Pragmatist:3 · director:2 · lit_scout:2 · reasoner:3 · reflector:3`

See [`paper.md`](paper.md) for the write-up this run produced, and [`experiments.jsonl`](experiments.jsonl) for the raw ledger.
