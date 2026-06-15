<div align="center">

# 🔬 ScholarLoop

### An autonomous, multi-agent research framework that replicates a PhD's workflow — on a single-GPU budget.

**read literature → find a gap → propose an idea → run real experiments → measure → remember → write & self-review**

[![tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)](#-quickstart)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#-quickstart)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-research%20preview-orange)](#-status--roadmap)

</div>

---

ScholarLoop closes the loop that a PhD student actually runs: it reads the literature, forms a
literature-grounded hypothesis, **runs real ML experiments**, judges them against a frozen
ground-truth metric, accumulates intuition from its failures, and drafts a peer-reviewed write-up
— autonomously, with a deterministic harness keeping the agents honest.

It's inspired by Karpathy's minimalist
[`autoresearch`](https://github.com/karpathy/autoresearch) (a 5-minute train loop optimizing one
metric), and extends that kernel with the parts a PhD actually needs: **literature grounding, a
persistent memory, multi-agent judgment, a multi-fidelity compute funnel, and hard
anti-reward-hacking guards.**

```
  L4 · Director         ledger trends + literature hotspots → direction · topic · budget
        │
  L3 · Lit Scout        arXiv → structured, cited findings
       Reasoner         constraints + literature + skills → a structured next experiment
        │
  L2 · Debate Panel     Innovator · Pragmatist · Contrarian — "worth a GPU?"
       Funnel           smoke → verify → full   (a cheap screen kills most ideas)
       Runner           real torch experiment · frozen-metric scored · recorded
       Reflector        outcome (esp. mispredictions) → decaying skill library
       Advisor          PROCEED · REFINE · PIVOT  loop control
        │
  L5 · Writer + Reviewer  findings → number-grounded draft → peer review
  ──────────────────────────────────────────────────────────────────────────
  L0 · Ledger + Skill Library     the durable memory every layer reads from
```

Every agent runs on **one shared harness** (typed JSON-schema I/O, validate→retry, full call
tracing). The open-ended reasoning is the LLM's job; everything checkable — search-space pruning,
dedup, calibration, number-grounding, promotion gates — is **deterministic, unit-tested code,
never delegated to an agent.**

---

## ✨ Highlights

> **The gist:** a complete PhD research loop — not just an experiment runner — built so that the
> LLM does only the open-ended reasoning while every checkable step is deterministic, tested, and
> impossible to game.

| | |
|---|---|
| 🧪 **Runs real experiments** | Drives a real PyTorch training run (`engines/torch_vision`, an MLP on sklearn digits — CPU-fast, no download). **Pluggable domains:** a new domain = one YAML profile + an engine pair, **zero orchestrator changes.** |
| 🤖 **8 agents, one harness** | Director · Lit Scout · Reasoner · Debate Panel · Reflector · Advisor · Writer · Reviewer — all on a single harness with typed JSON-schema I/O, validate→retry, and **one shared audit trace** of every call. |
| 🔭 **Literature-grounded ideas** | The **Lit Scout** pulls real arXiv papers and extracts structured, *cited* techniques that feed the Reasoner — so it isn't blind local hill-climbing. |
| 🧭 **Reasons about its search space** | Deterministic **region pruning, sensitivity, and dedup** from the ledger — and the constraints actually *bind* (a ruled-out or repeated config is refused, not just discouraged). |
| 💸 **Budget-aware funnel** | One idea climbs **smoke → verify → full**; each tier gated (beat the frontier; beat it *on the worst seed*). Bad ideas die after one cheap run; marginal ones never burn a full run. |
| 🗳️ **Multi-agent judgment** | A debate panel (personas from the profile) votes *whether an idea is worth a GPU* — **never** an LLM-judge on the metric itself (the measurement decides quality). |
| 🎯 **Predict-then-verify** | Every idea predicts its effect *before* running; the loop scores the prediction against the measurement (`calibration_error`) — reasoning is demoted by evidence, not trusted on faith. |
| 🧠 **Compounding intuition** | A **Reflector** turns failures (especially big mispredictions) into lessons in a **time-decaying skill library**, re-injected into the Reasoner next round — no training, backbone-agnostic. |
| 🛡️ **Can't be reward-hacked** | Three layered guards (below) — proven by a bundled adversarial `cheater` engine whose faked metric is overridden by the honest one. |
| ✅ **Honest & fully testable** | **76 tests, no API key or GPU required** — the entire loop runs deterministically against a `MockLLM`. |

### 🛡️ Trustworthy by construction

The whole system is built so the agents *can't* quietly cheat or hallucinate their way to a "win":

- **Two-phase frozen scoring** — `train.py` is editable but only produces a model and is handed
  the *training split only*; a **frozen scorer, run from pristine ROOT**, computes the trusted
  metric. A faked `SCHOLARLOOP_RESULT` line is ignored; the val set is never reachable for training.
- **`forbidden_edits` allowlist** — the source-edit channel may *only* replace the train file
  (normpath-checked, traversal- and shadowing-proof), in an isolated copy that never touches the repo.
- **VerifiedRegistry** — every number in a generated write-up must trace to a real measurement, or
  the draft is flagged.
- **Adversarially hardened** — three independent review passes found and fixed real bugs (path
  traversal, runtime tamper, arbitrary-unpickle, ID reuse, fail-open gates); residual boundaries
  that genuinely need sandboxing are documented honestly, not hidden.

---

## 🚀 Quickstart

```bash
pip install -e .            # pyyaml + jsonschema
pip install -e ".[dev]"     # + pytest        (optional: ".[llm]" for the real Anthropic client)

# 1. The whole loop in under a second (stub engine, no GPU, no API key):
python examples/quickstart.py

# 2. A full campaign on REAL torch — the entire agent chain, MockLLM-scripted:
python examples/campaign_demo.py

# 3. The tests:
pytest -q
```

`quickstart.py` runs one idea through the multi-fidelity funnel and prints the tiers:

```
baseline to beat: 4.9% val_top1_err

  smoke  val_top1_err=3.7644%  [kept]
  verify val_top1_err=3.8004%  [kept]
  full   val_top1_err=3.7644%  [kept]

the idea climbed 3 fidelity tier(s); 3 kept.
```

---

## 🎬 See it run — a real campaign

`examples/campaign_demo.py` runs the **full chain on real torch**, scripted by `MockLLM` so it's
deterministic and free. Abridged output:

```text
=== CAMPAIGN: digits-mlp (real torch) — baseline 5.0% ===
[Director] direction: scale width/depth and tune the optimizer  | topic: mlp digit classification
[LitScout] grounded priors: ['wider/deeper layers (arXiv:1512.03385)', 'SGD momentum + cosine (arXiv:1608.03983)']

--- idea 1 ---
[Reasoner] config={lr:0.2, hidden:128, depth:2, epochs:40}   predicted_delta=-1.5
[Debate]   decision=run
[Funnel]   smoke 4.67% [kept] -> verify 4.96% [kept]        # robustness gate stops it before full
[Advisor]  proceed — new frontier set; keep scaling

--- idea 2 ---
[Debate] REJECTED [Innovator=reject, Pragmatist=reject, Contrarian=run] -> skipped, no GPU spent

--- idea 3 ---
[Reasoner] config={lr:0.5, hidden:8, depth:1, epochs:10}     predicted_delta=-1.0
[Funnel]   smoke 52.0% [discarded]                           # tiny model dies after ONE cheap run
[Advisor]  pivot — tiny models are a dead end  | predict-vs-measured calib_err=48.33

=== SKILL LIBRARY (accumulated intuition) ===
- [capacity, w=0.90] wider+deeper MLP (hidden>=128, depth>=2) clears the baseline
- [capacity, w=0.80] tiny hidden width (<=8) underfits badly — avoid
```

What this one run demonstrates: literature grounding feeding the Reasoner, the debate panel
**saving a GPU run**, the funnel **killing a bad idea cheaply**, predict-then-verify **quantifying a
wildly wrong prediction** (`calib_err=48.3`), and the advisor **pivoting** (which re-consults the
Director) — all on real measurements, every agent call auditable in one trace.

---

## 🧭 Design principles (the harness contract)

1. **Deterministic control flow; model-driven steps.** The orchestrator owns the loop, gating, and
   budget. Agents only do open-ended reasoning. An LLM never decides control flow.
2. **Typed I/O per agent.** Every agent declares a JSON schema; the harness validates and retries.
   Downstream code always receives a well-formed object — never prose to parse.
3. **Checkable work stays in code.** Dedup, calibration, constraint enforcement, number-grounding,
   and promotion gates are deterministic — the agent proposes, the code verifies.
4. **Tools are deterministic; extraction is an agent.** The Lit Scout's arXiv fetch is plain HTTP
   (injectable/mockable); only turning papers into findings is an LLM step.
5. **No agent holds durable state.** State lives in the ledger, so long runs don't drift.
6. **The metric is the only ground truth.** LLM judges rank and review; they are *never* the
   optimization target.

See [`DESIGN.md`](DESIGN.md) for the full architecture, the three reference projects it distills
([AI-Researcher](https://github.com/HKUDS/AI-Researcher),
[AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw), Karpathy's `autoresearch`), and
the honestly-documented residual boundaries (a deliberately filesystem-adversarial `train.py` and
public-dataset val reconstruction still need container sandboxing).

---

## 🗂️ Layout

```
scholarloop/
  agent.py        Agent harness — schema-validated I/O, retry, AgentTrace
  orchestrator.py the loop: propose → debate → funnel → reflect → advise; PROCEED/REFINE/PIVOT
  reasoner.py     constraints + literature + skills → structured proposal (+ source-edit channel)
  reasoning.py    deterministic search-space analysis + predict-then-verify calibration
  litscout.py     arXiv retrieval (deterministic) + finding extraction (agent)
  debate.py       persona critics vote whether an idea is worth a GPU
  reflector.py    run outcome → one lesson
  skills.py       time-decaying skill library (content-hash dedup)
  advisor.py      PROCEED / REFINE / PIVOT
  director.py     L4 — sets campaign direction + topic
  paper.py        L5 — Writer + Reviewer, with number-grounding
  runner.py       two-phase frozen scoring; budget; multi-fidelity; source-edit isolation
  ledger.py · registry.py · profile.py · llm.py
profiles/         one YAML per domain (image-classification stub, digits-mlp torch, …)
engines/          (editable train.py, frozen prepare.py) pairs the profiles point at
examples/         quickstart.py (60s) · campaign_demo.py (full chain on real torch)
tests/            76 tests, all runnable without an API key or GPU
```

---

## 📊 Status & roadmap

**Research preview** — the full PhD-workflow skeleton runs end-to-end on real experiments, with
the anti-reward-hacking guards in place and adversarially reviewed.

- [x] L0–L5 layers, multi-agent harness, multi-fidelity funnel
- [x] Real torch engine + pluggable domains
- [x] Two-phase frozen scoring + edit-channel isolation (hardened via adversarial review)
- [ ] Run against a live LLM (`AnthropicLLM` is wired; swap it in for `MockLLM`)
- [ ] A second domain (NLP / RL) to stress the pluggable-domain claim
- [ ] Container sandboxing to close the residual reward-hacking boundaries

---

## 🙏 Acknowledgements

Distilled from three lines of work: Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch) (minimal experiment kernel),
[AI-Researcher](https://github.com/HKUDS/AI-Researcher) (smoke-then-scale, paper+code grounding),
and [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) (pluggable domains,
anti-hallucination, failure→skill self-evolution).

## 📄 License

[MIT](LICENSE).
