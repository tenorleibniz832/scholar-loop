# A Well-Regularized Wide MLP Reliably Beats the Validation Baseline

> **Peer review:** reject · score 2/10 — The paper tests whether a wider, well-regularized single-hidden-layer MLP (hidden=128, weight_decay=5e-4, lr=0.05, 100 epochs) can beat a 5.0 validation top-1 error baseline. Across three identical runs, the authors report val_top1_err of 4.6667, 4.8148, and 4.6667, all below 5.0, and conclude that prior failures were due to undertraining at lowered learning rates rather than capacity limits.  
> **Number-grounding:** ✅ every number traces to a recorded measurement

## Abstract

We investigate whether a wider, well-regularized single-hidden-layer MLP trained at a moderate learning rate with sufficient epochs can surpass a 5.0 validation top-1 error baseline. Across three independent runs with identical configuration (hidden=128, weight_decay=5e-4, lr=0.05, depth=1, epochs=100), we observe val_top1_err values of 4.6667, 4.8148, and 4.6667, all below the 5.0 baseline. These results support the hypothesis that prior failures stemmed from undertraining at reduced learning rates rather than from limited model capacity.

## Introduction

Earlier attempts to improve validation performance lowered the learning rate but failed to beat a 5.0 val_top1_err baseline. We hypothesize that those failures reflect undertraining at the reduced learning rate rather than insufficient model capacity. Following standard guidance on regularization and ensuring adequate training iterations for convergence (Goodfellow, Bengio & Courville, Deep Learning, 2016, Ch. 7-8), we test a wider, well-regularized MLP trained at a moderate learning rate for sufficient epochs.

## Method

We train a single-hidden-layer MLP (depth=1) with hidden width 128 using weight decay of 5e-4 for regularization. The learning rate is set to a moderate 0.05, and training runs for 100 epochs to ensure convergence at this learning rate. This configuration follows the principle that regularization combined with sufficient training iterations enables stable convergence (Goodfellow, Bengio & Courville, 2016, Ch. 7-8). We repeat the experiment three times to assess stability.

## Results

All three runs surpassed the 5.0 val_top1_err baseline. Run exp_0003 achieved val_top1_err=4.6667, exp_0004 achieved 4.8148, and exp_0005 achieved 4.6667, each using the identical configuration {lr=0.05, hidden=128, depth=1, weight_decay=0.0005, epochs=100}. The consistency across runs indicates stable convergence under this setting.

## Conclusion

A wider, well-regularized MLP (hidden=128, weight_decay=5e-4) trained at lr=0.05 for 100 epochs reliably beats the 5.0 validation top-1 error baseline, with errors of 4.6667, 4.8148, and 4.6667 across three runs. This supports the view that prior shortfalls were due to undertraining at lowered learning rates rather than capacity limits, consistent with established guidance on regularization and convergence (Goodfellow, Bengio & Courville, 2016, Ch. 7-8).

---

### Reviewer notes

**Strengths**

- Reports results across three repeated runs, providing some indication of variance/stability.
- Clearly states the hypothesis, configuration, and the baseline being compared against.
- The narrow margin between runs suggests reasonable reproducibility of the reported numbers.

**Weaknesses**

- The contribution is extremely thin: it amounts to a single hyperparameter configuration beating a single threshold, with no ablations isolating the effect of width, learning rate, regularization, or epochs. The causal claim about 'undertraining vs. capacity' is unsupported by controlled experiments.
- Critical experimental details are missing: dataset name, task, number of classes, train/test split, evaluation protocol, optimizer, batch size, and what the baseline actually is. Without these the results are uninterpretable and unverifiable.
- The margin above baseline (4.6-4.8 vs 5.0) is tiny and reported without any statistical analysis, confidence intervals, or significance test. With only three runs and no error bars on the baseline itself, 'reliably beats' is overstated.
- The hypothesis (undertraining caused prior failures) is not actually tested—there is no comparison against the lower-learning-rate configurations or against a narrower/undertrained model. The conclusion does not follow from the evidence presented.
- No comparison to baselines beyond the single 5.0 threshold, no related work, and no discussion of generalization to other settings. The reliance on a general textbook citation does not substantiate specific design choices.
- The reported errors look unusually precise/low (e.g., 4.6667) but the scale and meaning of 'top-1 error' values around 5.0 are never explained (percent? something else?), raising clarity concerns.

*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; every reported number is checked against the experiment registry.*
