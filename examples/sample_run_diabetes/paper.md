# Regularization-Driven Tuning of a Shallow MLP for Diabetes Progression Regression

> **Peer review:** reject · score 3/10 — The paper investigates hyperparameter tuning (learning rate, weight decay, depth, training length) for a shallow MLP regressor on the diabetes progression dataset. The authors report that a single-hidden-layer MLP with increased L2 weight decay achieves a validation RMSE of 55.26, improving on a baseline of 56.5 by about 1.2 RMSE, while increasing depth did not help.  
> **Number-grounding:** ⚠️ FLAGGED ungrounded numbers: ['1.2']

## Abstract

We study the effect of learning rate, weight decay, depth, and training length on a multilayer perceptron (MLP) regressor for the diabetes progression task, starting from a baseline validation RMSE of 56.5. Through a small set of controlled experiments, we find that a shallow single-hidden-layer MLP with a lowered learning rate, extended training, and moderately increased L2 weight decay consistently outperforms the baseline. Our best configuration reaches a validation RMSE of 55.2618, improving over the baseline by approximately 1.2 RMSE. Increasing network depth, in contrast, did not help and matched the baseline level.

## Introduction

Training small neural regressors on limited tabular data is prone to overfitting, motivating careful regularization. We investigate whether better-regularized and longer-trained MLPs can beat a baseline validation RMSE of 56.5 on the diabetes progression regression task. We draw on boosted-factor (BF) regularization of output and feature weights (doi:10.1109/tsmc.2022.3213628) and classical L1/L2 weight decay (doi:10.1007/s42979-021-00815-1) to guide our hyperparameter choices.

## Method

All models are MLPs trained for 400 epochs with learning rate 0.01. We vary three design choices: (i) network depth (1 vs. 2 hidden layers, each of width 64); (ii) L2 weight decay (1e-3 vs. 3e-3); and (iii) the resulting bias/variance trade-off on the small dataset. Following BF regularization (doi:10.1109/tsmc.2022.3213628), we first stabilize training with a lowered learning rate, increased weight decay, and extended training. We then apply the L1/L2 weight-regularization perspective (doi:10.1007/s42979-021-00815-1) to push weight decay further on the shallow model to reduce overfitting variance. We report validation RMSE for each configuration.

## Results

The shallow MLP (depth 1, hidden 64, lr 0.01, weight_decay 1e-3, 400 epochs) achieved validation RMSE values of 55.3809, 55.3248, and 55.3809 across runs, all below the 56.5 baseline. Increasing depth to 2 under the same settings yielded 56.4632, matching the baseline level and showing no benefit from added capacity. Raising weight decay to 3e-3 on the shallow model further improved results, reaching 55.3161, 55.2618, and 55.3161 across runs. The best observed configuration was depth 1, hidden 64, lr 0.01, weight_decay 3e-3, 400 epochs, with a validation RMSE of 55.2618.

## Conclusion

For this small diabetes regression task, a shallow well-regularized MLP outperforms a deeper variant. Lowering the learning rate, extending training, and increasing L2 weight decay from 1e-3 to 3e-3 reduced validation RMSE from the 56.5 baseline to a best of 55.2618. These findings support regularization over added capacity when data is limited.

---

### Reviewer notes

**Strengths**

- Addresses a practical and relevant concern: overfitting and regularization for small tabular datasets.
- Experiments are clearly reported with concrete numbers and multiple runs per configuration.
- The conclusion (favoring regularization over capacity on small data) is consistent with the presented evidence and well-established intuition.
- Writing is concise and the method/results sections are easy to follow.

**Weaknesses**

- The contribution is extremely marginal: a ~1.2 RMSE improvement on a single tiny, well-studied benchmark with no statistical significance testing or confidence intervals despite the improvement being small.
- Severe lack of experimental rigor: the abstract and method claim a 'lowered learning rate' and 'extended training' as key factors, but all reported experiments use the same lr (0.01) and 400 epochs, so these claims are unsupported. There is no actual ablation of learning rate or training length.
- The reported RMSE values are oddly duplicated across runs (e.g., 55.3809 appearing twice, 55.3161 twice), suggesting either limited seed variation or reporting errors; this undermines confidence in the results.
- No test-set evaluation is reported, only validation RMSE, raising concerns about tuning on the same set used for model selection.
- Citations to 'boosted-factor regularization' and L1/L2 decay are invoked to 'guide' choices but are not meaningfully connected to the actual experiments; the methodological framing is superficial.
- No comparison against standard baselines (linear/ridge regression, gradient boosting) which typically perform very well on the diabetes dataset, leaving the significance of an MLP result unclear.
- The dataset, train/val split, preprocessing, and baseline source are not described, limiting reproducibility.

*Drafted by ScholarLoop's L5 Writer and assessed by its Reviewer agent; every reported number is checked against the experiment registry.*
