"""EDITABLE training script for the digits-mlp engine.

Builds the (frozen) model via `prepare.build_model`, trains it on the FROZEN training split
(`prepare.load_train()` — it never sees the val set), and saves a `state_dict` + config to
$SCHOLARLOOP_ARTIFACT. It does NOT compute or report the metric — the runner runs the frozen
`prepare score` step (from pristine ROOT) on the artifact for the trusted number, so an edited
train.py cannot fabricate it. The runner injects a hyperparameter override via $SCHOLARLOOP_CONFIG.

The agent searches architecture *shape* (depth/hidden) and the training procedure (optimizer,
schedule, epochs) here. Arbitrary new model *classes* are intentionally out of reach: the model
is built by frozen code so the trusted scorer never has to unpickle agent-defined objects.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from engines.torch_vision import prepare

HPARAMS = {
    "lr": 0.1,
    "hidden": 64,
    "depth": 1,
    "weight_decay": 1e-4,
    "epochs": 40,
}


def effective_hparams() -> dict:
    return {**HPARAMS, **json.loads(os.environ.get("SCHOLARLOOP_CONFIG", "{}"))}


def train_model(seed: int, h: dict) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, y_tr = prepare.load_train()           # training data only — no val leakage
    model = prepare.build_model(h)              # frozen architecture, shaped by hparams
    opt = torch.optim.SGD(model.parameters(), lr=float(h["lr"]), momentum=0.9,
                          weight_decay=float(h["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    model.train()
    for _ in range(int(h["epochs"])):
        opt.zero_grad()
        loss_fn(model(xt), yt).backward()
        opt.step()
    return model


def main() -> None:
    seed = int(os.environ.get("SCHOLARLOOP_SEED", "0"))
    h = effective_hparams()
    model = train_model(seed, h)
    # save weights only (a state_dict), so the frozen scorer never unpickles agent objects
    torch.save({"config": h, "state_dict": dict(model.state_dict())},
               os.environ["SCHOLARLOOP_ARTIFACT"])


if __name__ == "__main__":
    main()
