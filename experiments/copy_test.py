"""Test "kopiowania": cel = bity X z ekranu (zadanie LINIOWE, bez XOR). Wymaga kodowania binary.

    python -m experiments.copy_test --set task.screen_encoding=binary
    python -m experiments.copy_test --set task.screen_encoding=binary model.input_scale=4

Strata -> ~0: ścieżka ekran -> graf -> odczyt działa (problem Rate to nieliniowość/XOR).
Strata stoi na 0.69: ścieżka sygnału jest martwa (zanik gradientu / sygnał nie dociera).
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from models import build_model_from_cfg
from task import Splits, TaskConfig, make_batch
from train.train import load_config, to_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/rate_small.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    assert cfg["task"]["screen_encoding"] == "binary", "ustaw --set task.screen_encoding=binary"
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = TaskConfig(**cfg["task"])
    sp, rng = Splits(tcfg), np.random.default_rng(0)
    model = build_model_from_cfg(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, a.steps + 1):
        b = to_torch(make_batch(sp, "train", 64, tcfg, rng), dev)
        target = b["screens"][..., :6]                    # kopiuj bity X z ekranu (zadanie liniowe)
        loss = nn.functional.binary_cross_entropy_with_logits(model(b["screens"], b["legs_in"]), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(step, round(loss.item(), 4))


if __name__ == "__main__":
    main()
