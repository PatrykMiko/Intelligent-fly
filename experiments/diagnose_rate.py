"""Diagnostyka modelu Rate (bez pełnego treningu).

    python -m experiments.diagnose_rate
    python -m experiments.diagnose_rate --set model.init_gain=14 model.input_scale=4

Wypisuje:
  1) rozkład wejść do tanh na starcie (czy sieć pracuje w zakresie nieliniowym),
  2) normy gradientów per parametr (czy gradient w ogóle dociera do wejścia i odczytu),
  3) próbę zapamiętania JEDNEGO stałego batcha 16 epizodów (czy ścieżka wejście -> wyjście w ogóle działa).
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from models import build_model_from_cfg, count_params
from task import Splits, TaskConfig, make_batch
from train.train import load_config, to_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/rate_small.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--overfit-steps", type=int, default=400)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = TaskConfig(**cfg.get("task", {}))
    b = to_torch(make_batch(Splits(tcfg), "train", 16, tcfg, np.random.default_rng(0)), dev)
    model = build_model_from_cfg(cfg).to(dev)
    print("parametry:", count_params(model))

    # 1) aktywność neuronów na starcie
    rec, orig = [], model.act
    model.act = lambda z: (rec.append(z.detach()), orig(z))[1]
    logits = model(b["screens"], b["legs_in"])
    model.act = orig
    z = torch.cat([r.flatten() for r in rec])
    print(f"wejście do tanh: mean|z|={z.abs().mean():.3f}  frakcja |z|<0.3 (liniowo)={(z.abs() < 0.3).float().mean():.3f}"
          f"  frakcja |z|>2 (nasycone)={(z.abs() > 2).float().mean():.3f}")
    print(f"logity: std po batchu={logits.std(0).mean():.4f}  średnia |logit|={logits.abs().mean():.4f}")

    # 2) normy gradientów per parametr
    loss = nn.functional.binary_cross_entropy_with_logits(logits, b["targets"])
    loss.backward()
    for n, p in model.named_parameters():
        g = float("nan") if p.grad is None else p.grad.norm().item()
        print(f"  grad {n:14s} |g|={g:.2e}   |w|={p.norm().item():.2e}")

    # 3) zapamiętanie jednego stałego batcha
    model.zero_grad()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("lr", 3e-3)))
    for step in range(1, a.overfit_steps + 1):
        lg = model(b["screens"], b["legs_in"])
        loss = nn.functional.binary_cross_entropy_with_logits(lg, b["targets"])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            exact = ((lg > 0).float() == b["targets"]).all(-1).float().mean().item()
            print(f"  overfit krok {step:4d} strata {loss.item():.4f} exact {exact:.3f}")


if __name__ == "__main__":
    main()
