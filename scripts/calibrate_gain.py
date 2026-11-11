"""Dobór wzmocnienia (gain) modelu Rate bez treningu: nasycenie, czułość na bieżącą instrukcję vs na historię.

    python -m scripts.calibrate_gain --config experiments/configs/rate_small.yaml --gains 2 4 6 8 10 --set model.reset_each_step=true
    python -m scripts.calibrate_gain --gains 1 1.5 1.9 2.3 --set model.reset_each_step=false

Dla każdego gain wypisuje na wyjściu modelu (logity w ostatnim kroku epizodu):
  nasycenie = frakcja wejść do aktywacji z |z| > 2 (średnio po wszystkich kroków wewnętrznych),
  input-sens = średnia |zmiana logitu| po zmianie instrukcji w OSTATNIM kroku,
  history-sens = średnia |zmiana logitu| po zmianie instrukcji w PIERWSZYM kroku,
  oraz promień spektralny rho grafu (gain krytyczny ~ 1 / rho).
Dobry punkt: małe nasycenie (<~15%), duży input-sens względem średniego |logitu|; przy reset_each_step=true history-sens = 0.
Metoda użyta w docs/eksperymenty.md E8 (tam w numpy, na aktywności neuronów ruchowych).
"""
import argparse
import copy

import numpy as np
import torch

from graphs import spectral_radius
from models import build_graph, build_model_from_cfg
from task import Splits, TaskConfig, make_batch
from train.train import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/rate_small.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--gains", nargs="+", type=float, default=[2, 4, 6, 8, 10])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rho-iters", type=int, default=300)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = TaskConfig(**cfg.get("task", {}))
    b = make_batch(Splits(tcfg), "train", a.batch, tcfg, np.random.default_rng(1))
    scr, legs = torch.as_tensor(b["screens"]).to(dev), torch.as_tensor(b["legs_in"]).to(dev)
    scr_last, scr_first = scr.clone(), scr.clone()
    scr_last[:, -1] = torch.roll(scr[:, -1], 7, dims=-1)
    scr_first[:, 0] = torch.roll(scr[:, 0], 7, dims=-1)

    rho = spectral_radius(build_graph(cfg["model"].get("graph")), iters=a.rho_iters)
    print(f"promień spektralny rho = {rho:.3f}; gain krytyczny ~ 1/rho = {1 / rho:.2f}; "
          f"reset_each_step = {cfg['model'].get('reset_each_step', False)}")
    print("gain (rho_eff) | nasycenie | mean|logit| | input-sens | history-sens")
    for g in a.gains:
        c = copy.deepcopy(cfg)
        c["model"]["init_gain"] = g
        torch.manual_seed(0)
        model = build_model_from_cfg(c).to(dev).eval()
        sat, orig = [], model.act
        model.act = lambda z: (sat.append((z.detach().abs() > 2).float().mean().item()), orig(z))[1]
        with torch.no_grad():
            y0 = model(scr, legs)[:, -1]
            yl = model(scr_last, legs)[:, -1]
            yf = model(scr_first, legs)[:, -1]
        model.act = orig
        print(f"{g:5.2f} ({g * rho:4.2f}) | {np.mean(sat):.3f} | {y0.abs().mean().item():.4f} | "
              f"{(y0 - yl).abs().mean().item():.4f} | {(y0 - yf).abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
