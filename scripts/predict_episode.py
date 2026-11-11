"""Uruchom wytrenowany model na epizodzie JSON i wypełnij pola predicted_bits / correct.

    python -m scripts.predict_episode --run runs/gru/seed0 --episode docs/przyklad_epizodu.json --out odpowiedz.json

W trybie legs_mode=teacher model dostaje prawdziwe położenie nóg z pola legs_before; w init_only tylko w kroku 0.
"""
import argparse
import json

import numpy as np
import torch

from models import build_model_from_cfg
from task.generator import write_episode_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="katalog przebiegu, np. runs/gru/seed0")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()

    ck = torch.load(f"{a.run}/{a.ckpt}", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model_from_cfg(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    ep = json.load(open(a.episode, encoding="utf-8"))
    steps = ep["steps"]
    screens = torch.tensor([s["screen"] for s in steps], dtype=torch.float32)[None]
    legs = torch.tensor([s["legs_before"] for s in steps], dtype=torch.float32)[None]
    if cfg.get("task", {}).get("legs_mode", "teacher") == "init_only":
        legs[:, 1:] = 0
    with torch.no_grad():
        pred = (model(screens, legs)[0] > 0).int().tolist()
    for s, p in zip(steps, pred):
        s["predicted_bits"] = p
        s["correct"] = bool(p == s["target_bits"])
    write_episode_json(ep, a.out)
    print(f"poprawnych kroków: {sum(s['correct'] for s in steps)}/{len(steps)} -> {a.out}")


if __name__ == "__main__":
    main()
