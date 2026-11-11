"""Krok 16: dobór rozmiarów modeli do podobnej liczby parametrów.

    python -m scripts.count_params --config experiments/configs/rate_small.yaml
Wypisuje budżet (liczba trenowanych parametrów Rate) i `hidden` dla każdego modelu odniesienia.
"""
import argparse

import yaml

from models import build_model_from_cfg, count_params, match_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/rate_small.yaml")
    ap.add_argument("--budget", type=int, default=None, help="ręczny budżet (domyślnie: parametry modelu z configu)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    budget = a.budget or count_params(build_model_from_cfg(cfg))
    print(f"budżet = {budget:,} trenowanych parametrów ({cfg['model']['name']})")
    for name in ("mlp", "elman", "gru", "lstm", "transformer"):
        h, n = match_hidden(name, budget)
        print(f"  {name:12s} hidden={h:4d}  parametry={n:,}  (odchylenie {100 * (n - budget) / budget:+.1f}%)")


if __name__ == "__main__":
    main()
