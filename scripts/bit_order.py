"""Kolejność uczenia bitów (RQ5): w którym kroku treningu dokładność każdego bitu pierwszy raz przekracza próg.

    python -m scripts.bit_order "runs/gru/seed*/metrics.csv" --split val --thr 0.9
Dla wielu plików (ziaren) wypisuje medianę kroku i liczbę ziaren, które próg osiągnęły ('-' = nigdy).
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics


def first_step_above(rows: list, split: str, thr: float, n_bits: int = 6) -> list:
    """rows: lista słowników z metrics.csv. Zwraca listę n_bits: pierwszy krok z acc >= thr albo None."""
    out = [None] * n_bits
    for r in rows:
        if r["split"] != split:
            continue
        for i in range(n_bits):
            if out[i] is None and r.get(f"bit{i}") not in (None, "") and float(r[f"bit{i}"]) >= thr:
                out[i] = int(r["step"])
    return out


def load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="wzorzec glob do metrics.csv, w cudzysłowie")
    ap.add_argument("--split", default="val")
    ap.add_argument("--thr", type=float, default=0.9)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit("brak plików")
    per = [first_step_above(load(f), a.split, a.thr) for f in files]
    print(f"{len(files)} plików; podział {a.split}; próg {a.thr}")
    print("| bit | mediana kroku | ziarna z progiem |\n|---|---|---|")
    for i in range(6):
        steps = [p[i] for p in per if p[i] is not None]
        med = f"{statistics.median(steps):.0f}" if steps else "-"
        print(f"| {i} | {med} | {len(steps)}/{len(per)} |")


if __name__ == "__main__":
    main()
