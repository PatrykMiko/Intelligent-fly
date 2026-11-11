"""Zbiorcza tabela wyników z runs/<nazwa>/seed*/results.json: średnia ± odchylenie standardowe po ziarnach.

    python -m scripts.summarize_runs runs                       # tabela dokładności całej liczby
    python -m scripts.summarize_runs runs --metric loss
    python -m scripts.summarize_runs runs --bits --md docs/wyniki_auto.md

Uwaga: to wartości z OSTATNIEGO kroku treningu (results.json["final"]), nie z best.pt.
Odchylenie liczone z n-1 (próbkowe); przy jednym ziarnie wypisujemy "n=1".
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SPLITS = ("val", "test_pairs", "test_x", "test_states")


def load_runs(root) -> dict:
    runs: dict = {}
    for f in sorted(Path(root).glob("*/seed*/results.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        runs.setdefault(r["name"], []).append(r)
    return runs


def mean_sd(xs):
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) if len(xs) > 1 else float("nan"))


def summarize(runs: dict, metric: str = "exact") -> list:
    rows = []
    for name, rs in sorted(runs.items()):
        row = {"name": name, "n_seeds": len(rs), "n_params": rs[0].get("n_params"), "steps": rs[0].get("steps")}
        for s in SPLITS:
            vals = [r["final"][s][metric] for r in rs if s in r.get("final", {})]
            row[s] = mean_sd(vals) if vals else None
        rows.append(row)
    return rows


def per_bit(runs: dict) -> dict:
    return {name: {s: [statistics.fmean(r["final"][s]["bit_acc"][i] for r in rs) for i in range(6)] for s in SPLITS}
            for name, rs in sorted(runs.items())}


def fmt(ms, digits=3) -> str:
    if ms is None:
        return "-"
    m, sd = ms
    return f"{m:.{digits}f} (n=1)" if sd != sd else f"{m:.{digits}f} ± {sd:.{digits}f}"


def to_markdown(rows: list, metric: str, bits: dict | None = None) -> str:
    out = [f"| model | ziarna | parametry | kroki | " + " | ".join(SPLITS) + " |",
           "|---|---|---|---|" + "---|" * len(SPLITS)]
    for r in rows:
        out.append(f"| {r['name']} | {r['n_seeds']} | {r['n_params']:,} | {r['steps']} | "
                   + " | ".join(fmt(r[s]) for s in SPLITS) + " |")
    out.append(f"\n(metryka: {metric}; średnia ± odchylenie po ziarnach, ostatni krok treningu)")
    if bits:
        out.append("\n**Dokładność per bit (bit 0..5), średnia po ziarnach:**\n")
        out += ["| model | podział | " + " | ".join(f"bit{i}" for i in range(6)) + " |", "|---|---|" + "---|" * 6]
        for name, d in bits.items():
            for s in SPLITS:
                out.append(f"| {name} | {s} | " + " | ".join(f"{x:.3f}" for x in d[s]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="runs")
    ap.add_argument("--metric", default="exact", choices=["exact", "loss"])
    ap.add_argument("--bits", action="store_true", help="dołącz dokładność per bit (RQ5)")
    ap.add_argument("--md", default=None, help="zapisz tabelę do pliku Markdown")
    a = ap.parse_args()
    runs = load_runs(a.root)
    if not runs:
        raise SystemExit(f"brak results.json w {a.root}/*/seed*/")
    md = to_markdown(summarize(runs, a.metric), a.metric, per_bit(runs) if a.bits else None)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
        print("zapisano", a.md)


if __name__ == "__main__":
    main()
