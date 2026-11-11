"""Ewaluacja checkpointów (best.pt / last.pt) na stałych zbiorach ewaluacyjnych + średnia ± odchylenie po ziarnach.

    python -m scripts.eval_checkpoints "runs/rate_best/seed*" --ckpt best.pt
    python -m scripts.eval_checkpoints "runs/rate_best/seed*" --ckpt last.pt --episodes 1024

best.pt jest wybierany po dokładności na `val` (wartość maksymalna w trakcie treningu), więc nie jest podatny na skok straty
w ostatnim kroku. Dla Rate wypisuje też wartość `gain` z checkpointu (mechanizm weight decay, E29).
"""
from __future__ import annotations

import argparse
import glob
import statistics

SPLITS = ("val", "test_pairs", "test_x", "test_states")


def eval_run(run_dir: str, ckpt: str = "best.pt", episodes: int | None = None, device=None) -> dict:
    import torch
    from models import build_model_from_cfg
    from task import Splits, TaskConfig
    from train.train import evaluate, make_eval_sets

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(f"{run_dir}/{ckpt}", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(ck["model"])
    tcfg = TaskConfig(**cfg.get("task", {}))
    n = episodes or int(cfg["train"].get("eval_episodes", 512))
    sets = make_eval_sets(Splits(tcfg), tcfg, n, device)
    res = {s: evaluate(model, sets[s])["exact"] for s in SPLITS}
    res["step"] = ck.get("step")
    res["gain"] = float(model.gain.detach()) if hasattr(model, "gain") else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help='wzorzec glob katalogów przebiegów, w cudzysłowie, np. "runs/rate_best/seed*"')
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--episodes", type=int, default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(a.runs))
    if not dirs:
        raise SystemExit("brak katalogów")
    rows = []
    print(f"| przebieg | krok | " + " | ".join(SPLITS) + " | gain |\n|---|---|" + "---|" * (len(SPLITS) + 1))
    for d in dirs:
        r = eval_run(d, a.ckpt, a.episodes)
        rows.append(r)
        g = "-" if r["gain"] is None else f"{r['gain']:.2f}"
        print(f"| {d} | {r['step']} | " + " | ".join(f"{r[s]:.3f}" for s in SPLITS) + f" | {g} |")
    if len(rows) > 1:
        ms = [f"{statistics.fmean(r[s] for r in rows):.3f} ± {statistics.stdev(r[s] for r in rows):.3f}" for s in SPLITS]
        print(f"| **średnia ± sd ({len(rows)} ziaren)** | | " + " | ".join(ms) + " | |")


if __name__ == "__main__":
    main()
