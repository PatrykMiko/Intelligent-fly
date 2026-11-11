"""Pętla treningowa (BPTT, Adam, logi CSV, checkpointy).

Użycie (z katalogu głównego repo):
    python -m train.train --config experiments/configs/gru.yaml
    python -m train.train --config experiments/configs/gru.yaml --set seed=1 model.hidden=64 train.steps=2000
    # opcjonalnie: --set train.lr_schedule=cosine   (spadek lr do 0 pod koniec treningu)
    #              --set train.weight_decay=0.05    (AdamW; domyślnie 0 = zwykły Adam)
    #              --set "train.wd_exclude=[gain,bias,alpha_logit,b_out]"   (te parametry bez decay)

Wyniki w  runs/<name>/seed<seed>/ :
    config.yaml   pełny (po nadpisaniach) config tego przebiegu
    metrics.csv   wiersze: step, split, loss, exact, bit0..bit5, n_steps
    best.pt       checkpoint z najlepszą dokładnością całej liczby na `val`
    last.pt       ostatni checkpoint
    results.json  końcowe metryki per podział + liczba parametrów + czas
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from models import build_model_from_cfg, count_params
from task import SPLITS, Splits, TaskConfig, make_batch

LN2 = float(np.log(2.0))
EVAL_SPLITS = tuple(s for s in SPLITS if s != "train")


# ------------------------------------------------------------------ config
def deep_set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        deep_set(cfg, key, yaml.safe_load(raw))
    return cfg


# ------------------------------------------------------------------ dane
KEYS = ("screens", "legs_in", "targets", "measure")


def to_torch(batch: dict, device) -> dict:
    return {k: torch.as_tensor(batch[k]).to(device) for k in KEYS}


@torch.no_grad()
def evaluate(model: nn.Module, data: dict, chunk: int = 256) -> dict:
    """Metryki liczone tylko na krokach z maską `measure`."""
    was_training = model.training
    model.eval()
    n = data["screens"].shape[0]
    steps = loss_sum = exact_sum = 0.0
    bit_sum = torch.zeros(6, device=data["screens"].device)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        logits = model(data["screens"][sl], data["legs_in"][sl])
        tgt, m = data["targets"][sl], data["measure"][sl].float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        correct = ((logits > 0).float() == tgt).float()
        loss_sum += float((bce.mean(-1) * m).sum())
        exact_sum += float((correct.min(-1).values * m).sum())
        bit_sum += (correct * m[..., None]).sum((0, 1))
        steps += float(m.sum())
    model.train(was_training)
    d = max(steps, 1.0)
    return {"loss": loss_sum / d, "exact": exact_sum / d, "bit_acc": (bit_sum / d).tolist(), "n_steps": int(steps)}


def make_eval_sets(splits: Splits, task_cfg: TaskConfig, n_episodes: int, device) -> dict:
    rng = np.random.default_rng(task_cfg.split_seed + 12345)         # stałe zbiory ewaluacyjne dla wszystkich ziaren
    return {s: to_torch(make_batch(splits, s, n_episodes, task_cfg, rng), device) for s in EVAL_SPLITS}


# ------------------------------------------------------------------ optymalizator
def build_optimizer(model: nn.Module, tcfg: dict):
    """Adam (weight_decay = 0, jak dotychczas) albo AdamW.

    train.weight_decay: współczynnik AdamW (domyślnie 0).
    train.wd_exclude:   lista fragmentów nazw parametrów, które NIE są zmniejszane przez weight decay
                        (np. [gain, bias, alpha_logit, b_out] dla Rate: decay tylko na W_in, W_sens, v).
    """
    lr = float(tcfg.get("lr", 3e-3))
    wd = float(tcfg.get("weight_decay", 0.0))
    if wd <= 0:
        return torch.optim.Adam(model.parameters(), lr=lr)
    excl = list(tcfg.get("wd_exclude") or [])
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if any(e in name for e in excl) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr)


# ------------------------------------------------------------------ główna pętla
def run(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = cfg.get("device", "auto")
    device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))

    task_cfg = TaskConfig(**cfg.get("task", {}))
    splits = Splits(task_cfg)
    tcfg = cfg["train"]
    model = build_model_from_cfg(cfg).to(device)
    n_params = count_params(model)

    run_dir = Path(cfg.get("out_dir", "runs")) / cfg["name"] / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[{cfg['name']} seed={seed}] parametry: {n_params:,}  urządzenie: {device}  strata zgadywania = ln2 = {LN2:.4f}")
    print("podziały:", splits.describe())

    eval_sets = make_eval_sets(splits, task_cfg, int(tcfg.get("eval_episodes", 512)), device)
    opt = build_optimizer(model, tcfg)
    steps, bs = int(tcfg["steps"]), int(tcfg.get("batch_size", 128))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
             if tcfg.get("lr_schedule") == "cosine" else None)      # domyślnie wyłączone (stała lr)
    clip = float(tcfg.get("grad_clip", 1.0))
    eval_every, log_every = int(tcfg.get("eval_every", 250)), int(tcfg.get("log_every", 50))

    csv_path = run_dir / "metrics.csv"
    fields = ["step", "split", "loss", "exact"] + [f"bit{i}" for i in range(6)] + ["n_steps"]
    best_val, t0, last_eval = -1.0, time.time(), {}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()

        def log_row(step, split, m):
            row = {"step": step, "split": split, "loss": m["loss"], "exact": m.get("exact", ""), "n_steps": m.get("n_steps", "")}
            for i in range(6):
                row[f"bit{i}"] = m["bit_acc"][i] if "bit_acc" in m else ""
            w.writerow(row); fh.flush()

        model.train()
        for step in range(1, steps + 1):
            b = to_torch(make_batch(splits, "train", bs, task_cfg, rng), device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(b["screens"], b["legs_in"]), b["targets"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
            if step % log_every == 0:
                log_row(step, "train_batch", {"loss": loss.item()})
            if step % eval_every == 0 or step == steps:
                last_eval = {s: evaluate(model, d) for s, d in eval_sets.items()}
                for s, m in last_eval.items():
                    log_row(step, s, m)
                v = last_eval["val"]
                gtxt = f" | gain {model.gain.item():.2f}" if hasattr(model, "gain") else ""
                print(f"  krok {step:6d} | train {loss.item():.4f} | val loss {v['loss']:.4f} exact {v['exact']:.3f} "
                      f"| test_pairs {last_eval['test_pairs']['exact']:.3f} test_x {last_eval['test_x']['exact']:.3f} "
                      f"test_states {last_eval['test_states']['exact']:.3f}" + gtxt)
                if v["exact"] > best_val:
                    best_val = v["exact"]
                    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, run_dir / "best.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": steps}, run_dir / "last.pt")
    results = {"name": cfg["name"], "seed": seed, "n_params": n_params, "steps": steps,
               "seconds": round(time.time() - t0, 1), "ln2": LN2,
               "final_gain": (model.gain.item() if hasattr(model, "gain") else None), "final": last_eval, "best_val_exact": best_val}
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KLUCZ=WARTOŚĆ", help="nadpisania, np. seed=1 model.hidden=64")
    args = ap.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
