"""Analiza błędów po przeniesieniach (RQ5): gdzie model się myli - zależnie od bitu, operacji, przeniesienia i długości łańcucha.

    python -m scripts.error_analysis --run runs/gru/seed0 --split test_pairs
    python -m scripts.error_analysis --run runs/gru_binary/seed0 --split test_x --episodes 4096 --ckpt last.pt

Rdzeń (error_table, carry_run_length) używa tylko numpy; PyTorch ładowany jest dopiero w main().
"""
from __future__ import annotations

import argparse

import numpy as np

N_BITS = 6


def carry_run_length(c) -> np.ndarray:
    """c (...,6) przeniesienia WEJŚCIOWE; wynik (...,6): długość ciągu jedynek kończącego się w bicie i
    (c[i], c[i-1], ...). 0 = brak przeniesienia do bitu i."""
    c = np.asarray(c).astype(np.int64)
    out, run = np.zeros_like(c), np.zeros(c.shape[:-1], dtype=np.int64)
    for i in range(N_BITS):
        run = (run + 1) * c[..., i]
        out[..., i] = run
    return out


def error_table(pred, target, carries, op, measure, state_prev=None, x=None) -> dict:
    """pred/target/carries: (N,T,6); op, measure: (N,T). Zwraca dwie tabele (listy słowników) z odsetkiem błędów.

    by_carry: dla każdego bitu i każdej pary (operacja, przeniesienie wejściowe 0/1)
    by_run:   dla każdego bitu i długości łańcucha przeniesienia 0..5
    by_bits:  (tylko gdy podano state_prev i x) dla każdego bitu: błąd wg (operacja, bit stanu s_i, bit X x_i, przeniesienie c_i)
    """
    m = np.asarray(measure, bool)
    wrong = np.asarray(pred) != np.asarray(target)
    op, carries = np.asarray(op), np.asarray(carries)
    run = carry_run_length(carries)

    def rate(sel, bit):
        n = int(sel.sum())
        return n, (float(wrong[..., bit][sel].mean()) if n else float("nan"))

    by_carry, by_run = [], []
    for bit in range(N_BITS):
        for opv, name in ((1, "+"), (-1, "-")):
            for cin in (0, 1):
                n, e = rate(m & (op == opv) & (carries[..., bit] == cin), bit)
                by_carry.append({"bit": bit, "op": name, "carry_in": cin, "n": n, "error": e})
        for L in range(0, N_BITS):
            n, e = rate(m & (run[..., bit] == L), bit)
            by_run.append({"bit": bit, "run": L, "n": n, "error": e})
    out = {"by_carry": by_carry, "by_run": by_run}
    pr, tg = np.asarray(pred), np.asarray(target)
    by_t = []
    for step in range(pr.shape[1]):
        sel = m[:, step]
        n = int(sel.sum())
        ok = (pr[:, step] == tg[:, step]).all(-1)
        by_t.append({"t": step, "n": n, "exact": float(ok[sel].mean()) if n else float("nan")})
    out["by_t"] = by_t
    if state_prev is not None and x is not None:
        from task import to_bits
        sb, xb = to_bits(state_prev), to_bits(x)
        by_bits = []
        for bit in range(N_BITS):
            for opv, name in ((1, "+"), (-1, "-")):
                for s_i in (0, 1):
                    for x_i in (0, 1):
                        for c_i in (0, 1):
                            sel = m & (op == opv) & (sb[..., bit] == s_i) & (xb[..., bit] == x_i) & (carries[..., bit] == c_i)
                            n, e = rate(sel, bit)
                            by_bits.append({"bit": bit, "op": name, "s": s_i, "x": x_i, "c": c_i, "n": n, "error": e})
        out["by_bits"] = by_bits
    return out


def _fmt(e) -> str:
    return "-" if e != e else f"{100 * e:.1f}%"


def to_markdown(tab: dict, detail_bits=(5,)) -> str:
    out = ["**Błędy per bit, operacja i przeniesienie wejściowe** (n = liczba kroków; błąd w % kroków)\n",
           "| bit | op | carry=0 (n) | błąd | carry=1 (n) | błąd |", "|---|---|---|---|---|---|"]
    d = {(r["bit"], r["op"], r["carry_in"]): r for r in tab["by_carry"]}
    for bit in range(N_BITS):
        for op in ("+", "-"):
            a, b = d[(bit, op, 0)], d[(bit, op, 1)]
            out.append(f"| {bit} | {op} | {a['n']} | {_fmt(a['error'])} | {b['n']} | {_fmt(b['error'])} |")
    out += ["\n**Błędy per bit i długość łańcucha przeniesienia** (0 = brak przeniesienia do tego bitu)\n",
            "| bit | " + " | ".join(f"L={L}" for L in range(N_BITS)) + " |", "|---|" + "---|" * N_BITS]
    d = {(r["bit"], r["run"]): r for r in tab["by_run"]}
    for bit in range(N_BITS):
        out.append(f"| {bit} | " + " | ".join(f"{_fmt(d[(bit, L)]['error'])} ({d[(bit, L)]['n']})" for L in range(N_BITS)) + " |")
    if "by_t" in tab:
        out += ["\n**Dokładność całej liczby wg kroku t w epizodzie** (n = liczba zmierzonych kroków; dla `init_only` spadek z t oznacza gubienie stanu)\n",
                "| t | " + " | ".join(str(r["t"]) for r in tab["by_t"]) + " |", "|---|" + "---|" * len(tab["by_t"]),
                "| exact | " + " | ".join("-" if r["exact"] != r["exact"] else f"{r['exact']:.2f}" for r in tab["by_t"]) + " |",
                "| n | " + " | ".join(str(r["n"]) for r in tab["by_t"]) + " |"]
    if "by_bits" in tab:
        for bit in detail_bits:
            out += [f"\n**Bit {bit}: błąd wg (operacja, bit stanu s, bit X x, przeniesienie c)** - n w nawiasie\n",
                    "| op | s | x | c=0 | c=1 |", "|---|---|---|---|---|"]
            d = {(r["bit"], r["op"], r["s"], r["x"], r["c"]): r for r in tab["by_bits"]}
            for op in ("+", "-"):
                for s_i in (0, 1):
                    for x_i in (0, 1):
                        a, b = d[(bit, op, s_i, x_i, 0)], d[(bit, op, s_i, x_i, 1)]
                        out.append(f"| {op} | {s_i} | {x_i} | {_fmt(a['error'])} ({a['n']}) | {_fmt(b['error'])} ({b['n']}) |")
    return "\n".join(out)


def main():
    import torch
    from models import build_model_from_cfg
    from task import Splits, TaskConfig, make_batch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="katalog przebiegu, np. runs/gru/seed0")
    ap.add_argument("--split", default="test_pairs")
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--md", default=None)
    ap.add_argument("--detail-bits", nargs="+", type=int, default=[5], help="bity, dla których wypisać rozbicie wg (op, s, x, c)")
    a = ap.parse_args()
    ck = torch.load(f"{a.run}/{a.ckpt}", map_location="cpu")
    cfg = ck["cfg"]
    model = build_model_from_cfg(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tcfg = TaskConfig(**cfg.get("task", {}))
    b = make_batch(Splits(tcfg), a.split, a.episodes, tcfg, np.random.default_rng(777))
    with torch.no_grad():
        logits = model(torch.as_tensor(b["screens"]), torch.as_tensor(b["legs_in"]))
    pred = (logits > 0).int().numpy()
    tab = error_table(pred, b["targets"].astype(int), b["carries"], b["op"], b["measure"], b["state_prev"], b["x"])
    md = f"### {a.run} ({a.ckpt}), podział {a.split}, {a.episodes} epizodów\n\n" + to_markdown(tab, tuple(a.detail_bits))
    print(md)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
