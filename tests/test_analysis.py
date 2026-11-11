"""Testy skryptów analizy (numpy + biblioteka standardowa)."""
import csv
import tempfile
from pathlib import Path

import numpy as np

from scripts.bit_order import first_step_above, load
from scripts.error_analysis import carry_run_length, error_table, to_markdown
from task import Splits, TaskConfig, make_batch


def test_carry_run_length_known():
    c = np.array([[0, 1, 1, 0, 1, 1]], dtype=np.uint8)
    assert carry_run_length(c).tolist() == [[0, 1, 2, 0, 1, 2]]
    assert carry_run_length(np.zeros((2, 3, 6))).sum() == 0


def test_error_table_recovers_injected_errors():
    cfg = TaskConfig(seq_len=16)
    b = make_batch(Splits(cfg), "train", 600, cfg, np.random.default_rng(0))
    tgt = b["targets"].astype(int)
    rng = np.random.default_rng(1)
    pred = tgt.copy()
    flip = (b["carries"][..., 5] == 1) & (rng.random(tgt.shape[:2]) < 0.5)   # błędy tylko w bicie 5, gdy carry=1
    pred[..., 5][flip] ^= 1
    tab = error_table(pred, tgt, b["carries"], b["op"], b["measure"])
    rows = {(r["bit"], r["op"], r["carry_in"]): r for r in tab["by_carry"]}
    for op in ("+", "-"):
        assert rows[(5, op, 0)]["error"] == 0.0
        assert abs(rows[(5, op, 1)]["error"] - 0.5) < 0.1
        assert rows[(3, op, 1)]["error"] == 0.0                          # inne bity bez błędów
    runs = {(r["bit"], r["run"]): r for r in tab["by_run"]}
    assert runs[(5, 0)]["error"] == 0.0 and runs[(5, 1)]["error"] > 0.3
    md = to_markdown(tab)
    assert "L=5" in md and "| 5 | + |" in md


def test_error_table_respects_measure_mask():
    cfg = TaskConfig(seq_len=4)
    b = make_batch(Splits(cfg), "test_pairs", 100, cfg, np.random.default_rng(0))
    tgt = b["targets"].astype(int)
    pred = 1 - tgt                                                       # wszystko źle
    tab = error_table(pred, tgt, b["carries"], b["op"], b["measure"])
    total_n = sum(r["n"] for r in tab["by_carry"] if r["bit"] == 0)
    assert total_n == int(b["measure"].sum())                           # liczą się tylko kroki z maską
    assert all(r["error"] == 1.0 for r in tab["by_carry"] if r["n"])


def test_bit_order_first_step_above():
    rows = []
    for step, accs in ((250, [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]), (500, [0.95, 0.6, 0.3, 0.2, 0.1, 0.0]),
                       (750, [1.0, 0.92, 0.91, 0.5, 0.1, 0.0])):
        r = {"step": step, "split": "val"}
        r.update({f"bit{i}": a for i, a in enumerate(accs)})
        rows.append(r)
        rows.append({**r, "split": "test_x", **{f"bit{i}": 1.0 for i in range(6)}})   # inny podział ignorowany
    assert first_step_above(rows, "val", 0.9) == [500, 750, 750, None, None, None]


def test_bit_order_reads_csv():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "metrics.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "split", "loss", "exact"] + [f"bit{i}" for i in range(6)] + ["n_steps"])
            w.writerow([250, "train_batch", 0.69, "", "", "", "", "", "", "", ""])       # puste pola bitów
            w.writerow([250, "val", 0.5, 0.1] + [0.95] * 6 + [100])
        assert first_step_above(load(str(p)), "val", 0.9) == [250] * 6


def test_error_table_by_bits_isolates_injected_pattern():
    from task import to_bits
    cfg = TaskConfig(seq_len=16)
    b = make_batch(Splits(cfg), "train", 800, cfg, np.random.default_rng(3))
    tgt = b["targets"].astype(int)
    pred = tgt.copy()
    s5, x5 = to_bits(b["state_prev"])[..., 5], to_bits(b["x"])[..., 5]
    bad = (b["op"] == 1) & (s5 == 1) & (x5 == 1)                       # błąd tylko dla (+, s5=1, x5=1) w bicie 5
    pred[..., 5][bad] ^= 1
    tab = error_table(pred, tgt, b["carries"], b["op"], b["measure"], b["state_prev"], b["x"])
    rows = {(r["bit"], r["op"], r["s"], r["x"], r["c"]): r for r in tab["by_bits"]}
    for c in (0, 1):
        if rows[(5, "+", 1, 1, c)]["n"]:
            assert rows[(5, "+", 1, 1, c)]["error"] == 1.0
        assert rows[(5, "-", 1, 1, c)]["error"] in (0.0,) or rows[(5, "-", 1, 1, c)]["n"] == 0
        assert rows[(5, "+", 0, 0, c)]["error"] in (0.0,) or rows[(5, "+", 0, 0, c)]["n"] == 0
    assert "Bit 5" in to_markdown(tab, (5,))


def test_error_table_by_t_detects_memory_decay():
    cfg = TaskConfig(seq_len=10)
    b = make_batch(Splits(cfg), "train", 400, cfg, np.random.default_rng(4))
    tgt = b["targets"].astype(int)
    pred = tgt.copy()
    rng = np.random.default_rng(5)
    late = np.zeros(tgt.shape[:2], bool); late[:, 5:] = True
    flip = late & (rng.random(tgt.shape[:2]) < 0.5)                    # błędy tylko w krokach t >= 5
    pred[..., 0][flip] ^= 1
    tab = error_table(pred, tgt, b["carries"], b["op"], b["measure"])
    ex = [r["exact"] for r in tab["by_t"]]
    assert all(e == 1.0 for e in ex[:5]) and all(abs(e - 0.5) < 0.1 for e in ex[5:])
    assert "exact" in to_markdown(tab)
