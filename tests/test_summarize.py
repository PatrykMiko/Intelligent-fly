"""Testy skryptu zbiorczego (tylko biblioteka standardowa). Uruchom: pytest tests/test_summarize.py"""
import json
import math
import tempfile
from pathlib import Path

from scripts.summarize_runs import fmt, load_runs, mean_sd, per_bit, summarize, to_markdown


def _fake_run(root, name, seed, exact_by_split, bits=0.9):
    d = Path(root) / name / f"seed{seed}"
    d.mkdir(parents=True)
    final = {s: {"loss": 0.1, "exact": e, "bit_acc": [bits] * 6, "n_steps": 100} for s, e in exact_by_split.items()}
    (d / "results.json").write_text(json.dumps({"name": name, "seed": seed, "n_params": 1234, "steps": 10,
                                                "final": final}), encoding="utf-8")


def _splits(e):
    return {"val": e, "test_pairs": e, "test_x": e / 2, "test_states": e}


def test_mean_sd_and_fmt():
    m, sd = mean_sd([0.9, 1.0])
    assert abs(m - 0.95) < 1e-9 and abs(sd - 0.0707106781) < 1e-6
    assert math.isnan(mean_sd([0.5])[1])
    assert fmt((0.95, 0.05)) == "0.950 ± 0.050" and fmt((0.5, float("nan"))) == "0.500 (n=1)" and fmt(None) == "-"


def test_load_and_summarize_two_models():
    with tempfile.TemporaryDirectory() as d:
        for seed, e in enumerate([0.90, 0.94]):
            _fake_run(d, "gru", seed, _splits(e))
        _fake_run(d, "mlp", 0, _splits(0.5))
        runs = load_runs(d)
        assert set(runs) == {"gru", "mlp"} and len(runs["gru"]) == 2
        rows = {r["name"]: r for r in summarize(runs)}
        assert rows["gru"]["n_seeds"] == 2 and abs(rows["gru"]["val"][0] - 0.92) < 1e-9
        assert abs(rows["gru"]["test_x"][0] - 0.46) < 1e-9
        assert math.isnan(rows["mlp"]["val"][1])
        md = to_markdown(list(rows.values()), "exact", per_bit(runs))
        assert "| gru | 2 |" in md and "bit5" in md and "(n=1)" in md


def test_empty_root_gives_no_runs():
    with tempfile.TemporaryDirectory() as d:
        assert load_runs(d) == {}
