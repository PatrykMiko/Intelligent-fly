"""Testy modeli (wymagają PyTorcha). Uruchom: pytest tests/test_models.py"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graphs import make_random_layered
from models import MODEL_NAMES, build_model, count_params, match_hidden
from models.rate import RateModel
from task import Splits, TaskConfig, make_batch


def _batch(n=4, T=5):
    cfg = TaskConfig(seq_len=T)
    b = make_batch(Splits(cfg), "train", n, cfg, np.random.default_rng(0))
    return torch.as_tensor(b["screens"]), torch.as_tensor(b["legs_in"]), torch.as_tensor(b["targets"])


def _small_graph():
    return make_random_layered(seed=0, n_input=32, n_sensor=12, n_hidden=64, motor_per_leg=4)


@pytest.mark.parametrize("name", [m for m in MODEL_NAMES if m != "rate"])
def test_baseline_shapes_and_grads(name):
    s, l, y = _batch()
    m = build_model(name, hidden=32)
    out = m(s, l)
    assert out.shape == (4, 5, 6)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_rate_forward_backward():
    s, l, y = _batch()
    m = RateModel(_small_graph(), inner_steps=4)
    out = m(s, l)
    assert out.shape == (4, 5, 6) and torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy_with_logits(out, y).backward()
    assert m.W_in.weight.grad.abs().sum() > 0
    assert m.v.grad.abs().sum() > 0


def test_rate_graph_is_frozen():
    m = RateModel(_small_graph())
    names = {n for n, _ in m.named_parameters()}
    assert not any("w_val" in n or "w_idx" in n for n in names)     # struktura nie jest parametrem
    assert {"gain", "bias", "alpha_logit", "v", "b_out", "W_in.weight", "W_sens.weight"} <= names


def test_rate_is_causal_in_time():
    s, l, _ = _batch(T=6)
    m = RateModel(_small_graph(), inner_steps=3).eval()
    a = m(s, l)
    s2 = s.clone(); s2[:, 4:] += 1.0                                # zmiana przyszłości...
    b = m(s2, l)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)            # ...nie zmienia przeszłości


def test_transformer_is_causal():
    s, l, _ = _batch(T=6)
    m = build_model("transformer", hidden=32).eval()
    a = m(s, l)
    s2 = s.clone(); s2[:, 4:] += 1.0
    assert torch.allclose(a[:, :4], m(s2, l)[:, :4], atol=1e-5)


def test_param_matching():
    budget = count_params(RateModel(_small_graph()))
    for name in ("mlp", "elman", "gru", "lstm", "transformer"):
        h, n = match_hidden(name, budget)
        assert abs(n - budget) / budget < 0.15, (name, h, n, budget)


def test_linear_model_cannot_beat_chance_on_xor_like_task():
    """Wyrocznia z planu: model liniowy na kroku nie rozwiąże XOR-owego zadania (strata ~ ln2)."""
    cfg = TaskConfig(seq_len=4)
    sp = Splits(cfg)
    rng = np.random.default_rng(0)
    m = build_model("linear")
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(300):
        b = make_batch(sp, "train", 128, cfg, rng)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            m(torch.as_tensor(b["screens"]), torch.as_tensor(b["legs_in"])), torch.as_tensor(b["targets"]))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() > 0.66


def test_rate_input_scale_scales_initial_input_weights():
    m1 = RateModel(_small_graph(), input_scale=1.0)
    m3 = RateModel(_small_graph(), input_scale=3.0)
    assert m3.W_in.weight.abs().mean() > 2 * m1.W_in.weight.abs().mean()


def test_rate_reset_each_step_makes_steps_independent():
    s, l, _ = _batch(T=5)
    m = RateModel(_small_graph(), inner_steps=3, reset_each_step=True).eval()
    full = m(s, l)
    for t in range(5):                                              # krok t sam = krok t w całym epizodzie
        alone = m(s[:, t:t + 1], l[:, t:t + 1])[:, 0]
        assert torch.allclose(full[:, t], alone, atol=1e-5)
    s2 = s.clone(); s2[:, :3] += 1.0                                # zmiana wczesnych ekranów...
    assert torch.allclose(full[:, 3:], m(s2, l)[:, 3:], atol=1e-5)  # ...nie zmienia późniejszych wyjść


def test_build_optimizer_weight_decay_groups():
    from train.train import build_optimizer
    m = RateModel(_small_graph())
    assert isinstance(build_optimizer(m, {"lr": 1e-3}), torch.optim.Adam)           # wd = 0: zwykły Adam
    opt = build_optimizer(m, {"lr": 1e-3, "weight_decay": 0.1, "wd_exclude": ["gain", "bias", "alpha_logit", "b_out"]})
    assert isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 2
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    plain = {id(p) for p in opt.param_groups[1]["params"]}
    assert opt.param_groups[0]["weight_decay"] == 0.1 and opt.param_groups[1]["weight_decay"] == 0.0
    assert id(m.W_in.weight) in decayed and id(m.v) in decayed and id(m.gain) in plain and id(m.bias) in plain
    opt2 = build_optimizer(m, {"lr": 1e-3, "weight_decay": 0.1})                    # bez wykluczeń: jedna grupa
    assert len(opt2.param_groups) == 1


def test_eval_checkpoints_roundtrip(tmp_path):
    from scripts.eval_checkpoints import eval_run
    from models import build_model_from_cfg
    cfg = {"name": "t", "task": {"seq_len": 4}, "model": {"name": "gru", "hidden": 8},
           "train": {"steps": 1, "eval_episodes": 16}}
    model = build_model_from_cfg(cfg)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": 7}, tmp_path / "best.pt")
    r = eval_run(str(tmp_path), "best.pt", device=torch.device("cpu"))
    assert r["step"] == 7 and r["gain"] is None
    assert all(0.0 <= r[s] <= 1.0 for s in ("val", "test_pairs", "test_x", "test_states"))
