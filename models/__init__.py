"""Fabryka modeli. Wspólny interfejs: model(screens, legs) -> logity (B, T, 6)."""
from __future__ import annotations

import numpy as np

from graphs import CONTROLS, GraphSpec, apply_control, make_random_layered

from .baselines import LinearModel, MLPModel, RecurrentModel, TinyTransformer
from .rate import RateModel

MODEL_NAMES = ("linear", "mlp", "elman", "gru", "lstm", "transformer", "rate")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_graph(gcfg: dict) -> GraphSpec:
    """Graf z configu: kind = random_layered (na razie jedyny; connectome dojdzie w kroku 7) + control."""
    gcfg = dict(gcfg or {})
    kind = gcfg.pop("kind", "random_layered")
    control = gcfg.pop("control", "none")
    control_seed = gcfg.pop("control_seed", 0)
    if kind != "random_layered":
        raise NotImplementedError("wczytywanie connectome dojdzie w kroku 7 (graphs/connectome.py)")
    g = make_random_layered(**gcfg)
    return apply_control(g, control, seed=control_seed)


def build_model(name: str, hidden: int = 128, graph: GraphSpec | None = None, **kw):
    if name == "linear":
        return LinearModel()
    if name == "mlp":
        return MLPModel(hidden=hidden, **kw)
    if name in ("elman", "gru", "lstm"):
        return RecurrentModel(kind=name, hidden=hidden)
    if name == "transformer":
        return TinyTransformer(hidden=hidden, **kw)
    if name == "rate":
        if graph is None:
            raise ValueError("model rate wymaga grafu (sekcja model.graph w configu)")
        return RateModel(graph, **kw)
    raise ValueError(f"model musi być z {MODEL_NAMES}")


def build_model_from_cfg(cfg: dict):
    m = dict(cfg["model"])
    name = m.pop("name")
    graph_cfg = m.pop("graph", None)
    graph = build_graph(graph_cfg) if name == "rate" else None
    return build_model(name, graph=graph, **m)


def match_hidden(name: str, budget: int, candidates=range(2, 1025), **kw) -> tuple[int, int]:
    """Krok 16: rozmiar `hidden`, przy którym model `name` ma liczbę parametrów najbliższą `budget`.

    Zwraca (hidden, liczba_parametrów).
    """
    best = None
    for h in candidates:
        n = count_params(build_model(name, hidden=h, **kw))
        if best is None or abs(n - budget) < abs(best[1] - budget):
            best = (h, n)
        if n > budget * 1.5:
            break
    return best
