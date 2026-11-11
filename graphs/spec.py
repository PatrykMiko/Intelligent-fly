"""Graf połączeń (connectome lub graf zastępczy) + grafy kontrolne (RQ2). Tylko numpy.

GraphSpec trzyma krawędzie jako listę (src -> dst), znak (+1 pobudzające / -1 hamujące) i moduł wagi.
Znak jest cechą NEURONU PRESYNAPTYCZNEGO (neuroprzekaźnik; prawo Dale'a), więc wszystkie krawędzie
wychodzące z jednego neuronu mają ten sam znak. Moduły wag normalizujemy tak, by suma modułów
wag wchodzących do każdego neuronu wynosiła 1.

Role neuronów:
  input_idx  - neurony, do których trafia ekran (64 piksele -> uczona macierz)
  sensor_idx - neurony czucia położenia nóg (6 bitów -> uczona macierz)
  motor_idx  - lista 6 tablic: neurony ruchowe każdej nogi (odczyt bitu nogi i)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphSpec:
    n: int
    src: np.ndarray
    dst: np.ndarray
    sign: np.ndarray            # float32, +1 / -1 na krawędź
    mag: np.ndarray             # float32, > 0
    input_idx: np.ndarray
    sensor_idx: np.ndarray
    motor_idx: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    def copy(self, **changes) -> "GraphSpec":
        d = dict(n=self.n, src=self.src.copy(), dst=self.dst.copy(), sign=self.sign.copy(), mag=self.mag.copy(),
                 input_idx=self.input_idx, sensor_idx=self.sensor_idx, motor_idx=list(self.motor_idx),
                 meta=dict(self.meta))
        d.update(changes)
        return GraphSpec(**d)

    def normalize(self) -> "GraphSpec":
        """Suma modułów wag wchodzących do każdego neuronu = 1 (neurony bez wejść bez zmian)."""
        tot = np.zeros(self.n, dtype=np.float64)
        np.add.at(tot, self.dst, self.mag)
        denom = np.where(tot[self.dst] > 0, tot[self.dst], 1.0)
        self.mag = (self.mag / denom).astype(np.float32)
        return self

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.dst, minlength=self.n)

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.src, minlength=self.n)

    def inhibitory_fraction(self) -> float:
        return float((self.sign < 0).mean())

    def save(self, path) -> None:
        np.savez_compressed(path, n=self.n, src=self.src, dst=self.dst, sign=self.sign, mag=self.mag,
                            input_idx=self.input_idx, sensor_idx=self.sensor_idx,
                            motor_idx=np.array(self.motor_idx, dtype=object), allow_pickle=True)

    @staticmethod
    def load(path) -> "GraphSpec":
        z = np.load(path, allow_pickle=True)
        return GraphSpec(int(z["n"]), z["src"], z["dst"], z["sign"], z["mag"], z["input_idx"], z["sensor_idx"],
                         list(z["motor_idx"]))


# ------------------------------------------------------------ graf zastępczy (mały)
def _random_edges(rng, src_ids, dst_ids, p):
    """Losowe krawędzie src_ids -> dst_ids; każda para z prawdopodobieństwem p."""
    total = len(src_ids) * len(dst_ids)
    k = rng.binomial(total, p)
    flat = rng.choice(total, size=k, replace=False)
    return src_ids[flat // len(dst_ids)], dst_ids[flat % len(dst_ids)]


def make_random_layered(seed: int = 0, n_input: int = 256, n_sensor: int = 96, n_hidden: int = 1024,
                        n_hidden_layers: int = 3, motor_per_leg: int = 32, p_ff: float = 0.02,
                        p_rec: float = 0.01, p_fb: float = 0.01, inhib_frac: float = 0.2) -> GraphSpec:
    """Mały losowy graf warstwowy do szybkich eksperymentów (kilka tysięcy neuronów).

    Warstwy: [wejście+czucie] -> H1 -> ... -> Hk -> [ruchowe (6 x motor_per_leg)]
    Krawędzie: w przód między sąsiednimi warstwami (p_ff), rekurencyjne wewnątrz każdej ukrytej (p_rec)
    i sprzężenie zwrotne z ostatniej ukrytej do pierwszej (p_fb) - pętle dają sieci "pamięć".
    """
    rng = np.random.default_rng(seed)
    sizes = [n_input + n_sensor] + [n_hidden] * n_hidden_layers + [6 * motor_per_leg]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    layers = [np.arange(offs[i], offs[i + 1]) for i in range(len(sizes))]
    n = int(offs[-1])

    S, D = [], []
    def add(a, b, p):
        s, d = _random_edges(rng, a, b, p)
        S.append(s); D.append(d)
    for i in range(len(layers) - 1):
        add(layers[i], layers[i + 1], p_ff)
    for h in layers[1:-1]:
        add(h, h, p_rec)
    add(layers[-2], layers[1], p_fb)
    src, dst = np.concatenate(S), np.concatenate(D)

    neuron_sign = np.where(rng.random(n) < inhib_frac, -1.0, 1.0).astype(np.float32)
    mag = rng.lognormal(0.0, 0.5, size=src.size).astype(np.float32)
    g = GraphSpec(
        n=n, src=src, dst=dst, sign=neuron_sign[src], mag=mag,
        input_idx=layers[0][:n_input], sensor_idx=layers[0][n_input:],
        motor_idx=[layers[-1][i * motor_per_leg:(i + 1) * motor_per_leg] for i in range(6)],
        meta={"kind": "random_layered", "seed": seed},
    )
    return g.normalize()


# ------------------------------------------------------------------ grafy kontrolne
def control_degree_preserving(g: GraphSpec, rng) -> GraphSpec:
    """Losowy graf z tymi samymi stopniami: permutacja celów krawędzi (src bez zmian).

    Zachowuje stopień wyjściowy i wejściowy każdego neuronu oraz znaki (zależą od src).
    Mogą pojawić się pętle własne i krawędzie wielokrotne - akceptujemy (sumują się w macierzy rzadkiej).
    """
    out = g.copy(dst=g.dst[rng.permutation(g.n_edges)])
    out.meta["control"] = "degree_preserving"
    return out.normalize()


def control_shuffled_weights(g: GraphSpec, rng) -> GraphSpec:
    """Przetasowane wagi: topologia bez zmian, moduły i znaki rozlosowane między krawędziami."""
    perm = rng.permutation(g.n_edges)
    out = g.copy(sign=g.sign[perm], mag=g.mag[perm])
    out.meta["control"] = "shuffled_weights"
    return out.normalize()


def control_no_signs(g: GraphSpec) -> GraphSpec:
    """Bez znaków: wszystkie połączenia pobudzające."""
    out = g.copy(sign=np.ones_like(g.sign))
    out.meta["control"] = "no_signs"
    return out.normalize()


def control_random_signs(g: GraphSpec, rng) -> GraphSpec:
    """Losowe znaki: znak losowany dla neuronu presynaptycznego, z tą samą frakcją hamujących co w grafie."""
    p = g.inhibitory_fraction()
    neuron_sign = np.where(rng.random(g.n) < p, -1.0, 1.0).astype(np.float32)
    out = g.copy(sign=neuron_sign[g.src])
    out.meta["control"] = "random_signs"
    return out.normalize()


CONTROLS = ("none", "degree_preserving", "shuffled_weights", "no_signs", "random_signs")


def apply_control(g: GraphSpec, kind: str, seed: int = 0) -> GraphSpec:
    rng = np.random.default_rng(seed)
    if kind == "none":
        return g
    if kind == "degree_preserving":
        return control_degree_preserving(g, rng)
    if kind == "shuffled_weights":
        return control_shuffled_weights(g, rng)
    if kind == "no_signs":
        return control_no_signs(g)
    if kind == "random_signs":
        return control_random_signs(g, rng)
    raise ValueError(f"control musi być z {CONTROLS}")


def spectral_radius(g: GraphSpec, iters: int = 300, seed: int = 0) -> float:
    """Szacunek promienia spektralnego macierzy W = znak * moduł wagi (wzór Gelfanda, iteracja potęgowa).

    Działa na liście krawędzi (bincount), więc nadaje się też do pełnego connectome (25,6 mln krawędzi).
    Wzmocnienie `gain` jest "krytyczne" w okolicy 1 / rho: powyżej niego pętle zwrotne podtrzymują własną aktywność.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(g.n)
    x /= np.linalg.norm(x)
    w = (g.sign * g.mag).astype(np.float64)
    log_norm, hist = 0.0, []
    for _ in range(iters):
        y = np.bincount(g.dst, weights=w * x[g.src], minlength=g.n)
        nrm = float(np.linalg.norm(y))
        if nrm == 0.0:
            return 0.0
        log_norm += np.log(nrm)
        x = y / nrm
        hist.append(log_norm)
    half = iters // 2
    return float(np.exp((hist[-1] - hist[half - 1]) / (iters - half)))
