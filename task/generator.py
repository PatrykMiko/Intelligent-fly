"""Generator zadania „Mucha, która liczy”.

Stan = liczba 0..63 zapisana na 6 nogach (bit i = noga i, wartość 2**i).
Instrukcja = +X albo -X, nowy stan = (stan ± X) mod 64.
Ekran LED 8x8 (64 piksele) koduje instrukcję.

Moduł celowo używa TYLKO numpy (bez PyTorcha) - można go importować wszędzie.

Najważniejsze elementy:
  to_bits / from_bits / apply_op / carries   - arytmetyka i przeniesienia
  encode_screen                              - kodowanie ekranu LED
  TaskConfig, Splits                         - konfiguracja i podziały danych
  make_batch                                 - losowanie epizodów z danego podziału
  episode_to_dict / write_episode_json       - eksport epizodu do JSON (dla odtwarzacza w C++)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

N_BITS = 6
MOD = 64
N_PIXELS = 64
ADD, SUB = 1, -1

# "Ruch" m = (operacja, X). m = 0..63 -> +X, m = 64..127 -> -X.
N_MOVES = 128
MOVE_OP = np.where(np.arange(N_MOVES) < MOD, ADD, SUB).astype(np.int64)
MOVE_X = (np.arange(N_MOVES) % MOD).astype(np.int64)

SPLITS = ("train", "val", "test_pairs", "test_x", "test_states")
SCREEN_ENCODINGS = ("onehot_signed", "binary")
LEGS_MODES = ("teacher", "init_only")


# ----------------------------------------------------------------- arytmetyka
def to_bits(n) -> np.ndarray:
    """Liczba(y) 0..63 -> (..., 6) bity; bit i = noga i."""
    n = np.asarray(n, dtype=np.int64)
    return ((n[..., None] >> np.arange(N_BITS)) & 1).astype(np.uint8)


def from_bits(bits) -> np.ndarray:
    """(..., 6) bity -> liczba(y) 0..63."""
    bits = np.asarray(bits).astype(np.int64)
    return (bits * (1 << np.arange(N_BITS))).sum(-1)


def apply_op(state, op, x):
    """(stan ± X) mod 64; op = +1 (dodawanie) albo -1 (odejmowanie)."""
    return (np.asarray(state, dtype=np.int64) + np.asarray(op, dtype=np.int64) * np.asarray(x, dtype=np.int64)) % MOD


def carries(state, op, x) -> np.ndarray:
    """Przeniesienie (dodawanie) / pożyczka (odejmowanie) WEJŚCIOWE dla każdego bitu.

    Zwraca (..., 6): c[..., i] to przeniesienie z niższych bitów do bitu i (c[...,0] = 0).
    Bit wyniku i = s_i XOR x_i XOR c_i (dla obu operacji).
    """
    s = to_bits(state).astype(np.int64)
    xb = to_bits(x).astype(np.int64)
    op = np.asarray(op)
    c = np.zeros(np.broadcast_shapes(s.shape, xb.shape), dtype=np.int64)
    cur = np.zeros(c.shape[:-1], dtype=np.int64)
    for i in range(N_BITS):
        c[..., i] = cur
        si, xi = s[..., i], xb[..., i]
        add_next = (si & xi) | (si & cur) | (xi & cur)
        sub_next = ((1 - si) & xi) | ((1 - (si ^ xi)) & cur)
        cur = np.where(op == ADD, add_next, sub_next)
    return c.astype(np.uint8)


def all_triples():
    """Wszystkie 64*64*2 = 8192 trójki (stan, op, x) - do testów wyczerpujących."""
    s, x, o = np.meshgrid(np.arange(MOD), np.arange(MOD), np.array([ADD, SUB]), indexing="ij")
    return s.ravel(), o.ravel(), x.ravel()


# ---------------------------------------------------------------------- ekran
def encode_screen(op, x, encoding: str = "onehot_signed") -> np.ndarray:
    """Instrukcja -> ekran 8x8 jako (..., 64) float32.

    onehot_signed: zapalony jest piksel o indeksie X; wartość +1 (dodawanie) albo -1 (odejmowanie).
    binary:        piksele 0..5 = bity X, piksel 6 = 1 dla odejmowania, reszta 0.
    """
    if encoding not in SCREEN_ENCODINGS:
        raise ValueError(f"screen_encoding musi być z {SCREEN_ENCODINGS}")
    ob, xb = np.broadcast_arrays(np.asarray(op, dtype=np.int64), np.asarray(x, dtype=np.int64))
    scr = np.zeros(ob.shape + (N_PIXELS,), dtype=np.float32)
    if encoding == "onehot_signed":
        np.put_along_axis(scr, xb[..., None], ob[..., None].astype(np.float32), axis=-1)
    else:
        scr[..., :N_BITS] = to_bits(xb)
        scr[..., N_BITS] = (ob == SUB)
    return scr


# ------------------------------------------------------------------ konfiguracja
@dataclass
class TaskConfig:
    seq_len: int = 16                    # liczba instrukcji w epizodzie
    screen_encoding: str = "onehot_signed"
    screen_noise: float = 0.0            # odch. std. szumu gaussowskiego na ekranie (RQ7)
    legs_mode: str = "teacher"           # teacher: model widzi prawdziwy poprzedni stan nóg w każdym kroku
                                         # init_only: stan nóg widoczny tylko w kroku 0, potem model musi go pamiętać
    held_x: int = 6                      # ile wartości X całkowicie wyłączamy z treningu
    held_states: int = 6                 # ile stanów całkowicie wyłączamy z treningu
    val_frac: float = 0.05               # frakcja wolnych par (stan, X) na walidację
    test_frac: float = 0.10              # frakcja wolnych par (stan, X) na test "niewidziane pary"
    split_seed: int = 0

    def __post_init__(self):
        if self.screen_encoding not in SCREEN_ENCODINGS:
            raise ValueError(f"screen_encoding musi być z {SCREEN_ENCODINGS}")
        if self.legs_mode not in LEGS_MODES:
            raise ValueError(f"legs_mode musi być z {LEGS_MODES}")


class Splits:
    """Podziały danych (RQ4). Para = (stan, X); dotyczy obu operacji (+X i -X).

    train        : stan, X i para widziane; stan docelowy nie jest wyłączonym stanem
    val          : niewidziane pary (do doboru hiperparametrów)
    test_pairs   : niewidziane pary (stan i X widziane osobno, ale nie razem)
    test_x       : niewidziane wartości X (wszystkie pary z tymi X)
    test_states  : niewidziane stany (poprzedni stan nigdy nie występował w treningu)

    criteria[nazwa] to macierz bool (64 stany, 128 ruchów): czy dany ruch ze stanu
    należy do podziału. W podziałach ewaluacyjnych liczą się tylko kroki spełniające kryterium
    (maska `measure`); pozostałe kroki epizodu to losowy "wypełniacz" łączący łańcuch stanów.
    """

    def __init__(self, cfg: TaskConfig):
        rng = np.random.default_rng(cfg.split_seed)
        self.held_x = np.sort(rng.choice(MOD, cfg.held_x, replace=False))
        self.held_states = np.sort(rng.choice(MOD, cfg.held_states, replace=False))
        states = np.arange(MOD)
        self.next_state = (states[:, None] + MOVE_OP[None, :] * MOVE_X[None, :]) % MOD  # (64, 128)

        hs_row = np.isin(states, self.held_states)
        hx_col = np.isin(np.arange(MOD), self.held_x)
        free = ~hs_row[:, None] & ~hx_col[None, :]               # pary (stan, X) bez wyłączonych
        free_idx = np.flatnonzero(free.ravel())
        rng.shuffle(free_idx)
        n_val = int(round(cfg.val_frac * free_idx.size))
        n_test = int(round(cfg.test_frac * free_idx.size))
        pair_val = np.zeros(MOD * MOD, bool)
        pair_test = np.zeros(MOD * MOD, bool)
        pair_val[free_idx[:n_val]] = True
        pair_test[free_idx[n_val:n_val + n_test]] = True
        pair_val, pair_test = pair_val.reshape(MOD, MOD), pair_test.reshape(MOD, MOD)
        pair_train = free & ~pair_val & ~pair_test
        self.pair_train, self.pair_val, self.pair_test = pair_train, pair_val, pair_test

        to_moves = lambda P: P[:, MOVE_X]                         # (64, 64) -> (64, 128)
        next_held = np.isin(self.next_state, self.held_states)
        self.criteria = {
            "train": to_moves(pair_train) & ~next_held,
            "val": to_moves(pair_val),
            "test_pairs": to_moves(pair_test),
            "test_x": (~hs_row[:, None]) & hx_col[MOVE_X][None, :],
            "test_states": hs_row[:, None] & ~hx_col[MOVE_X][None, :],
        }
        for s in states[~hs_row]:
            if not self.criteria["train"][s].any():
                raise RuntimeError(f"stan {s} nie ma dozwolonych ruchów treningowych - zmniejsz held_*")

    def describe(self) -> dict:
        return {
            "held_x": self.held_x.tolist(),
            "held_states": self.held_states.tolist(),
            "n_ruchow_train": int(self.criteria["train"].sum()),
            **{f"n_ruchow_{k}": int(v.sum()) for k, v in self.criteria.items() if k != "train"},
        }


# ------------------------------------------------------------------ epizody
def make_batch(splits: Splits, split: str, n: int, cfg: TaskConfig, rng: np.random.Generator) -> dict:
    """Losuje n epizodów z danego podziału. Wszystko jako tablice numpy.

    Klucze (n = liczba epizodów, T = seq_len):
      screens (n,T,64) float32   ekran w każdym kroku (z ewentualnym szumem)
      legs_in (n,T,6)  float32   położenie nóg widziane przez model (zależy od legs_mode)
      targets (n,T,6)  float32   bity stanu docelowego
      measure (n,T)    bool      czy krok liczy się do metryk tego podziału
      state_prev, op, x, state_next (n,T) int64
      carries (n,T,6)  uint8     przeniesienia/pożyczki (do analizy błędów)
    """
    if split not in SPLITS:
        raise ValueError(f"split musi być z {SPLITS}")
    crit = splits.criteria[split]
    T = cfg.seq_len
    s = rng.choice(np.flatnonzero(crit.any(axis=1)), size=n)

    prev = np.empty((n, T), np.int64)
    op = np.empty((n, T), np.int64)
    x = np.empty((n, T), np.int64)
    nxt = np.empty((n, T), np.int64)
    measure = np.empty((n, T), bool)
    for t in range(T):
        allowed = crit[s]                                          # (n, 128)
        has = allowed.any(axis=1)
        if split == "train" and not has.all():
            raise RuntimeError("brak dozwolonego ruchu treningowego (nie powinno się zdarzyć)")
        keys = np.where(allowed, rng.random((n, N_MOVES)), -1.0)
        m = np.where(has, keys.argmax(axis=1), rng.integers(0, N_MOVES, size=n))
        prev[:, t], op[:, t], x[:, t] = s, MOVE_OP[m], MOVE_X[m]
        s = splits.next_state[s, m]
        nxt[:, t], measure[:, t] = s, has

    screens = encode_screen(op, x, cfg.screen_encoding)
    if cfg.screen_noise > 0:
        screens = screens + rng.normal(0.0, cfg.screen_noise, screens.shape).astype(np.float32)
    legs = to_bits(prev).astype(np.float32)
    if cfg.legs_mode == "init_only":
        legs[:, 1:] = 0.0
    return {
        "screens": screens,
        "legs_in": legs,
        "targets": to_bits(nxt).astype(np.float32),
        "measure": measure,
        "state_prev": prev, "op": op, "x": x, "state_next": nxt,
        "carries": carries(prev, op, x),
    }


# ------------------------------------------------------------------ eksport JSON
def episode_to_dict(batch: dict, i: int, cfg: TaskConfig, split: str = "train") -> dict:
    """Epizod nr i z batcha -> słownik w formacie opisanym w README (format_version 1)."""
    steps = []
    for t in range(batch["screens"].shape[1]):
        scr = batch["screens"][i, t]
        scr_list = scr.astype(int).tolist() if np.all(scr == np.round(scr)) else np.round(scr, 4).tolist()
        op, x = int(batch["op"][i, t]), int(batch["x"][i, t])
        steps.append({
            "t": t,
            "instruction": f"{'+' if op > 0 else '-'}{x}",
            "op": "+" if op > 0 else "-",
            "x": x,
            "screen": scr_list,                                    # 64 liczb, wiersz po wierszu (8x8)
            "state_before": int(batch["state_prev"][i, t]),
            "legs_before": to_bits(batch["state_prev"][i, t]).tolist(),
            "target_state": int(batch["state_next"][i, t]),
            "target_bits": to_bits(batch["state_next"][i, t]).tolist(),
            "predicted_bits": None,                                # wypełnia scripts/predict_episode.py
            "correct": None,
        })
    return {
        "format_version": 1,
        "meta": {"split": split, "modulus": MOD, "n_bits": N_BITS, "bit_order": "bit i = noga i, wartość 2^i",
                 "task_config": asdict(cfg)},
        "steps": steps,
    }


def write_episode_json(ep: dict, path) -> None:
    """Zapis z jednym krokiem na linię (czytelne diffy, mały plik)."""
    head = {k: v for k, v in ep.items() if k != "steps"}
    lines = ["{"] + [f' "{k}": {json.dumps(v, ensure_ascii=False)},' for k, v in head.items()] + [' "steps": [']
    n = len(ep["steps"])
    lines += ["  " + json.dumps(st) + ("," if j < n - 1 else "") for j, st in enumerate(ep["steps"])]
    lines += [" ]", "}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
