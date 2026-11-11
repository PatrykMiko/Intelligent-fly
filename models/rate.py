"""Model `Rate`: sieć rekurencyjna na ZAMROŻONYM grafie (connectome lub graf kontrolny).

Zamrożone: topologia, znaki i moduły wag połączeń (GraphSpec).
Uczone (RQ1 - "tylko parametry wejścia, wyjścia i dynamiki"):
  W_in       ekran (64 piksele) -> neurony wejściowe
  W_sens     położenie nóg (6 bitów) -> neurony czucia
  v, b_out   odczyt: logit nogi i = sum_j v_j * h_j (po neuronach ruchowych nogi i) + b_i
  alpha      stała "przeciekania" każdego neuronu (sigmoid)
  bias       próg każdego neuronu
  gain       jeden globalny skalar wzmocnienia wag synaptycznych
  reset_each_step=True zeruje stan sieci przed każdą instrukcją (dla legs_mode=teacher pamięć nie jest potrzebna;
  bez resetu stan nasyca się i 'zatrzaskuje' - zob. docs/eksperymenty.md, E7-E8).
  (input_scale mnoży tylko POCZĄTKOWE wagi W_in i W_sens - parametr diagnostyczny, domyślnie 1.0)

Dynamika w jednym kroku zadania (instrukcji): ekran i czucie są trzymane stałe, a sieć wykonuje
`inner_steps` kroków wewnętrznych:
    h <- (1 - alpha) * h + alpha * act(gain * W h + u)
Stan h przechodzi między krokami zadania (to jest "pamięć" sieci).

Uwaga o wydajności: macierz rzadka budowana jest w każdym forward (tanie dla ~1e5 krawędzi); dla pełnego
connectome (~2,5e7 krawędzi) trzeba będzie zbudować ją raz i trzymać na GPU (TODO).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from graphs import GraphSpec

N_SCREEN, N_LEGS = 64, 6


class RateModel(nn.Module):
    def __init__(self, graph: GraphSpec, inner_steps: int = 8, act: str = "tanh", init_gain: float = 8.0,
                 input_scale: float = 1.0, reset_each_step: bool = False):
        super().__init__()
        self.n, self.K = graph.n, inner_steps
        self.reset_each_step = reset_each_step                          # True: stan h zerowany przed każdą instrukcją
        self.register_buffer("w_idx", torch.as_tensor(np.stack([graph.dst, graph.src]), dtype=torch.long), persistent=False)
        self.register_buffer("w_val", torch.as_tensor(graph.sign * graph.mag, dtype=torch.float32), persistent=False)
        self.register_buffer("in_idx", torch.as_tensor(graph.input_idx, dtype=torch.long), persistent=False)
        self.register_buffer("sens_idx", torch.as_tensor(graph.sensor_idx, dtype=torch.long), persistent=False)
        motor_all = np.concatenate(graph.motor_idx)
        motor_leg = np.concatenate([np.full(len(a), i) for i, a in enumerate(graph.motor_idx)])
        self.register_buffer("motor_all", torch.as_tensor(motor_all, dtype=torch.long), persistent=False)
        self.register_buffer("motor_leg", torch.as_tensor(motor_leg, dtype=torch.long), persistent=False)

        self.W_in = nn.Linear(N_SCREEN, len(graph.input_idx), bias=False)
        self.W_sens = nn.Linear(N_LEGS, len(graph.sensor_idx), bias=False)
        with torch.no_grad():                                          # skala początkowa wag wejścia (diagnostyka)
            self.W_in.weight.mul_(input_scale)
            self.W_sens.weight.mul_(input_scale)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.alpha_logit = nn.Parameter(torch.zeros(self.n))          # alpha = 0.5 na starcie
        self.gain = nn.Parameter(torch.tensor(float(init_gain)))
        per_leg = max(1, len(graph.motor_idx[0]))
        self.v = nn.Parameter(torch.randn(len(motor_all)) / per_leg ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(N_LEGS))
        self.act = {"tanh": torch.tanh, "relu": torch.relu, "softplus": nn.functional.softplus}[act]

    def forward(self, screens: torch.Tensor, legs: torch.Tensor) -> torch.Tensor:
        B, T, _ = screens.shape
        W = torch.sparse_coo_tensor(self.w_idx, self.w_val, (self.n, self.n),
                                    check_invariants=False).coalesce()
        alpha = torch.sigmoid(self.alpha_logit)
        u_in, u_sens = self.W_in(screens), self.W_sens(legs)           # (B,T,n_in), (B,T,n_sens)
        h = screens.new_zeros(B, self.n)
        outs = []
        for t in range(T):
            if self.reset_each_step:                                    # model bezstanowy między instrukcjami
                h = screens.new_zeros(B, self.n)
            u = screens.new_zeros(B, self.n).index_add(1, self.in_idx, u_in[:, t])
            u = u.index_add(1, self.sens_idx, u_sens[:, t]) + self.bias
            for _ in range(self.K):
                rec = torch.sparse.mm(W, h.t()).t()                     # (B, n) = (W @ h^T)^T
                h = (1 - alpha) * h + alpha * self.act(self.gain * rec + u)
            motor = h[:, self.motor_all] * self.v
            outs.append(screens.new_zeros(B, N_LEGS).index_add(1, self.motor_leg, motor) + self.b_out)
        return torch.stack(outs, dim=1)
