"""Modele odniesienia. Wspólny interfejs: forward(screens (B,T,64), legs (B,T,6)) -> logity (B,T,6)."""
from __future__ import annotations

import torch
import torch.nn as nn

N_SCREEN, N_LEGS, N_OUT = 64, 6, 6
N_IN = N_SCREEN + N_LEGS


class LinearModel(nn.Module):
    """Model liniowy na kroku (bez pamięci). Spodziewane: strata ~0,693 (XOR nie jest liniowo separowalny)."""

    def __init__(self, hidden: int = 0):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)

    def forward(self, screens, legs):
        return self.fc(torch.cat([screens, legs], dim=-1))


class MLPModel(nn.Module):
    """MLP na kroku (bez pamięci): wejście = ekran + położenie nóg."""

    def __init__(self, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers, d = [], N_IN
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, N_OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, screens, legs):
        return self.net(torch.cat([screens, legs], dim=-1))


class RecurrentModel(nn.Module):
    """Elman RNN (tanh) / GRU / LSTM + liniowa głowica."""

    def __init__(self, kind: str = "gru", hidden: int = 128):
        super().__init__()
        if kind == "elman":
            self.rnn = nn.RNN(N_IN, hidden, batch_first=True, nonlinearity="tanh")
        elif kind == "gru":
            self.rnn = nn.GRU(N_IN, hidden, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(N_IN, hidden, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden, N_OUT)

    def forward(self, screens, legs):
        out, _ = self.rnn(torch.cat([screens, legs], dim=-1))
        return self.head(out)


class TinyTransformer(nn.Module):
    """Mały Transformer z maską przyczynową po osi czasu (kroków instrukcji). hidden = d_model (zaokrąglane do wielokrotności 4)."""

    def __init__(self, hidden: int = 64, n_layers: int = 2, n_heads: int = 4, max_len: int = 64):
        super().__init__()
        d = max(n_heads, n_heads * round(hidden / n_heads))
        self.inp = nn.Linear(N_IN, d)
        self.pos = nn.Parameter(torch.zeros(max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, N_OUT)

    def forward(self, screens, legs):
        x = self.inp(torch.cat([screens, legs], dim=-1))
        T = x.shape[1]
        x = x + self.pos[:T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.head(self.enc(x, mask=mask))
