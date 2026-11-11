"""Mini-eksperyment z planu (tydzień 3): model liniowy nie zejdzie poniżej ln2 = 0,693 na XOR, MLP rozwiąże.

    python -m experiments.xor_demo
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).repeat(64, 1)
Y = (X[:, :1] != X[:, 1:]).float()                        # XOR
print(f"strata zgadywania = ln2 = {np.log(2):.4f}")
for name, net in {"liniowy": nn.Linear(2, 1),
                  "MLP(2-8-1)": nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))}.items():
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(1500):
        loss = nn.functional.binary_cross_entropy_with_logits(net(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((net(X) > 0).float() == Y).float().mean().item()
    print(f"{name:12s} strata końcowa = {loss.item():.4f}   dokładność = {acc:.2f}")
