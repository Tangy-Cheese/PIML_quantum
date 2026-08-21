"""The k-eliminated PINN and its trainer -- shared by every published result.

k is not a network output. It is reconstructed in closed form from
(theta_s, theta_r) via the return condition (report.tex Eq. k-elimination),
so the return condition is satisfied identically and only two residuals --
tunnelling and rescattering -- enter the loss. This is the architecture of
Fig. 6 and Eq. loss-k-eliminated-full, and the loss report.tex states was
"used to produce every result in Secs. 5.2-6".

An earlier generation of scripts instead emitted k as a free sixth network
output and trained on three residuals (the unreduced system,
Eq. loss-unreduced-full). Those scripts have been removed; everything under
scripts/figures/ uses the module below, so the code matches the claim. The
only surviving copies of the free-k trainer are vendored under
future_work/pmd/legacy/ as dependencies of unpublished work.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class PINN(nn.Module):
    """input_dim=1: p_par only (1D slice, p_perp=0 implicitly).
    input_dim=2: (p_par, p_perp).
    unbounded=True is the "baseline" ablation config
    (raw*10+centre, no tanh window) for a like-for-like ablation comparison.
    """

    def __init__(self, centres, halfwidths, input_dim=1, hidden=64, depth=3,
                 unbounded=False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]
        self.net = nn.Sequential(*layers)
        self.unbounded = unbounded
        self.register_buffer("c", torch.tensor(centres))
        self.register_buffer("w", torch.tensor(halfwidths))

    def forward(self, z):
        raw = self.net(z)
        o = raw * 10.0 + self.c if self.unbounded else self.c + self.w * torch.tanh(raw)
        return o[:, 0] + 1j * o[:, 1], o[:, 2] + 1j * o[:, 3]


def residuals(th_s, th_r, p_par, p_perp, A_0, I_P):
    """Closed-form k (return condition solved analytically); two residuals,
    tunnelling and rescattering, matching continuation.py's residual_c."""
    k = -A_0 * (torch.sin(th_r) - torch.sin(th_s)) / (th_r - th_s)
    A_s = A_0 * torch.cos(th_s)
    A_r = A_0 * torch.cos(th_r)
    R1 = 0.5 * (k + A_s) ** 2 + I_P
    R2 = 0.5 * (p_par + A_r) ** 2 + 0.5 * p_perp ** 2 - 0.5 * (k + A_r) ** 2
    return R1, R2


def residuals_bichrom(th_s, th_r, p_par, p_perp, A_1, A_2, phi, I_P):
    """Two-harmonic generalisation of residuals(), for

        A(th) = A_1 cos(th) + A_2 cos(2 th + phi).

    The return condition still eliminates k in closed form, but through the
    antiderivative of the full two-harmonic field rather than A_1 sin(th)
    alone (cf. bichromatic.int_A):

        k = -[A_1 (sin th_r - sin th_s)
              + (A_2/2)(sin(2 th_r + phi) - sin(2 th_s + phi))] / (th_r - th_s)

    Reduces exactly to residuals() as A_2 -> 0.
    """
    intA_s = A_1 * torch.sin(th_s) + 0.5 * A_2 * torch.sin(2 * th_s + phi)
    intA_r = A_1 * torch.sin(th_r) + 0.5 * A_2 * torch.sin(2 * th_r + phi)
    k = -(intA_r - intA_s) / (th_r - th_s)
    A_s = A_1 * torch.cos(th_s) + A_2 * torch.cos(2 * th_s + phi)
    A_r = A_1 * torch.cos(th_r) + A_2 * torch.cos(2 * th_r + phi)
    R1 = 0.5 * (k + A_s) ** 2 + I_P
    R2 = 0.5 * (p_par + A_r) ** 2 + 0.5 * p_perp ** 2 - 0.5 * (k + A_r) ** 2
    return R1, R2


def train(model, z, p_par, p_perp, A_0, I_P, epochs, lr=2e-3, patience=300):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience)
    best, state = np.inf, None
    for _ in range(epochs):
        opt.zero_grad()
        th_s, th_r = model(z)
        R1, R2 = residuals(th_s, th_r, p_par, p_perp, A_0, I_P)
        L = (R1.abs() ** 2).mean() + (R2.abs() ** 2).mean()
        L.backward()
        opt.step()
        sch.step(L.item())
        if L.item() < best:
            best = L.item()
            state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best


def train_bichrom(model, z, p_par, p_perp, A_1, A_2, phi, I_P, epochs,
                  lr=2e-3, patience=300):
    """As train(), but with the two-harmonic k-elimination. Identical
    optimiser, schedule and best-checkpoint protocol."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience)
    best, state = np.inf, None
    for _ in range(epochs):
        opt.zero_grad()
        th_s, th_r = model(z)
        R1, R2 = residuals_bichrom(th_s, th_r, p_par, p_perp, A_1, A_2, phi, I_P)
        L = (R1.abs() ** 2).mean() + (R2.abs() ** 2).mean()
        L.backward()
        opt.step()
        sch.step(L.item())
        if L.item() < best:
            best = L.item()
            state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best
