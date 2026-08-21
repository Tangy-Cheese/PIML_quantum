"""Rescattered ATI saddle points in a bichromatic linearly polarised field.

    A(theta) = A_1 cos(theta) + A_2 cos(2 theta + phi)

Adding the second harmonic breaks the half-cycle symmetry E(t + T/2) = -E(t),
so orbit pairs split and the relative phase phi becomes a third real control
parameter alongside (p_par, p_perp).

The analytic elimination of k via the return condition carries over unchanged.
Writing I(theta) for the antiderivative of A, the return condition gives

    k = -[I(theta_r) - I(theta_s)] / (theta_r - theta_s)

and differentiating, using only I' = A,

    dk/dtheta_s = +(A(theta_s) + k) / (theta_r - theta_s)
    dk/dtheta_r = -(A(theta_r) + k) / (theta_r - theta_s)

which is independent of the field. Only A, dA and the antiderivatives change.
"""

import numpy as np

OMEGA = 0.057
I_P = 0.5

A_1 = 0.0534 / 0.057
RATIO = 0.3
PHI = 0.0

U_P = (0.0534**2) / (4 * OMEGA**2)


def A_2():
    return RATIO * A_1


def A(th, phi=None):
    p = PHI if phi is None else phi
    return A_1 * np.cos(th) + A_2() * np.cos(2 * th + p)


def dA(th, phi=None):
    p = PHI if phi is None else phi
    return -A_1 * np.sin(th) - 2 * A_2() * np.sin(2 * th + p)


def int_A(th, phi=None):
    """Antiderivative of A."""
    p = PHI if phi is None else phi
    return A_1 * np.sin(th) + 0.5 * A_2() * np.sin(2 * th + p)


def k_of(th_s, th_r, phi=None):
    return -(int_A(th_r, phi) - int_A(th_s, phi)) / (th_r - th_s)


def residual_c(th_s, th_r, p_par, p_perp, phi=None):
    k = k_of(th_s, th_r, phi)
    R_tun = 0.5 * (k + A(th_s, phi)) ** 2 + I_P
    R_en = (0.5 * (p_par + A(th_r, phi)) ** 2 + 0.5 * p_perp**2
            - 0.5 * (k + A(th_r, phi)) ** 2)
    return R_tun, R_en


def deriv_c(th_s, th_r, p_par, p_perp, phi=None):
    D = th_r - th_s
    k = k_of(th_s, th_r, phi)
    A_s, A_r = A(th_s, phi), A(th_r, phi)

    dk_s = (A_s + k) / D
    dk_r = -(A_r + k) / D

    d1_s = (k + A_s) * (dk_s + dA(th_s, phi))
    d1_r = (k + A_s) * dk_r
    d2_s = -(k + A_r) * dk_s
    d2_r = (p_par + A_r) * dA(th_r, phi) - (k + A_r) * (dk_r + dA(th_r, phi))

    return (d1_s, d1_r, 0.0 + 0.0j), (d2_s, d2_r, p_par + A_r + 0.0j)


def hess_det(th_s, th_r, p_par, p_perp, phi=None):
    (a_s, a_r, _), (b_s, b_r, _) = deriv_c(th_s, th_r, p_par, p_perp, phi)
    return a_s * b_r - a_r * b_s


def _cr(c):
    return np.array([[c.real, -c.imag], [c.imag, c.real]])


def F_real(x, p_par, p_perp, phi=None):
    R1, R2 = residual_c(x[0] + 1j * x[1], x[2] + 1j * x[3], p_par, p_perp, phi)
    return np.array([R1.real, R1.imag, R2.real, R2.imag])


def J_real(x, p_par, p_perp, phi=None):
    (a_s, a_r, a_p), (b_s, b_r, b_p) = deriv_c(
        x[0] + 1j * x[1], x[2] + 1j * x[3], p_par, p_perp, phi)
    J = np.zeros((4, 5))
    J[0:2, 0:2] = _cr(a_s)
    J[0:2, 2:4] = _cr(a_r)
    J[2:4, 0:2] = _cr(b_s)
    J[2:4, 2:4] = _cr(b_r)
    J[0, 4], J[1, 4] = a_p.real, a_p.imag
    J[2, 4], J[3, 4] = b_p.real, b_p.imag
    return J


def newton(x0, p_par, p_perp, phi=None, tol=1e-12, max_iter=80):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        F = F_real(x, p_par, p_perp, phi)
        if not np.isfinite(F).all():
            return x, False
        if np.linalg.norm(F) < tol:
            return x, True
        try:
            dx = np.linalg.solve(J_real(x, p_par, p_perp, phi)[:, :4], -F)
        except np.linalg.LinAlgError:
            return x, False
        if not np.isfinite(dx).all() or np.linalg.norm(dx) > 20.0:
            return x, False
        x = x + dx
    return x, np.linalg.norm(F_real(x, p_par, p_perp, phi)) < tol


def cond_fixed_p(x, p_par, p_perp, phi=None):
    return np.linalg.cond(J_real(x, p_par, p_perp, phi)[:, :4])


def is_backscattering(x, p_par, phi=None, tol=1e-4):
    th_s, th_r = x[0] + 1j * x[1], x[2] + 1j * x[3]
    k = k_of(th_s, th_r, phi)
    return abs(p_par + k + 2 * A(th_r, phi)) < tol


def classical_seeds(phi=None, n_grid=400, max_cycles=2.5):
    """Seeds from the classical return condition in the I_p -> 0 limit."""
    th_s = np.linspace(0.02, 2 * np.pi, n_grid)
    th_r = np.linspace(0.05, 2 * np.pi * max_cycles, 4 * n_grid)
    seeds = []
    for ts in th_s:
        k_cl = -A(ts, phi)
        g = k_cl * (th_r - ts) + (int_A(th_r, phi) - int_A(ts, phi))
        mask = th_r > ts + 0.2
        tr_m, g_m = th_r[mask], g[mask]
        for i in np.where(np.sign(g_m[:-1]) != np.sign(g_m[1:]))[0]:
            seeds.append((ts, 0.5 * (tr_m[i] + tr_m[i + 1])))
    return seeds


def find_orbits(p_par, p_perp, phi=None, im_seeds=(0.4, 0.8, 1.2),
                tol=1e-11, dedupe=1e-5, max_delay=None, backscatter_only=True):
    if max_delay is None:
        max_delay = 2.5 * 2 * np.pi
    found = []
    for ts, tr in classical_seeds(phi):
        for im in im_seeds:
            x, ok = newton(np.array([ts, im, tr, 0.05]), p_par, p_perp, phi, tol=tol)
            if not ok or x[1] <= 1e-3:
                continue
            delay = x[2] - x[0]
            if delay <= 0.2 or delay > max_delay:
                continue
            if backscatter_only and not is_backscattering(x, p_par, phi):
                continue
            shift = 2 * np.pi * np.floor(x[0] / (2 * np.pi))
            x = x - np.array([shift, 0.0, shift, 0.0])
            if any(np.linalg.norm(x - y) < dedupe * max(1.0, np.linalg.norm(y))
                   for y in found):
                continue
            found.append(x)
    found.sort(key=lambda v: v[2] - v[0])
    return found


def caustic_point(v0, p_perp, phi=None, tol=1e-13):
    from scipy.optimize import fsolve

    def G(v):
        th_s, th_r = v[0] + 1j * v[1], v[2] + 1j * v[3]
        p = v[4] + 1j * v[5]
        R1, R2 = residual_c(th_s, th_r, p, p_perp, phi)
        d = hess_det(th_s, th_r, p, p_perp, phi)
        return [R1.real, R1.imag, R2.real, R2.imag, d.real, d.imag]

    sol, _, ier, _ = fsolve(G, v0, full_output=True, xtol=tol)
    return sol, (ier == 1 and np.linalg.norm(G(sol)) < 1e-9)


def _tangent(J, prev=None):
    _, _, Vt = np.linalg.svd(J)
    t = Vt[-1] / np.linalg.norm(Vt[-1])
    if prev is not None and np.dot(t, prev) < 0:
        t = -t
    return t


def continue_branch(x0, p0, p_perp, p_lo, p_hi, phi=None, ds=2e-3, ds_min=1e-8,
                    ds_max=1e-2, max_steps=60000, tol=1e-11, direction=1.0):
    """Pseudo-arclength continuation in p_par at fixed p_perp and phi."""
    y = np.concatenate([np.asarray(x0, float), [p0]])
    t = _tangent(J_real(y[:4], y[4], p_perp, phi))
    if t[4] * direction < 0:
        t = -t

    path, tangents = [y.copy()], [t.copy()]
    step = ds
    for _ in range(max_steps):
        y_prev, t_prev = path[-1], tangents[-1]
        accepted = False
        while step >= ds_min:
            y_new = y_prev + step * t_prev
            ok = False
            for _ in range(40):
                F = F_real(y_new[:4], y_new[4], p_perp, phi)
                g = np.dot(t_prev, y_new - y_prev) - step
                G = np.concatenate([F, [g]])
                if not np.isfinite(G).all():
                    break
                if np.linalg.norm(G) < tol:
                    ok = True
                    break
                Jaug = np.vstack([J_real(y_new[:4], y_new[4], p_perp, phi), t_prev])
                try:
                    dy = np.linalg.solve(Jaug, -G)
                except np.linalg.LinAlgError:
                    break
                if not np.isfinite(dy).all() or np.linalg.norm(dy) > 5.0:
                    break
                y_new = y_new + dy
            if ok:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        t_new = _tangent(J_real(y_new[:4], y_new[4], p_perp, phi), prev=t_prev)
        path.append(y_new.copy())
        tangents.append(t_new.copy())
        step = min(step * 1.25, ds_max)
        if y_new[4] < p_lo or y_new[4] > p_hi:
            break
    return np.array(path), np.array(tangents)


def sample_path(path, p_targets, p_perp, phi=None, tol=1e-12):
    """Interpolate a traced branch onto a regular grid and polish."""
    p_path = path[:, 4]
    out = {}
    for i in range(len(p_path) - 1):
        a, b = p_path[i], p_path[i + 1]
        lo, hi = min(a, b), max(a, b)
        for j in np.where((p_targets >= lo) & (p_targets < hi))[0]:
            w = 0.0 if b == a else (p_targets[j] - a) / (b - a)
            guess = path[i, :4] + w * (path[i + 1, :4] - path[i, :4])
            x, ok = newton(guess, p_targets[j], p_perp, phi, tol=tol)
            if ok and j not in out:
                out[j] = x
    return out
