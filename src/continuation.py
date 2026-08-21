"""Pseudo-arclength continuation for the rescattered ATI saddle-point system.

Phase units theta = omega * t, matching rescattered.py. The intermediate
momentum k is eliminated analytically via the return condition, leaving two
complex equations in two complex unknowns (four real). The momentum p_par is
promoted to an unknown and an arclength normalisation row is appended, so the
augmented 5x5 Jacobian stays nonsingular at a simple fold and the branch is
traced straight through the coalescence.
"""

import numpy as np

OMEGA = 0.057
E_0 = 0.0534
I_P = 0.5
U_P = E_0**2 / (4 * OMEGA**2)
A_0 = E_0 / OMEGA


def A(th):
    return A_0 * np.cos(th)


def dA(th):
    return -A_0 * np.sin(th)


def k_of(th_s, th_r):
    """Return condition solved analytically for the parallel intermediate
    momentum. k_perp vanishes for linear polarisation.
    """
    return -A_0 * (np.sin(th_r) - np.sin(th_s)) / (th_r - th_s)


def residual_c(th_s, th_r, p_par, p_perp):
    k = k_of(th_s, th_r)
    R_tun = 0.5 * (k + A(th_s)) ** 2 + I_P
    R_en = 0.5 * (p_par + A(th_r)) ** 2 + 0.5 * p_perp**2 - 0.5 * (k + A(th_r)) ** 2
    return R_tun, R_en


def deriv_c(th_s, th_r, p_par, p_perp):
    D = th_r - th_s
    k = k_of(th_s, th_r)
    A_s, A_r = A(th_s), A(th_r)

    dk_s = (A_s + k) / D
    dk_r = -(A_r + k) / D

    d1_s = (k + A_s) * (dk_s + dA(th_s))
    d1_r = (k + A_s) * dk_r
    d2_s = -(k + A_r) * dk_s
    d2_r = (p_par + A_r) * dA(th_r) - (k + A_r) * (dk_r + dA(th_r))

    d1_p = 0.0 + 0.0j
    d2_p = p_par + A_r + 0.0j
    return (d1_s, d1_r, d1_p), (d2_s, d2_r, d2_p)


def _cr(c):
    return np.array([[c.real, -c.imag], [c.imag, c.real]])


def F_real(x, p_par, p_perp):
    th_s = x[0] + 1j * x[1]
    th_r = x[2] + 1j * x[3]
    R1, R2 = residual_c(th_s, th_r, p_par, p_perp)
    return np.array([R1.real, R1.imag, R2.real, R2.imag])


def J_real(x, p_par, p_perp):
    """Exact real 4x5 Jacobian; columns Re th_s, Im th_s, Re th_r, Im th_r, p_par."""
    th_s = x[0] + 1j * x[1]
    th_r = x[2] + 1j * x[3]
    (a_s, a_r, a_p), (b_s, b_r, b_p) = deriv_c(th_s, th_r, p_par, p_perp)
    J = np.zeros((4, 5))
    J[0:2, 0:2] = _cr(a_s)
    J[0:2, 2:4] = _cr(a_r)
    J[2:4, 0:2] = _cr(b_s)
    J[2:4, 2:4] = _cr(b_r)
    J[0, 4], J[1, 4] = a_p.real, a_p.imag
    J[2, 4], J[3, 4] = b_p.real, b_p.imag
    return J


def newton(x0, p_par, p_perp, tol=1e-12, max_iter=80):
    """Plain Newton at fixed momentum. The solver that fails at coalescence."""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        F = F_real(x, p_par, p_perp)
        if not np.isfinite(F).all():
            return x, False
        if np.linalg.norm(F) < tol:
            return x, True
        try:
            dx = np.linalg.solve(J_real(x, p_par, p_perp)[:, :4], -F)
        except np.linalg.LinAlgError:
            return x, False
        if not np.isfinite(dx).all() or np.linalg.norm(dx) > 20.0:
            return x, False
        x = x + dx
    return x, np.linalg.norm(F_real(x, p_par, p_perp)) < tol


def cond_fixed_p(x, p_par, p_perp):
    return np.linalg.cond(J_real(x, p_par, p_perp)[:, :4])


def classical_seeds(n_grid=600, max_cycles=2.5):
    """Real (theta_s, theta_r) pairs from the simple-man return condition
    cos(th_s)(th_r - th_s) = sin(th_r) - sin(th_s), which is the I_p -> 0
    limit of the coupled system. Used only to seed the first solve.
    """
    th_s = np.linspace(0.02, 2 * np.pi, n_grid)
    th_r = np.linspace(0.05, 2 * np.pi * max_cycles, 4 * n_grid)
    seeds = []
    for ts in th_s:
        g = np.cos(ts) * (th_r - ts) - (np.sin(th_r) - np.sin(ts))
        mask = th_r > ts + 0.2
        tr_m, g_m = th_r[mask], g[mask]
        sign_change = np.where(np.sign(g_m[:-1]) != np.sign(g_m[1:]))[0]
        for i in sign_change:
            seeds.append((ts, 0.5 * (tr_m[i] + tr_m[i + 1])))
    return seeds


def is_backscattering(x, p_par, tol=1e-4):
    """The quadratic energy condition admits both scattering signs. Forward
    scattering gives k = p_par identically and does not contribute to the
    plateau; backscattering satisfies p_par + k + 2 A(theta_r) = 0.
    """
    th_s = x[0] + 1j * x[1]
    th_r = x[2] + 1j * x[3]
    k = k_of(th_s, th_r)
    return abs(p_par + k + 2 * A(th_r)) < tol


def fold_to_cycle(x):
    """Shift a solution by a whole number of field periods so that
    Re(theta_s) lies in [0, 2 pi).
    """
    shift = 2 * np.pi * np.floor(x[0] / (2 * np.pi))
    y = x.copy()
    y[0] -= shift
    y[2] -= shift
    return y


def find_orbits(p_par, p_perp, im_seeds=(0.4, 0.8, 1.2), tol=1e-11,
                dedupe=1e-5, max_delay=None, backscatter_only=True):
    """Multi-start Newton from classical seeds, folded to one cycle and
    deduplicated. Returns orbits ordered by return delay, matching the
    1-4 labelling convention of Becker et al.
    """
    if max_delay is None:
        max_delay = 2.5 * 2 * np.pi
    found = []
    for ts, tr in classical_seeds():
        for im in im_seeds:
            x, ok = newton(np.array([ts, im, tr, 0.05]), p_par, p_perp, tol=tol)
            if not ok or x[1] <= 1e-3:
                continue
            delay = x[2] - x[0]
            if delay <= 0.2 or delay > max_delay:
                continue
            if backscatter_only and not is_backscattering(x, p_par):
                continue
            x = fold_to_cycle(x)
            dup = any(np.linalg.norm(x - y) < dedupe * max(1.0, np.linalg.norm(x))
                      for y in found)
            if not dup:
                found.append(x)
    found.sort(key=lambda v: v[2] - v[0])
    return found


def _tangent(J, prev=None):
    _, _, Vt = np.linalg.svd(J)
    t = Vt[-1] / np.linalg.norm(Vt[-1])
    if prev is not None and np.dot(t, prev) < 0:
        t = -t
    return t


def continue_branch(x0, p0, p_perp, p_lo, p_hi, ds=3e-3, ds_min=1e-7,
                    ds_max=3e-2, max_steps=40000, tol=1e-11, direction=1.0):
    """Trace one branch in p_par at fixed p_perp.

    Unknown vector y = (Re th_s, Im th_s, Re th_r, Im th_r, p_par).
    """
    y = np.concatenate([np.asarray(x0, float), [p0]])
    t = _tangent(J_real(y[:4], y[4], p_perp))
    if t[4] * direction < 0:
        t = -t

    path, tangents, conds = [y.copy()], [t.copy()], [cond_fixed_p(y[:4], y[4], p_perp)]
    step = ds

    for _ in range(max_steps):
        y_prev, t_prev = path[-1], tangents[-1]
        accepted = False

        while step >= ds_min:
            y_new = y_prev + step * t_prev
            ok = False
            for _ in range(40):
                F = F_real(y_new[:4], y_new[4], p_perp)
                g = np.dot(t_prev, y_new - y_prev) - step
                G = np.concatenate([F, [g]])
                if not np.isfinite(G).all():
                    break
                if np.linalg.norm(G) < tol:
                    ok = True
                    break
                Jaug = np.vstack([J_real(y_new[:4], y_new[4], p_perp), t_prev])
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

        t_new = _tangent(J_real(y_new[:4], y_new[4], p_perp), prev=t_prev)
        path.append(y_new.copy())
        tangents.append(t_new.copy())
        conds.append(cond_fixed_p(y_new[:4], y_new[4], p_perp))
        step = min(step * 1.25, ds_max)

        if y_new[4] < p_lo or y_new[4] > p_hi:
            break

    return np.array(path), np.array(tangents), np.array(conds)


def fold_indices(tangents):
    dp = tangents[:, 4]
    return np.where(np.sign(dp[:-1]) * np.sign(dp[1:]) < 0)[0]


def _antideriv(th, c_par, c_perp2):
    """Antiderivative of (c_par + A_0 cos th)^2 + c_perp2 with respect to th."""
    return ((c_par**2 + c_perp2) * th
            + 2 * c_par * A_0 * np.sin(th)
            + A_0**2 * (th / 2 + np.sin(2 * th) / 4))


def action(th_s, th_r, p_par, p_perp, th_ref=0.0):
    """Semiclassical action, eq. 2.3.2, in phase units.

    The overall 1/omega converts the phase integral back to time. The
    upper limit of the final propagation integral is replaced by th_ref;
    the discarded remainder depends on p but not on the orbit, so it is a
    common phase at each momentum and cancels in |M_p|^2.
    """
    k = k_of(th_s, th_r)
    S = I_P * th_s
    S -= 0.5 * (_antideriv(th_r, k, 0.0) - _antideriv(th_s, k, 0.0))
    S -= 0.5 * (_antideriv(th_ref, p_par, p_perp**2) - _antideriv(th_r, p_par, p_perp**2))
    return S / OMEGA


def hess_det(th_s, th_r, p_par, p_perp):
    (a_s, a_r, _), (b_s, b_r, _) = deriv_c(th_s, th_r, p_par, p_perp)
    return a_s * b_r - a_r * b_s


def orbit_amplitude(th_s, th_r, p_par, p_perp, prefactor=True, im_cut=60.0):
    """Single-orbit contribution. V_pk and d set to unity for a first pass;
    the plateau and fringe structure come from exp(iS).
    """
    S = action(th_s, th_r, p_par, p_perp)
    if not np.isfinite(S) or S.imag < -1e-9 or S.imag > im_cut:
        return 0.0 + 0.0j
    amp = np.exp(1j * S)
    if prefactor:
        det = hess_det(th_s, th_r, p_par, p_perp)
        if abs(det) < 1e-12:
            return 0.0 + 0.0j
        amp = amp * (2 * np.pi * 1j) / (OMEGA * np.sqrt(det))
    return amp


def sample_path(path, p_targets, p_perp, tol=1e-12):
    """Interpolate a traced branch onto a regular p_par grid, then polish
    with fixed-p Newton. Near a fold this is the hybrid scheme: continuation
    supplies a guess good enough for Newton to converge where it would
    otherwise diverge.
    """
    p_path = path[:, 4]
    out = {}
    for i in range(len(p_path) - 1):
        a, b = p_path[i], p_path[i + 1]
        lo, hi = min(a, b), max(a, b)
        idx = np.where((p_targets >= lo) & (p_targets < hi))[0]
        for j in idx:
            w = 0.0 if b == a else (p_targets[j] - a) / (b - a)
            guess = path[i, :4] + w * (path[i + 1, :4] - path[i, :4])
            x, ok = newton(guess, p_targets[j], p_perp, tol=tol)
            if ok:
                out.setdefault(j, []).append(x)
    return out


def caustic_point(v0, p_perp, tol=1e-13):
    """Solve the discriminant system F = 0, det(Hessian) = 0 jointly for
    (theta_s, theta_r, p_par) with p_par complex. Locates the caustic
    exactly rather than inferring it from where a solver misbehaves.
    """
    from scipy.optimize import fsolve

    def G(v):
        th_s = v[0] + 1j * v[1]
        th_r = v[2] + 1j * v[3]
        p = v[4] + 1j * v[5]
        R1, R2 = residual_c(th_s, th_r, p, p_perp)
        d = hess_det(th_s, th_r, p, p_perp)
        return [R1.real, R1.imag, R2.real, R2.imag, d.real, d.imag]

    sol, _, ier, _ = fsolve(G, v0, full_output=True, xtol=tol)
    return sol, (ier == 1 and np.linalg.norm(G(sol)) < 1e-10)


def caustic_locus(v_start, p_perp_grid):
    """Trace the caustic ring by continuation in p_perp."""
    v = np.array(v_start, float)
    out = []
    for pp in p_perp_grid:
        v_new, ok = caustic_point(v, pp)
        if not ok:
            break
        v = v_new
        out.append((pp, v[4] + 1j * v[5]))
    return out


def ordinary_prefactor(th_s, th_r, p_par, p_perp):
    """Prefactor of the ordinary two-dimensional saddle-point contribution,
    i.e. the contribution with exp(iS) divided out.
    """
    det = hess_det(th_s, th_r, p_par, p_perp)
    if abs(det) < 1e-14:
        return np.inf + 0j
    return (2 * np.pi * 1j) / (OMEGA * np.sqrt(det))


def uniform_pair(x1, x2, p_par, p_perp):
    """Uniform (Airy) approximation for a coalescing pair of saddles.

    The local action is mapped onto the cubic normal form
    S(t) -> u^3/3 - zeta u + A, with

        A    = (S_1 + S_2) / 2
        zeta = [3 (S_2 - S_1) / 4]^(2/3)

    and the slowly varying prefactor expanded to linear order in u. The
    coefficients are fixed by demanding that the result reduce to the sum
    of the two ordinary contributions as |zeta| -> infinity, which gives

        g_1 = m_1 zeta^(1/4) exp(-i pi/4) / sqrt(pi)
        g_2 = m_2 zeta^(1/4) exp(+i pi/4) / sqrt(pi)

    with m_j the ordinary prefactors. The Airy function interpolates
    continuously between the separated and coalescent regimes, so no
    explicit rule for switching orbits on or off is required.
    """
    from scipy.special import airy

    th_s1, th_r1 = x1[0] + 1j * x1[1], x1[2] + 1j * x1[3]
    th_s2, th_r2 = x2[0] + 1j * x2[1], x2[2] + 1j * x2[3]

    S1 = action(th_s1, th_r1, p_par, p_perp)
    S2 = action(th_s2, th_r2, p_par, p_perp)
    m1 = ordinary_prefactor(th_s1, th_r1, p_par, p_perp)
    m2 = ordinary_prefactor(th_s2, th_r2, p_par, p_perp)

    if not (np.isfinite(m1) and np.isfinite(m2)):
        return 0.0 + 0.0j

    d = 0.75 * (S2 - S1)
    zeta = d ** (2.0 / 3.0)
    if zeta.real < 0:
        S1, S2 = S2, S1
        m1, m2 = m2, m1
        d = 0.75 * (S2 - S1)
        zeta = d ** (2.0 / 3.0)

    A_mid = 0.5 * (S1 + S2)
    root = np.sqrt(zeta)
    q4 = zeta ** 0.25

    g1 = m1 * q4 * np.exp(-1j * np.pi / 4) / np.sqrt(np.pi)
    g2 = m2 * q4 * np.exp(+1j * np.pi / 4) / np.sqrt(np.pi)

    p_c = 0.5 * (g1 + g2)
    q_c = (g1 - g2) / (2 * root)

    ai, aip, _, _ = airy(-zeta)
    return 2 * np.pi * np.exp(1j * A_mid) * (p_c * ai - 1j * q_c * aip)


def safe_airy(w):
    """Airy functions with a guard against scipy's signed-zero branch.

    scipy.special.airy branches on the sign bit of a zero imaginary part, so
    airy(-z) and airy(complex(-z.real, 0.0)) disagree when z is real. Ai is an
    entire function, so both must give the same value; the -0.0 path returns a
    different branch. Building the argument as -zeta produces exactly that
    negative zero whenever zeta is real, which is the common case below and
    above a caustic. Forcing +0.0 fixes it.
    """
    from scipy.special import airy
    w = np.asarray(w, dtype=complex)
    im = np.where(w.imag == 0.0, 0.0, w.imag)
    return airy(w.real + 1j * im)


def uniform_sweep(xa_list, xb_list, p_grid, p_perp):
    """Uniform approximation applied along a momentum sweep with the branch
    of zeta tracked continuously.

    The labelling of the pair and the branch of zeta^(2/3) cannot be chosen
    pointwise: below the caustic S_2 - S_1 is real and zeta is real positive,
    while above it the difference rotates into the imaginary axis. Choosing
    the branch by a sign test at each momentum introduces a discontinuity
    exactly at the caustic. Here arg(S_2 - S_1) is unwrapped along the sweep
    starting from the well separated regime, where the labelling is
    unambiguous.
    """
    from scipy.special import airy

    S1 = np.empty(len(p_grid), dtype=complex)
    S2 = np.empty(len(p_grid), dtype=complex)
    m1 = np.empty(len(p_grid), dtype=complex)
    m2 = np.empty(len(p_grid), dtype=complex)

    for j, (xa, xb, p) in enumerate(zip(xa_list, xb_list, p_grid)):
        tsa, tra = xa[0] + 1j * xa[1], xa[2] + 1j * xa[3]
        tsb, trb = xb[0] + 1j * xb[1], xb[2] + 1j * xb[3]
        S1[j] = action(tsb, trb, p, p_perp)
        S2[j] = action(tsa, tra, p, p_perp)
        m1[j] = ordinary_prefactor(tsb, trb, p, p_perp)
        m2[j] = ordinary_prefactor(tsa, tra, p, p_perp)

    d = 0.75 * (S2 - S1)
    phase = np.unwrap(np.angle(d))
    zeta = np.abs(d) ** (2.0 / 3.0) * np.exp(1j * (2.0 / 3.0) * phase)

    A_mid = 0.5 * (S1 + S2)
    root = np.sqrt(np.abs(zeta)) * np.exp(0.5j * (2.0 / 3.0) * phase)
    q4 = np.abs(zeta) ** 0.25 * np.exp(0.25j * (2.0 / 3.0) * phase)

    g1 = m1 * q4 * np.exp(-1j * np.pi / 4) / np.sqrt(np.pi)
    g2 = m2 * q4 * np.exp(+1j * np.pi / 4) / np.sqrt(np.pi)
    p_c = 0.5 * (g1 + g2)
    q_c = (g1 - g2) / (2 * root)

    ai, aip, _, _ = airy(-zeta)
    M = 2 * np.pi * np.exp(1j * A_mid) * (p_c * ai - 1j * q_c * aip)
    return M, zeta, S1, S2


def uniform_sweep_stable(xa_list, xb_list, p_grid, p_perp):
    """Uniform (Airy) approximation evaluated without intermediate overflow.

    The naive form multiplies exp(i A_mid) by Ai(-zeta). Away from the
    caustic Ai(-zeta) grows like exp(2/3 |zeta|^{3/2}) while exp(i A_mid)
    decays by a compensating amount, so the product is bounded but the
    factors are not: at Im(zeta) ~ 40 the intermediate value is ~1e84.

    The cancellation is exact and can be done analytically. Writing
    w = -zeta and using the scipy exponentially scaled Airy functions

        Ai(w)  = eAi(w)  exp(-(2/3) w^{3/2})
        Ai'(w) = eAip(w) exp(-(2/3) w^{3/2})

    and noting that (2/3) zeta^{3/2} = (S_2 - S_1)/2 by construction of
    zeta, together with A_mid = (S_1 + S_2)/2, gives

        (2/3) w^{3/2} = -i (S_2 - S_1)/2
        exp(i A_mid) Ai(-zeta) = eAi exp(i S_2)

    so the whole amplitude carries a single bounded exponential exp(i S_2).
    """
    from scipy.special import airye

    n = len(p_grid)
    S1 = np.empty(n, dtype=complex)
    S2 = np.empty(n, dtype=complex)
    m1 = np.empty(n, dtype=complex)
    m2 = np.empty(n, dtype=complex)

    for j, (xa, xb, p) in enumerate(zip(xa_list, xb_list, p_grid)):
        tsa, tra = xa[0] + 1j * xa[1], xa[2] + 1j * xa[3]
        tsb, trb = xb[0] + 1j * xb[1], xb[2] + 1j * xb[3]
        S1[j] = action(tsb, trb, p, p_perp)
        S2[j] = action(tsa, tra, p, p_perp)
        m1[j] = ordinary_prefactor(tsb, trb, p, p_perp)
        m2[j] = ordinary_prefactor(tsa, tra, p, p_perp)

    d = 0.75 * (S2 - S1)
    collapsed = np.abs(d) < 1e-12
    phase = np.unwrap(np.angle(np.where(collapsed, 1.0, d)))
    zeta = np.abs(d) ** (2.0 / 3.0) * np.exp(1j * (2.0 / 3.0) * phase)

    root = np.sqrt(np.abs(zeta)) * np.exp(0.5j * (2.0 / 3.0) * phase)
    q4 = np.abs(zeta) ** 0.25 * np.exp(0.25j * (2.0 / 3.0) * phase)

    g1 = m1 * q4 * np.exp(-1j * np.pi / 4) / np.sqrt(np.pi)
    g2 = m2 * q4 * np.exp(+1j * np.pi / 4) / np.sqrt(np.pi)
    p_c = 0.5 * (g1 + g2)
    q_c = np.where(np.abs(root) > 1e-14, (g1 - g2) / (2 * root), 0.0)
    A_mid = 0.5 * (S1 + S2)

    Ai, Aip, _, _ = safe_airy(-zeta)
    M = 2 * np.pi * np.exp(1j * A_mid) * (p_c * Ai - 1j * q_c * Aip)

    M = np.where(collapsed, 0.0, M)
    return M, zeta, S1, S2


def pair_amplitude(xa_list, xb_list, p_grid, p_perp, zeta_switch=3.0,
                   blend=1.0):
    """Hybrid uniform / ordinary amplitude for a coalescing pair.

    The uniform formula retains both members of the pair by construction.
    That is correct near the coalescence, where the two are indistinguishable
    and the Airy function performs the switching. Far past the coalescence one
    member lies on the far side of a Stokes line: its Im(S) turns negative and
    its contribution grows exponentially, so it must be dropped. The uniform
    formula has no mechanism to do that.

    Here the uniform form is used where |zeta| < zeta_switch and the ordinary
    sum, with exponentially growing saddles discarded, is used outside, with a
    linear blend across a transition band so the amplitude stays continuous.
    """
    Mu, zeta, S1, S2 = uniform_sweep_stable(xa_list, xb_list, p_grid, p_perp)

    Mo = np.zeros(len(p_grid), dtype=complex)
    for j, (xa, xb, p) in enumerate(zip(xa_list, xb_list, p_grid)):
        for x in (xa, xb):
            Mo[j] += orbit_amplitude(x[0] + 1j * x[1], x[2] + 1j * x[3],
                                     p, p_perp, im_cut=400.0)

    az = np.abs(zeta)
    w = np.clip((az - zeta_switch) / blend, 0.0, 1.0)

    # Past the pair's own caustic one member crosses a Stokes line: its
    # Im(S) turns negative and its contribution grows exponentially. The
    # uniform form retains both members unconditionally, so it must not be
    # used there. Force the filtered ordinary sum wherever either member is
    # exponentially growing.
    growing = (S1.imag < 0) | (S2.imag < 0)
    w = np.where(growing, 1.0, w)

    Mu = np.where(np.isfinite(Mu), Mu, 0.0)
    Mo = np.where(np.isfinite(Mo), Mo, 0.0)
    return (1 - w) * Mu + w * Mo, zeta


def uniform_anchored(xa_list, xb_list, p_grid, p_perp, j_caustic):
    """Uniform Airy amplitude with the zeta branch anchored at the caustic.

    Unwrapping arg(S_2 - S_1) along the sweep fixes the branch correctly only
    if the sweep begins in the well separated oscillatory region. Pairs whose
    caustic lies at low energy spend most of the sweep beyond it, and the
    accumulated phase then places zeta on the wrong branch: the amplitude
    grows past the cutoff instead of decaying.

    Anchoring removes the dependence on phase history. Below the caustic zeta
    is real positive, so Ai(-zeta) oscillates and the two saddles interfere;
    above it zeta is real negative, so Ai(-zeta) decays and the growing saddle
    is suppressed automatically. This is the standard Airy convention for a
    fold and it requires only the caustic index, which is available from the
    branch separation minimum.

    The construction takes zeta real, which assumes the caustic lies close to
    the real momentum plane. For the monochromatic field Im(p_c) = 0.018, and
    at the critical bichromatic phase it is zero to machine precision.
    """
    from scipy.special import airye

    n = len(p_grid)
    S1 = np.array([action(x[0] + 1j * x[1], x[2] + 1j * x[3], p, p_perp)
                   for x, p in zip(xb_list, p_grid)])
    S2 = np.array([action(x[0] + 1j * x[1], x[2] + 1j * x[3], p, p_perp)
                   for x, p in zip(xa_list, p_grid)])
    m1 = np.array([ordinary_prefactor(x[0] + 1j * x[1], x[2] + 1j * x[3], p, p_perp)
                   for x, p in zip(xb_list, p_grid)])
    m2 = np.array([ordinary_prefactor(x[0] + 1j * x[1], x[2] + 1j * x[3], p, p_perp)
                   for x, p in zip(xa_list, p_grid)])

    d = 0.75 * (S2 - S1)
    sgn = np.where(np.arange(n) < j_caustic, 1.0, -1.0)
    zeta = sgn * np.abs(d) ** (2.0 / 3.0)

    A_mid = 0.5 * (S1 + S2)
    zc = zeta.astype(complex)
    root = np.sqrt(zc)
    q4 = zc ** 0.25

    g1 = m1 * q4 * np.exp(-1j * np.pi / 4) / np.sqrt(np.pi)
    g2 = m2 * q4 * np.exp(+1j * np.pi / 4) / np.sqrt(np.pi)
    P = 0.5 * (g1 + g2)
    Q = np.where(np.abs(root) > 1e-14, (g1 - g2) / (2 * root), 0.0)

    Ai, Aip, _, _ = safe_airy(-zc)
    M = 2 * np.pi * np.exp(1j * A_mid) * (P * Ai - 1j * Q * Aip)
    return np.where(np.isfinite(M), M, 0.0), zeta


def caustic_index(xa_list, xb_list):
    d = np.array([np.linalg.norm(np.asarray(a) - np.asarray(b))
                  for a, b in zip(xa_list, xb_list)])
    return int(np.argmin(d)), d.min()
