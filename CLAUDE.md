# Rescattered ATI PINN — orientation for a new collaborator

*Written for Claude Code, or any human picking this up cold. Read this and the
two documents it points at before touching anything: several "obvious" next
steps are already done, and several things that look done are single-seed.*

## Where things are

| you want | read |
|---|---|
| what the project is, how to run it | [`README.md`](README.md) |
| which script made which figure or number | [`docs/figure_map.md`](docs/figure_map.md) |
| what is verified, what is qualified, what is not claimed | [`docs/scope_and_caveats.md`](docs/scope_and_caveats.md) |
| unpublished code and open threads | [`future_work/README.md`](future_work/README.md) |

The repository holds only what backs the dissertation, plus a clearly fenced
`future_work/`. Anything not in one of those two categories has been removed —
if you are looking for a script you remember and cannot find it, it was
superseded, and its replacement is named in `docs/figure_map.md`.

## The one-paragraph summary

Extend the windowed PINN of Chen, Hashim & Faria (arXiv:2603.15786) from direct
ATI — one saddle-point condition, one complex unknown — to rescattered ATI:
three coupled conditions (tunnelling, return, rescattering), reduced to two
complex equations in $(\theta_s, \theta_r)$ by eliminating the intermediate
momentum $k$ in closed form. At a single momentum the system has several
coexisting valid solutions (nine, searching return delays to 2.5 field cycles).
The central claim is that a continuous PINN cannot silently jump between them
the way fixed-momentum Newton iteration can, and that this — not Jacobian
ill-conditioning, which is a non-issue generically — is the failure mode worth
caring about.

Two remaining complex equations after eliminating $k$:

$$F_1: \tfrac12[k + A(\theta_s)]^2 + I_p = 0 \qquad
F_2: \tfrac12[p_\parallel + A(\theta_r)]^2 + \tfrac12 p_\perp^2
     - \tfrac12[k + A(\theta_r)]^2 = 0$$

with $k(\theta_s,\theta_r) = -A_0(\sin\theta_r - \sin\theta_s)/(\theta_r - \theta_s)$
in the monochromatic case; `src/bichromatic.py` generalises this to two
harmonics.

## Conventions any new code must match

* Phase units ($\theta = \omega t$), not real time.
* `torch.set_default_dtype(torch.float64)` — everything in double precision.
* PINN pattern: `nn.Linear` stack with `Tanh`, output squashed as
  `centre + halfwidth * tanh(raw)` (the window). Raw unconstrained output only
  in the explicit `unbounded=True` ablation path.
* Adam, `lr=2e-3`, `ReduceLROnPlateau(factor=0.5, ...)`, best-state
  checkpointing keyed to the **physics loss** — the problem is unsupervised, so
  there is no held-out set.
* `C.F_real(x, p_par, p_perp)` is the canonical residual. Always report accuracy
  and validity through it, never through the training loss curve: the two
  diverge near a fold, which is itself one of the results.
* Family indexing: 0 unpaired, 1&2 pair A (short, long), 3&4 pair B, 5&6 pair C,
  7&8 pair D — increasing return delay, per `find_orbits`'s own sort.
* Figure colours: `#452a7a` / `#241040` / `#9b7fd4` for the PINN, `#d05070` /
  `#c0392b` for contrast, baseline and warnings.
* Paths come from `src/paths.py` (`out("category")`), never from the working
  directory. New scripts follow the same bootstrap as the existing ones.

## Numbers not worth re-deriving

$p_c = 2.197280 + 0.017697i$ (pair A, $p_\perp = 0$, monochromatic) ·
$\phi^\ast = 0.2489\pi$ · $\kappa_{\max} = 33$ (generic, across $I_p$) ·
$\kappa = 1.9\times10^{15}$ (at $\phi^\ast$) · cutoff-law fit $a = 10.0076$,
$b = 0.5340$ vs. published $10.007$, $0.538$ ·
median $\lvert\Delta\theta_s\rvert = 5.34\times10^{-4}$ (8 families, 1D) ·
$1.77\times10^{-5}$ and $8.13\times10^{-5}$ (branches A/B at $\phi^\ast$) ·
Newton at $\phi^\ast$ lands 12.67 away · $-4.37$, the impossible return delay
from the generic branch-misidentification case.

## Working style that suits this project

The remaining work is run-read-traceback-fix iteration against a real runtime
(`future_work/pmd/` especially), not design in the abstract. Run the thing,
read the error, fix it — do not reason about shapes and interfaces from a
chat window when the interpreter will answer in seconds.
