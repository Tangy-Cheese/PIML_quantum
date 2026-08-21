# Windowed PINNs for the Rescattered-ATI Saddle-Point Equations

Code, data and figure pipeline for the MSc dissertation *"Robustness of
Physics-Informed Neural Networks for the Rescattered Above-Threshold Ionisation
Saddle-Point Equations"*.

The project extends the windowed, unsupervised physics-informed neural network
(PINN) of Chen, Hashim & Figueira de Morisson Faria (arXiv:2603.15786) from
**direct** ATI — one saddle-point condition in one complex unknown — to
**rescattered** ATI: three coupled conditions reduced, by closed-form
elimination of the intermediate momentum $k$, to two complex equations in the
ionisation and return phases $(\theta_s, \theta_r)$.

Two solvers are implemented and cross-checked against each other:

| | method | role |
|---|---|---|
| **Reference** | pseudo-arclength Newton continuation, $\lVert F\rVert \le 10^{-12}$ | ground truth; traces a branch through folds |
| **PINN** | windowed, unsupervised, one network per orbit family | the object under test |

Everything in `output/` was produced by the scripts in `scripts/`, and every
figure and table in the dissertation traces to exactly one of them — see
[`docs/figure_map.md`](docs/figure_map.md).

## Physical setup

Phase units throughout ($\theta = \omega t$), atomic units, double precision.

* Field: monochromatic $A(\theta) = A_0\cos\theta$, or bichromatic
  $A(\theta) = A_1\cos\theta + A_2\cos(2\theta + \phi)$ with $A_2/A_1 = 0.3$.
* Hydrogen at 800 nm, $10^{14}\,\mathrm{W/cm^2}$:
  $\omega = 0.057$, $E_0 = 0.0534$, $I_p = 0.5$, $\gamma = 1.07$.
* Orbit families are indexed by increasing return delay:
  0 unpaired, 1–2 pair A, 3–4 pair B, 5–6 pair C, 7–8 pair D.

Reference values reproduced by this code:

| quantity | value |
|---|---|
| pair A caustic, $p_\perp = 0$ | $p_c = 2.197280 + 0.017697\,i$ |
| critical bichromatic phase | $\phi^\ast = 0.2489\pi$ (where $\mathrm{Im}\,p_c \to 0$) |
| condition number, generic field | $\kappa \le 33$ across four decades of $I_p$ |
| condition number at $\phi^\ast$ | $\kappa \approx 1.9\times10^{15}$ |
| cutoff-law fit vs. published $10.007\,U_p + 0.538\,I_p$ | $a = 10.0076$, $b = 0.5340$ |

## Layout

```
src/                     importable modules — no side effects, never run directly
  continuation.py          residuals, exact Jacobian, Newton, pseudo-arclength
                           continuation, discriminant caustic solver, action,
                           Hessian determinant, uniform (Airy) amplitudes
  bichromatic.py           the same physics for an omega-2omega field
  pinn_common.py           the k-eliminated, two-residual PINN and its trainer
  paths.py                 repo-anchored output paths (see "Paths" below)

scripts/
  ground_truth/          continuation reference data consumed by later stages
  diagnostics/           solver-side results: conditioning, caustic ring,
                         I_p scaling, cutoff law, phase scan
  figures/               every figure and table in the dissertation
  run_all.sh             the full pipeline in dependency order

output/<category>/       all generated .npz data and .png/.pdf figures
docs/                    figure map, scope and caveats
future_work/             kept but unpublished work — see future_work/README.md
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # numpy, scipy, matplotlib, torch
```

Developed on Python 3.14, numpy 2.4, scipy 1.17, torch 2.12, matplotlib 3.10.
CPU only; no GPU is required or used. Training one family takes on the order of
ten seconds and a full forward pass over the 2D momentum grid a few
milliseconds, so the whole pipeline is minutes, not hours.

## Reproducing everything

```bash
bash scripts/run_all.sh
```

or stage by stage, in this order (later stages read earlier stages' `.npz`):

```bash
# 1. continuation ground truth
python scripts/ground_truth/make_gt_pair_a.py        # gt_pair_a_{short,long}.npz
python scripts/ground_truth/make_gt_phistar.py       # critical_phi, gt_phistar_{pair,fine}

# 2. solver diagnostics (independent of the PINN)
python scripts/diagnostics/caustic_and_conditioning.py        # conditioning, caustic ring
python scripts/diagnostics/ip_scan.py                # I_p scaling  -> ip_scan.npz
python scripts/diagnostics/make_cutoff_law_fig.py    # needs ip_scan.npz
python scripts/diagnostics/phi_scan.py               # locates phi*

# 3. PINN results and figures
python scripts/figures/make_window_ablation.py         # window ablation, 5 seeds
python scripts/figures/make_all_families_1d.py          # all 8 families, 1D
python scripts/figures/replot_all_families_1d.py        # redraw only, no retraining
python scripts/figures/make_error_maps_2d.py            # all 8 families, 2D
python scripts/figures/make_pair_a_intensities.py             # pair A at 3 intensities
python scripts/figures/make_bichrom_phase_data.py all       # 3 relative phases
python scripts/figures/make_bichrom_phase_fig.py     # needs the above
python scripts/figures/make_field_comparison.py mono  # 5-seed monochromatic case
python scripts/figures/make_field_comparison.py fig   # figure + LaTeX table body
python scripts/figures/make_phistar_pinn.py               # Appendix D, phi* behaviour
python scripts/figures/make_polish_data.py mono      # hybrid PINN + Newton polish
python scripts/figures/make_polish_data.py phistar
python scripts/figures/make_polish_figs.py       # needs both polish cases
```

Scripts are seeded (`torch.manual_seed`), so a rerun reproduces the stored
numbers rather than merely similar ones.

### Paths

Every script resolves its paths from its own location via `src/paths.py`, never
from the working directory, so it can be run from anywhere:

```bash
python scripts/figures/make_error_maps_2d.py            # from the repo root
cd scripts/figures && python make_error_maps_2d.py      # or from its own directory
```

Figure scripts additionally copy their final figure into the dissertation source
tree, defaulting to `~/Documents/dissertation` and overridable:

```bash
ATI_FIGURE_DIR=/path/to/dissertation python scripts/figures/make_pair_a_intensities.py
```

If that directory does not exist the copy is skipped silently and the figure is
still written to `output/`.

## Conventions for new code

* Phase units, not real time; `torch.set_default_dtype(torch.float64)` everywhere.
* PINN output is squashed as `centre + halfwidth * tanh(raw)` — the window.
  Raw unconstrained output appears only in the explicit `unbounded=True` ablation.
* Adam, `lr = 2e-3`, `ReduceLROnPlateau`, best-state checkpointing keyed to the
  **physics loss** (the problem is unsupervised — there is no held-out set).
* Accuracy is always reported through `continuation.F_real`, the canonical
  residual — never the training loss curve, which can and does diverge from the
  true residual near a fold.
* Figure colours: `#452a7a` / `#241040` / `#9b7fd4` for the PINN, `#d05070` /
  `#c0392b` for contrast, baseline and warnings.

## Scope

The trajectory results are complete and validated. The amplitude/PMD stage is
**not** claimed: `uniform_anchored` in `continuation.py` was calibrated at
$\phi^\ast$ and does not reduce correctly to the ordinary stationary-phase
amplitude far from the caustic, so no momentum distribution is presented. That
work, and the honest list of everything else still open, is in
[`docs/scope_and_caveats.md`](docs/scope_and_caveats.md) and
[`future_work/README.md`](future_work/README.md).

## Licence

MIT — see [`LICENSE`](LICENSE).
