# Figure and table map

Every figure and table in `report.tex` that is not taken from the literature,
with the script that produces it and the artifact it writes. Run order is the
order of `scripts/run_all.sh`; anything listed under "requires" must have been
run first.

## Figures

| report figure | script | writes | requires |
|---|---|---|---|
| Newton condition number vs. $p_\parallel$ (`newton_conditioning.png`) | `scripts/diagnostics/caustic_and_conditioning.py` | `output/diagnostics/newton_conditioning.png` | — |
| Loss against training epoch, window ablation (`window_ablation_loss.png`) | `scripts/figures/make_window_ablation.py` | `output/windowing/` | `make_gt_pair_a.py` |
| Pair A at three intensities, 3×3 (`pair_a_intensities.png`) | `scripts/figures/make_pair_a_intensities.py` | `output/pair_a/` | — |
| Bichromatic pair A at three phases (`bichrom_phases.png`) | `scripts/figures/make_bichrom_phase_data.py all` → `make_bichrom_phase_fig.py` | `output/bichromatic/` | — |
| 2D error maps, all eight families (`error_maps_2d.png`) | `scripts/figures/make_error_maps_2d.py` | `output/families/` | — |
| Accuracy vs. final energy, all eight families (`all_families_error_1d.png`) | `scripts/figures/make_all_families_1d.py` → `replot_all_families_1d.py` | `output/families/` | — |
| Appendix D: approach to the caustic (`polish_approach.png`) | `scripts/figures/make_polish_data.py {mono,phistar}` → `make_polish_figs.py` | `output/polish/` | `make_gt_phistar.py` |
| Appendix D: error before/after Newton polish (`polish_comparison.png`) | as above | `output/polish/` | as above |

`xenon.png` and `three_step.jpg` in the dissertation are reproduced from the
literature (Schafer *et al.*; three-step-model sketch) and have no generating
script here.

## Tables and quoted numbers

| report element | script | writes |
|---|---|---|
| Window half-width ablation table (5 seeds) | `scripts/figures/make_window_ablation.py` | `output/windowing/window_ablation.npz` |
| Accuracy on the two physical fields (mono vs. bichromatic $\phi=0.6\pi$) | `scripts/figures/make_field_comparison.py mono` then `fig` — prints the LaTeX table body | `output/comparison/` |
| Appendix D $\phi^\ast$ branch-separation numbers | `scripts/figures/make_phistar_pinn.py` | `output/phistar/` |
| Caustic location $p_c = 2.197280 + 0.017697i$, $\kappa$ on the real axis | `scripts/diagnostics/caustic_and_conditioning.py` | `output/diagnostics/caustic_conditioning.npz` |
| $\kappa \le 33$ across four decades of $I_p$ | `scripts/diagnostics/ip_scan.py` | `output/diagnostics/ip_scan.npz` |
| Cutoff-law fit vs. published $10.007\,U_p + 0.538\,I_p$ | `scripts/diagnostics/make_cutoff_law_fig.py` | `output/diagnostics/cutoff_law.png` |
| $\phi^\ast = 0.2489\pi$, $\kappa \approx 1.9\times10^{15}$ | `scripts/ground_truth/make_gt_phistar.py`, `scripts/diagnostics/phi_scan.py` | `output/ground_truth/critical_phi.npz`, `output/diagnostics/phi_scan.npz` |

## Output categories

| directory | holds |
|---|---|
| `ground_truth/` | continuation reference data (`gt_*.npz`, `critical_phi.npz`) read by everything downstream |
| `diagnostics/` | solver-side results, independent of the PINN |
| `windowing/` | window half-width ablation |
| `families/` | all eight orbit families, 1D and 2D |
| `pair_a/` | the pair A three-intensity figure |
| `bichromatic/` | $\omega$–$2\omega$ phase sweep |
| `comparison/` | monochromatic vs. bichromatic summary table and figure |
| `phistar/` | behaviour at the engineered coalescence |
| `polish/` | hybrid PINN + Newton polish test and Appendix D figures |
| `future_work_pmd/` | unpublished amplitude/PMD experiments — see `future_work/README.md` |

## Figures produced but not used in the dissertation

These are kept because they are cheap by-products of scripts that are needed
anyway, and they document intermediate checks:
`output/diagnostics/{branch_path,branch_separation,caustic_imag,caustic_ring,hessian_det,ip_scan_*,phi_scan_*}.png`,
`output/families/all_families_summary_1d.png`,
`output/windowing/{window_ablation_summary,window_ablation_error}.png`,
`output/phistar/{phistar_error,phistar_loss,phistar_separation}.png`,
`output/comparison/field_comparison.png`.
