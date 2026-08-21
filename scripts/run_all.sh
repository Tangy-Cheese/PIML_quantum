#!/usr/bin/env bash
#
# Full pipeline, in dependency order. Every stage writes into output/<category>/
# and later stages read earlier stages' .npz files, so the order matters.
#
#   bash scripts/run_all.sh              run everything
#   bash scripts/run_all.sh diagnostics  run one stage: groundtruth | diagnostics | figures
#
# Set ATI_FIGURE_DIR to copy the final figures into a dissertation source tree;
# without it the copy is skipped and figures are written to output/ only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
STAGE="${1:-all}"

run() {
    echo
    echo "=== $* ==="
    "$PY" "$ROOT/scripts/$@"
}

if [[ "$STAGE" == "all" || "$STAGE" == "groundtruth" ]]; then
    run ground_truth/make_gt_pair_a.py         # gt_pair_a_{short,long}.npz
    run ground_truth/make_gt_phistar.py        # critical_phi, gt_phistar_{pair,fine}
fi

if [[ "$STAGE" == "all" || "$STAGE" == "diagnostics" ]]; then
    run diagnostics/caustic_and_conditioning.py         # conditioning, caustic ring, branch separation
    run diagnostics/ip_scan.py                 # I_p scaling -> ip_scan.npz
    run diagnostics/make_cutoff_law_fig.py     # reads ip_scan.npz
    run diagnostics/phi_scan.py                # locates phi*
fi

if [[ "$STAGE" == "all" || "$STAGE" == "figures" ]]; then
    run figures/make_window_ablation.py        # window ablation, 5 seeds (needs gt_pair_a_short)
    run figures/make_all_families_1d.py           # all 8 families, 1D
    run figures/replot_all_families_1d.py         # redraw only, no retraining
    run figures/make_error_maps_2d.py             # all 8 families, 2D
    run figures/make_pair_a_intensities.py              # pair A, three intensities
    run figures/make_bichrom_phase_data.py all        # omega-2omega, three relative phases
    run figures/make_bichrom_phase_fig.py      # needs the three phases above
    run figures/make_field_comparison.py mono   # 5-seed monochromatic case
    run figures/make_field_comparison.py fig    # figure + LaTeX table body
    run figures/make_phistar_pinn.py                # Appendix D: phi* behaviour
    run figures/make_polish_data.py mono       # hybrid PINN + Newton polish
    run figures/make_polish_data.py phistar
    run figures/make_polish_figs.py        # needs both polish cases
fi

echo
echo "done -- artifacts in $ROOT/output"
