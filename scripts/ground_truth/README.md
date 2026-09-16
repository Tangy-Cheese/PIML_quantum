# scripts/ground_truth

## make_gt_phistar.py

Locate the critical bichromatic phase phi* and generate ground truth there.

phi* is the relative phase at which the caustic crosses the real momentum axis,
so that Im(p_c) = 0 and the fixed-momentum Jacobian becomes genuinely singular.

Produces:
    critical_phi.npz      phi*, p_c, and the coalescing solution
    gt_phistar_pair.npz   both branches on a uniform grid
    gt_phistar_fine.npz   both branches on a log grid approaching p_c

Runtime: about 5 minutes.
