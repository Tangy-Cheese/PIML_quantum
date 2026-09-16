# scripts/figures

## make_all_families_1d.py

All eight orbit families in one dimension.

Same 8-family 1D sweep, same p_grid, window sizing and seeds as the
original, but trained with the closed-form-k, two-residual PINN of
pinn_common.py instead of the free-k, three-residual one -- so the code
matches what report.tex claims (Eq. loss-k-eliminated-full) rather than
the "unreduced" system it was actually generated with.

Writes all_families_1d.npz, all_families_error_1d.png, all_families_summary_1d.png
into output/families/ (same filenames as the original, so report.tex's
\includegraphics lines do not need to change).

## make_bichrom_phase_data.py

Untuned bichromatic omega-2omega field, swept over relative phase.

    A(theta) = A_1 cos(theta) + A_2 cos(2 theta + phi),   A_2/A_1 = 0.3

using the same A_1 and harmonic ratio as the phi* construction
(bichromatic.py: A_1 = 0.0534/0.057, RATIO = 0.3), at three ordinary phases

    phi = 0.5 pi, 0.6 pi, 0.9 pi

all comfortably away from phi* = 0.2489 pi. The phases were chosen after
checking the caustic at each: |Im p_c| runs 5.7e-3 -> 2.1e-2 and kappa runs
2.9e3 -> 6.1e2 across the three, so all three are ordinary, off-axis
coalescences. Phases below ~0.4 pi were rejected because the caustic drifts
back toward the real axis there (at 0.2 pi, |Im p_c| = 1.1e-3 and
kappa = 1.8e4), which would reintroduce a near-degenerate case; that regime
belongs to Appendix D, not here.

Orbit pair A is identified independently at each phase, by adiabatic
continuation of the monochromatic pair A caustic in the harmonic ratio from
0 to 0.3 at that fixed phi -- the orbit splitting differs from phase to
phase, so the descendant pair cannot be assumed to be the same caustic.

    python make_bichrom_phase_data.py gt    0.6pi
    python make_bichrom_phase_data.py train 0.6pi
    python make_bichrom_phase_data.py all   0.6pi
    python make_bichrom_phase_data.py all   all      # every phase in sequence

Momentum range follows the same relative convention as the three-intensity
rows of the monochromatic pair A figure: [0.17, 1.16] x Re p_c. Training
matches that figure too (2500 epochs, pad-3.0 window box) rather than the
1500 epochs quoted in the report's hyperparameter table, so the two figures
are directly comparable.

## make_bichrom_phase_fig.py

Bichromatic phase-sweep figure: orbit pair A at three ordinary relative
phases, matching the layout of the monochromatic three-intensity figure.

Rows are relative phase (0.5 pi, 0.6 pi, 0.9 pi), columns are Re theta,
Im theta and the network's own physics residual -- the same three columns as
the intensity figure, so the two read as a pair: the intensity figure asks
whether the method holds as the coalescence relocates within one field, this
one asks whether it holds across the second field's own parameter space.

Also prints the summary table body for the write-up.

## make_error_maps_2d.py

Two-dimensional error maps for all eight orbit families.

Same 8-family 2D (p_par, p_perp) grid, window sizing and seeds as the
original, but trained with the closed-form-k, two-residual PINN of
pinn_common.py instead of the free-k, three-residual one.

Writes error_maps_2d.npz, error_maps_2d.png into output/families/
(same filenames as the original).

## make_field_comparison.py

Three validation cases side by side, and the summary table for the write-up.

Rows are field configurations rather than intensities:

    mono      monochromatic, orbit pair A, I = 1.0e14 W/cm^2 (the middle row
              of the existing three-intensity pair A figure)
    bichrom   untuned bichromatic omega-2omega, phi = 0.6 pi, A_2/A_1 = 0.3
    phistar   bichromatic at the engineered phase phi* = 0.2489 pi

Columns: Re theta, Im theta, physics residual ||F||, matching the layout of
the existing three-intensity figure.

Every case is run at 5 seeds. The monochromatic case is recomputed here at
seed 0 as a check that it reproduces the published figure's numbers exactly
before the extra seeds are added; this script never writes pair_a_intensities.png and
never changes the published monochromatic results.

    python make_field_comparison.py mono      # run + cache monochromatic, 5 seeds
    python make_field_comparison.py phistar   # run + cache phi* pair grid, 5 seeds
    python make_field_comparison.py fig       # figure + summary table

## make_orbit_families.py

Orbit families on the two field shapes used in this work -- idealised.

This is a schematic, not a data plot: it sketches where the release/return
phases of the shortest- and longest-delay orbit pairs sit relative to one
another and to the field, using hand-placed, deliberately well-spread
phase values rather than the literal converged saddle solutions. The real
solutions crowd together too tightly (several families share a release
phase to within the marker size) to read cleanly as points on the curve,
so this trades numerical exactness for a picture a reader can actually
parse. Exact values are given in-text and in the later validation figures
(Sec. validation-eight-orbit-families).

Points sit directly on the field curve, as in a genuine orbit-family plot,
just spaced out by hand instead of by the solver.

Writes output/families/orbit_families.png, and copies it to the report
figure directory when that exists.

## make_pair_a_intensities.py

Replacement for report.tex's old Figs. 7/8/9 (pairA_theta_real/imag/residual):
one 3x3 figure, rows = intensity, columns = Re(theta), Im(theta), physics
residual ||F||, for orbit pair A (short/long branches).

Adapted from the earlier pairA_resI.py (the pre-existing, correct reference for
this exact figure -- true k-elimination, closed-form k, caustic-based p_c
via continuation.caustic_point, momentum range set relative to the caustic)
rather than the earlier free-k PINN, and laid out as a 3x3 grid
instead of two separate Re/Im figures. Intensities match pairA_resI.py:
0.5, 1.0, 1.5 x 10^14 W/cm^2.

## make_phistar_construction.py

The phi* construction: the pair-A caustic locus as the relative phase varies.

Continues the pair-A caustic in the bichromatic relative phase phi, warm-started
from the converged phi* solution of output/ground_truth/critical_phi.npz, so the
caustic tracked is by construction the same one at every phase -- unlike
scripts/diagnostics/phi_scan.py, which re-detects caustics independently at each
phi and therefore loses branch identity where several lie close together.

Panel (a) is the locus in the complex momentum plane; panel (b) is |Im p_c|
against phi on a log axis, which is where the crossing of the real axis at phi*
is legible. The three relative phases used for the bichromatic validation of
Sec. 6.2 are computed independently by make_bichrom_phase_data.py and overlaid
as a consistency check on the continuation.

No scaling law is drawn on either panel: whether kappa_max follows
|Im p_c|^(-1/2) is an open question at the time of writing (the measured
exponent over the bichromatic phases is nearer -1.14), and this figure is the
construction, not the law.

Writes phistar_construction.png to the figure directory.

## make_phistar_pinn.py

Network behaviour at the engineered coalescence phi*.

Same phi* configuration, ground truth and evaluation as the original, but
trained with the closed-form-k, two-residual PINN of pinn_common.py instead
of the free-k, three-residual one. This is the network behind the
dissertation's headline phi* claim (Sec. 5.2), so it needs to match the
architecture actually described (Eq. loss-k-eliminated-full) like everything
else regenerated in this folder.

Writes phistar_pinn.npz, phistar_error.png, phistar_separation.png,
phistar_loss.png into output/phistar/ (same filenames as the original).

## make_polish_data.py

Hybrid PINN + Newton polish test, on a log grid approaching the caustic.

Two cases, run through identical machinery so they can be compared directly:

    mono     -- the ordinary orbit pair A coalescence of the monochromatic
                field, Re p_c = 2.1973 (Sec. 6.2's new headline case)
    phistar  -- the engineered bichromatic coalescence at phi* = 0.2489 pi,
                where Im p_c -> 0 by construction (Appendix D)

Both train the k-eliminated windowed PINN over 5 seeds, then polish each
network output with four damped Newton steps, and report the trajectory
error before and after polishing as a function of distance from the caustic,
together with whether the polished solution stayed on the branch it started
from.

    python make_polish_data.py mono
    python make_polish_data.py phistar

Writes polish_<case>.npz into output/polish/.

## make_polish_figs.py

Appendix D figures: behaviour on approach to the caustic, phi* against the
ordinary monochromatic case, with 5-seed spread.

    polish_approach.png   physics residual and branch separation against
                              distance from p_c, phi* vs ordinary, median over
                              5 seeds with min-max bands
    polish_comparison.png     PINN error before and after 4 damped Newton
                              steps, against distance from p_c, both cases

Reads output/polish/polish_{mono,phistar}.npz (make_polish_data.py).

## make_window_ablation.py

Window half-width ablation over five seeds.

Same 4-config (baseline/1x/3x/10x) x 5-seed window-width ablation as the
original, but trained with the closed-form-k, two-residual PINN of
pinn_common.py instead of the free-k, three-residual one.

The per-epoch "return residual" curve in the right panel of
window_ablation_loss.png is now a genuine diagnostic: k is reconstructed in
closed form from (theta_s, theta_r), so the return condition
k(th_r-th_s) + A_0(sin th_r - sin th_s) = 0 is satisfied identically by
construction, at every epoch, regardless of what the network has learned.
It is evaluated here purely to show that (it should sit at the
floating-point floor throughout, not track training progress), not because
it enters the loss -- only the tunnelling and rescattering residuals do.

Writes window_ablation.npz, window_ablation_loss.png, window_ablation_summary.png,
window_ablation_error.png into output/windowing/ (same filenames as the
original).

## make_window_ablation_loss_single.py

Single-column loss-curve figure for the window half-width ablation.

Same 4-config (baseline/1x/3x/10x) x 5-seed ablation, same seeds and same
training setup as make_window_ablation.py, so the numbers are identical to
the ones already quoted in Table 3 -- only the plotting differs:

  * left panel only (the per-config physics-loss curves); the right-hand
    per-residual diagnostic panel is not produced here,
  * sized for a single column rather than the full two-column width,
  * seeds are combined with the *mean* rather than the median.

Also writes window_ablation_hist.npz (the full per-epoch, per-seed loss
histories), which make_window_ablation.py discards -- with that on disk the
figure can be restyled later without retraining.

Writes window_ablation_loss_single.png and window_ablation_hist.npz into
output/windowing/.

## replot_all_families_1d.py

Redraw the all-families accuracy figure in the dissertation's standard
figure style, from the stored results of make_all_families_1d.py.

Reads output/families/all_families_1d.npz and re-renders only -- no network
is retrained here, so the plotted numbers are byte-identical to those already
reported in the text.

Differences from the original rendering: dissertation rcParams (serif, cm
mathtext, matched font sizes and tick style), no in-figure title (the caption
carries it), legend entries reduced to the orbit number alone, and 300 dpi.

## replot_error_maps_2d.py

Redraw the eight-family 2D error-map figure in the dissertation's standard
figure style, from the stored results of make_error_maps_2d.py.

Reads output/families/error_maps_2d.npz and re-renders only -- no network is
retrained here, so the plotted numbers are byte-identical to those already
reported in the text.

Differences from the original rendering: the same rcParams as
replot_all_families_1d.py (serif, cm mathtext, matched font sizes and tick
style), a canvas sized to the width the figure is actually printed at
(the full \textwidth of a two-column-spanning figure*, 17.6 cm) so that point
sizes on the page match Fig. 10 exactly rather than being shrunk by
\includegraphics, and 300 dpi.
