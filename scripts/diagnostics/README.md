# scripts/diagnostics

## make_cutoff_law_fig.py

Produces: cutoff_law.png

Fits the rescattered cutoff law E_c = b*Ip + a*Up from caustic locations
found by the discriminant system, and compares to the published values
10.007 Up + 0.538 Ip.

Slide reference: "Validation: cutoff law"
Requires: ip_scan.npz (run ip_scan.py first)
Runtime: <1 minute (uses saved data, no resolving)
