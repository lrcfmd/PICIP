"""
PICIP Tutorial — 3-D Phase Fields
===================================
Sections:  1. Quick start  2. Display modes  3. Detail view
           4. predicted_error  5. Multiple samples  6. Suggestions
           7. Charged phase fields (setup_charged)
Run top to bottom — each section opens a browser figure.

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md
  Setup tutorial: tutorial_setup.py
  2-D tutorial  : tutorial_2d.py
  Evaluation    : tutorial_evaluation.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter
from pymatgen.core import Composition


# Phase_Field defines the composition space PICIP operates over.
# setup_uncharged: grid of all compositions summing to 1, no charge constraint.
# 4 elements → 3-D tetrahedron — make_plotter returns a Cube automatically.
#
# >>> See `tutorial_setup.py` for the full menu: setup_uncharged / setup_charged /
#     setup_charge_ranges / setup_custom, n_points/resolution, precursor restriction.
#     Section 7 below runs this whole workflow again on a CHARGED field and spells
#     out exactly what has to change. <<<
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])

# ── Other 3-D phase fields you could use instead (uncomment to try) ───────────
#   Every option below is 3-D, so the rest of this tutorial runs unchanged.
#   The rule is: setup_uncharged and setup_charge_ranges give n_elements - 1
#   dimensions; setup_charged adds a second constraint and gives n_elements - 2.
#   For 2-D fields see tutorial_2d.py, and tutorial_setup.py for the full menu.
#
# # Charged system, fixed oxidation states — 5 elements → 3-D.  Section 7:
# pf.setup_charged({"Li": +1, "Ni": +3, "Mn": +4, "Co": +3, "O": -2})
#
# # Variable oxidation states — 4 elements → 3-D.  Mixed-valence elements take
# # a [min_charge, max_charge] pair; fixed ones take a scalar:
# pf.setup_charge_ranges({"Fe": [+2, +3], "Mn": [+2, +4], "Ti": +4, "O": -2})
#
# # Restrict to compositions reachable from a specific precursor set:
# pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"], precursors=["FeO", "MnO2", "TiO2", "CuO"])
#
# # Coarser/finer grid (3-D defaults to ~1000 points):
# pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"], n_points=300)   # faster, lower resolution
# pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"], n_points=4000)  # slower, higher resolution
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.  QUICK START
# ══════════════════════════════════════════════════════════════════════════════

# Create sample: measured composition + known co-existing phases + their mass fractions.
# add_mass_weights takes mass fractions as returned by Rietveld refinement.
# Use add_molar_weights instead if you have molar fractions.
sample = Sample(
    "s1",               # name — label used in plots and titles
    "Fe2Mn2Ti2Cu4",     # composition — measured sample (pymatgen formula string or Composition)
)
sample.add_knowns(["FeMn", "FeTi", "FeCu"])          # known co-existing phases
sample.add_mass_weights([0.25, 0.35, 0.40])          # mass fractions of each known phase (e.g. from Rietveld)
sample.set_predicted_error(0.3)                      # uncertainty in the phase_fractions (0=precise, 1=rough)

# Initialise PICIP and add the sample
picip = PICIP(pf)
picip.add_sample(sample)

# Run inference — returns a Prediction object.
# version='gaussian' is the default: the original PICIP algorithm from the
# paper, and the well-tested path for any number of knowns in any dimension.
pred = picip.run(
    n_l=50,   # number of ray directions (more = smoother, diminishing returns >100)
    n_p=10,   # points per projected ray (10–50 sufficient; use more for sharper densities)
)

# ── Opting into the Dirichlet method ──────────────────────────────────────────
#
# version='dirichlet' selects the Dirichlet method instead.  Rather than collapsing the
# knowns to their centroid, it puts a Dirichlet distribution over the mixing
# ratio of the known phases and projects that through the sample.  In 3-D the
# support is the k1–k2–k3 triangle, discretised uniformly in solid angle from
# the sample — the 2-D line method one dimension up.
#
# CAVEAT: it is implemented for exactly two configurations —
#     3-D constrained space with exactly 3 knowns   (this sample)
#     2-D constrained space with exactly 2 knowns   (see tutorial_2d.py)
# Ask for it anywhere else — 1 or 2 knowns here, or 4+ in a 3-D field — and
# PICIP warns that the method is untested there and falls back to Gaussian, so
# you still get a result, just not the one you asked for.  Watch for:
#
#   RuntimeWarning: The Dirichlet method (version='dirichlet') is implemented only for
#   2-D with 2 known phases and 3-D with 3 known phases ... falling back to the
#   Gaussian method.
#
# This sample has 3 knowns in a 3-D field, so it genuinely takes the Dirichlet
# path.  Everything downstream works identically; swap `pred_dirichlet` for
# `pred` below to compare them.
pred_dirichlet = picip.run(version='dirichlet', n_l=50, n_p=10, name='Dirichlet')
# ──────────────────────────────────────────────────────────────────────────────

# Suggestions — what to synthesise next — are covered in section 6.

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DISPLAY MODES
#     Three ways to render the 3-D probability cloud — toggle between them
#     in the legend.
#
#   'nonzero'  — every grid point with p > 0 (default; shows full cloud).
#   'top'      — scatter of the top X% of probability mass (top= sets X).
#   'surface'  — convex-hull isosurface of the top X% region (cleanest).
# ══════════════════════════════════════════════════════════════════════════════

pl = make_plotter(pf)
pred.name = 'nonzero'
pl.plot_prediction_results(pred, plot_average_known=True)
pred.name = 'top 50%'
pl.plot_prediction_results(pred, p_mode='top', top=0.5, plot_samples=False, visible='legendonly')
pred.name = 'surface 50%'
pl.plot_prediction_results(pred, p_mode='surface', top=0.5, plot_samples=False, visible='legendonly')
pred.name = 's1'   # restore
pl.show(title="2. Display modes — toggle in legend")
# ↑ Adjust top= to control how much of the probability cloud is shown.


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DETAIL VIEW — algorithm layers
#
#   a) Support assumption  — probability placed on the FeMn–FeTi–FeCu triangle,
#                            peaking at the supplied mass phase_fractions.
#   b) Projected distribution — rays cast from each support point through
#                               the sample to the far boundary.
#   c) Interpolated distribution — final probability density in phase field space.
#   d) Surface overlay — convex-hull isosurface of the top 50% region.
# ══════════════════════════════════════════════════════════════════════════════

s_detail = Sample("s1", "Fe2Mn2Ti2Cu4")
s_detail.add_knowns(["FeMn", "FeTi", "FeCu"])
s_detail.add_mass_weights([0.25, 0.35, 0.40])
s_detail.set_predicted_error(0.3)

picip_d = PICIP(pf)
picip_d.add_sample(s_detail)
# version='dirichlet' so all the layers below are populated — see the note in
# tutorial_2d.py section 2.
pred_d = picip_d.run(version='dirichlet', n_l=50, n_p=10)

pl = make_plotter(pf)
pl.plot_detail_view(pred_d, top_mode='surface', top=0.5)
pl.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EFFECT OF predicted_error
#     Low (0.05) = trust the phase_fractions precisely.  High (0.6) = phase_fractions uncertain.
#     Typical starting range: 0.1–0.3.
# ══════════════════════════════════════════════════════════════════════════════

errors = [(0.05, "very sharp"), (0.3, "default"), (0.6, "broad")]
preds_err = []
for err, label in errors:
    s = Sample("s1", "Fe2Mn2Ti2Cu4")
    s.add_knowns(["FeMn", "FeTi", "FeCu"])
    s.add_mass_weights([0.25, 0.35, 0.40])
    s.set_predicted_error(err)
    p = PICIP(pf)
    p.add_sample(s)
    preds_err.append(p.run(n_l=50, n_p=10, name=f"σ={err}"))

pl = make_plotter(pf)
pl.plot_prediction_results(preds_err[0], plot_average_known=True)
pl.plot_prediction_results(preds_err[1], plot_samples=False, visible='legendonly')
pl.plot_prediction_results(preds_err[2], plot_samples=False, visible='legendonly')
pl.show(title="4. Effect of predicted_error — toggle in legend")
# ↑ σ=0.05 on load (tight cloud).  Toggle σ=0.3 and σ=0.6 in the legend.
#   Higher σ = phase_fractions less trusted = broader probability cloud.


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MULTIPLE SAMPLES
#     Each sample has an associated probability density.  When multiple samples
#     are combined, their densities are multiplied — the result is the region
#     consistent with all samples simultaneously, which is typically far narrower
#     than any individual prediction.
#     Rays from samples with different known-phase ratios cross at the unknown.
# ══════════════════════════════════════════════════════════════════════════════

# s_c (30:30:40) and s_d (15:55:30) are constructed so both rays
# converge on TiCu₃.  K1=Fe₃Mn, K2=FeTi₃, K3=MnCu.

s_c = Sample("c", Composition({"Fe": 0.180, "Mn": 0.165, "Ti": 0.235, "Cu": 0.420}))
s_c.add_knowns(["Fe3Mn", "FeTi3", "MnCu"])
s_c.add_mass_weights([0.30, 0.30, 0.40])
s_c.set_predicted_error(0.3)

s_d = Sample("d", Composition({"Fe": 0.150, "Mn": 0.112, "Ti": 0.348, "Cu": 0.390}))
s_d.add_knowns(["Fe3Mn", "FeTi3", "MnCu"])
s_d.add_mass_weights([0.15, 0.55, 0.30])
s_d.set_predicted_error(0.3)

picip_multi = PICIP(pf)
picip_multi.add_sample(s_c)
picip_multi.add_sample(s_d)

pred_both = picip_multi.run(n_l=50, n_p=10)   # combined probability density

pl = make_plotter(pf)
pl.plot_prediction_results(pred_both, plot_average_known=True)
pl.show()
# ↑ The combined cloud concentrates near TiCu₃, where the two probability clouds intersect.


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SUGGESTIONS — what to synthesise next
#     Two batch-selection methods, shown side by side below.
#
#   a) suggest()          — the routine method.  The first point is the
#                           probability-weighted mean; the rest are drawn at
#                           random from the density, with every grid point
#                           within min_dist of an already-chosen point excluded
#                           before each draw.  Stochastic: only the 'mean' row
#                           repeats between runs.
#
#   b) suggest_kmedoids() — greedy weighted k-medoids over the grid.  Each pick
#                           minimises the posterior-weighted expected distance
#                           to the nearest already-chosen point, so the batch is
#                           chosen as a whole rather than point by point.
#                           Deterministic, but less tested — see PICIP_manual.md §7.
# ══════════════════════════════════════════════════════════════════════════════

picip_s = PICIP(pf)
picip_s.add_sample(s_detail)
pred_s = picip_s.run(n_l=50, n_p=10)

# ── 6a. Default method ────────────────────────────────────────────────────────
suggestions = picip_s.suggest(pred_s, n=5, min_dist=0.05)

print("\n6a. suggest() — probability-weighted sampling:")
suggestions.print_table()

suggestions.save("../output/tutorial_suggestions_3d.csv")

pl = make_plotter(pf)
pl.plot_prediction_results(pred_s, plot_average_known=True)
pl.show(title="6a. suggest() — probability-weighted sampling",
        save="../output/tutorial_output_3d")
# ↑ Suggestions are auto-plotted.  save= writes an interactive HTML.

# ── 6b. k-medoids alternative ─────────────────────────────────────────────────
# Overwrites pred_s.suggestions, so 6a is plotted before this runs.
suggestions_km = picip_s.suggest_kmedoids(pred_s, n=5)

print("\n6b. suggest_kmedoids() — greedy weighted k-medoids:")
suggestions_km.print_table()

suggestions_km.save("../output/tutorial_suggestions_3d_kmedoids.csv")

pl = make_plotter(pf)
pl.plot_prediction_results(pred_s, plot_average_known=True)
pl.show(title="6b. suggest_kmedoids() — deterministic batch",
        save="../output/tutorial_output_3d_kmedoids")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CHARGED PHASE FIELDS — the same workflow, charge-constrained
#     Everything above used setup_uncharged.  Real oxide and halide systems are
#     usually better described by setup_charged, which keeps only the
#     charge-neutral compositions.
#
#     >>> tutorial_setup.py covers all four setup methods in full:
#         setup_uncharged / setup_charged / setup_charge_ranges / setup_custom,
#         plus n_points, precursor restriction, and post-setup modifiers. <<<
# ══════════════════════════════════════════════════════════════════════════════

# ── What changes when you switch setup method ─────────────────────────────────
#
#   DIMENSIONALITY.  This is the thing that catches people out.  Every setup
#   applies the sum-to-1 constraint; setup_charged adds one more:
#
#       setup_uncharged(n elements)  →  constrained_dim = n - 1
#       setup_charged(n elements)    →  constrained_dim = n - 2
#
#   So to stay 3-D you need FIVE elements once charges are on, not the four
#   used in sections 1-6.  Fewer elements would drop the field below 3-D and
#   take you out of this tutorial's scope — tutorial_2d.py covers 2-D.
#
#   COMPOSITIONS.  The sample and every known phase must be charge neutral
#   under the charges you declared, or add_sample raises.  Nothing else about
#   Sample changes.
#
#   NUMBER OF KNOWNS FOR version='dirichlet'.  The Dirichlet method needs
#   (constrained_dim) knowns, so 3 knowns in a 3-D field — set by the
#   dimensionality, not by the setup method.  Three knowns work here for the
#   same reason they did in sections 1-6.
#
#   EVERYTHING ELSE IS IDENTICAL.  run(), suggest(), make_plotter() and the
#   plotting calls are unchanged — the field is still 3-D, so make_plotter
#   still returns a Cube.
#
#   MIXED VALENCE.  If an element takes more than one oxidation state, use
#   setup_charge_ranges instead.  It grids like setup_uncharged and filters
#   afterwards, so it gives n_elements - 1 dimensions — 4 elements for 3-D.
#   Mixed-valence elements take a [min_charge, max_charge] PAIR, not a list of
#   every allowed state; fixed-charge elements take a scalar:
#       pf.setup_charge_ranges({"Fe": [+2, +3], "Mn": [+2, +4], "Ti": +4, "O": -2})
#   See tutorial_setup.py section 3.
# ──────────────────────────────────────────────────────────────────────────────

pf_charged = Phase_Field()
pf_charged.setup_charged(
    {"Li": +1, "Ni": +3, "Mn": +4, "Co": +3, "O": -2},   # 5 elements + charge → 3-D
    n_points=900,
)
print(f"\n7. Charged Li-Ni-Mn-Co-O: constrained_dim={pf_charged.constrained_dim}, "
      f"{len(pf_charged.omega)} grid points")

# Charge-neutral sample and knowns.  Check the arithmetic if add_sample complains:
#   LiNiO2   → 1(+1) + 1(+3) + 2(-2) = 0
#   Li2MnO3  → 2(+1) + 1(+4) + 3(-2) = 0
sample_c = Sample("s1", "Li4NiCoMnO7")
sample_c.add_knowns(["LiNiO2", "LiCoO2", "Li2MnO3"])   # 3 knowns in 3-D → Dirichlet available
sample_c.add_mass_weights([0.35, 0.35, 0.30])
sample_c.set_predicted_error(0.3)

picip_c = PICIP(pf_charged)
picip_c.add_sample(sample_c)

pred_c = picip_c.run(n_l=50, n_p=10)                       # Gaussian (default)
pred_c_dir = picip_c.run(version='dirichlet', n_l=50, n_p=10,      # Dirichlet: 3-D + 3 knowns
                         name='Dirichlet')

pl = make_plotter(pf_charged)     # returns a Cube — 3-D, same as sections 1-6
pl.plot_prediction_results(pred_c, plot_average_known=True, p_mode='surface', top=0.5)
pl.show(title="7. setup_charged — Li-Ni-Mn-Co-O (charge neutral, 3-D)",
        save="../output/tutorial_charged_3d")
