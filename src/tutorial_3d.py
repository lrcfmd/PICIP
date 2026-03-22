"""
PICIP Tutorial — 3-D Phase Fields
===================================
Sections:  1. Quick start  2. Display modes  3. Detail view
           4. predicted_error  5. Multiple samples  6. Suggestions
Run top to bottom — each section opens a browser figure.

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md
  Setup tutorial: tutorial_setup.py
  2-D tutorial  : tutorial_2d.py
"""

from setup_phase_field import Phase_Field
from run_PICIP import PICIP, Sample
from visualise_cube import make_plotter
from pymatgen.core import Composition


# Phase_Field defines the composition space PICIP operates over.
# setup_uncharged: grid of all compositions summing to 1, no charge constraint.
# 4 elements → 3-D tetrahedron (Cube).  See tutorial_setup.py for all setup options.
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])


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
sample.set_predicted_error(0.3)                      # uncertainty in the weights (0=precise, 1=rough)

# Initialise PICIP and add the sample
picip = PICIP(pf)
picip.add_sample(sample)

# Run inference — returns a Prediction object
pred = picip.run(
    n_l=50,   # points along the known simplex (more = smoother, diminishing returns >100)
    n_p=10,   # points per projected ray (10–50 sufficient; use more for sharper densities)
)

# Get suggested next compositions to synthesise
suggestions = picip.suggest(
    pred,
    n=5,          # number of suggestions to return
    min_dist=0.05 # minimum spacing between suggestions in constrained-basis coordinates
)

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
#                            peaking at the supplied mass weights.
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
pred_d = picip_d.run(n_l=50, n_p=10)

pl = make_plotter(pf)
pl.plot_detail_view(pred_d, top_mode='surface', top=0.5)
pl.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EFFECT OF predicted_error
#     Low (0.05) = trust the weights precisely.  High (0.6) = weights uncertain.
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
#   Higher σ = weights less trusted = broader probability cloud.


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
#     suggest(pred, n) returns n compositions sampled from the probability density.
#     First = probability-weighted mean.  Remaining are sampled at high-density points.
#     min_dist: minimum separation between suggestions (in constrained-basis coords,
#               where 1.0 spans the full phase field).  Prevents clustered suggestions.
# ══════════════════════════════════════════════════════════════════════════════

picip_s = PICIP(pf)
picip_s.add_sample(s_detail)
pred_s = picip_s.run(n_l=50, n_p=10)

suggestions = picip_s.suggest(pred_s, n=5, min_dist=0.05)

print("\n6. Suggested next compositions:")
for label, row in zip(suggestions.labels, suggestions.standard):
    print(f"  {label:8s}  Fe={row[0]:.3f}  Mn={row[1]:.3f}  Ti={row[2]:.3f}  Cu={row[3]:.3f}")

suggestions.save("../output/tutorial_suggestions_3d.csv")

pl = make_plotter(pf)
pl.plot_prediction_results(pred_s, plot_average_known=True)
pl.show(save="../output/tutorial_output_3d")
# ↑ Suggestions are auto-plotted.  save= writes an interactive HTML.
