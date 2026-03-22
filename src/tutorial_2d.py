"""
PICIP Tutorial — 2-D Phase Fields
===================================
Sections:  1. Quick start  2. Detail view  3. predicted_error
           4. Multiple samples  5. Resolution  6. Suggestions
Run top to bottom — each section opens a browser figure.

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md
  Setup tutorial: tutorial_setup.py
  3-D tutorial  : tutorial_3d.py
"""

from setup_phase_field import Phase_Field
from run_PICIP import PICIP, Sample
from visualise_cube import make_plotter
from pymatgen.core import Composition


# Phase_Field defines the composition space PICIP operates over.
# setup_uncharged: grid of all compositions summing to 1, no charge constraint.
# 3 elements → 2-D triangle (Square).  See tutorial_setup.py for all setup options.
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])


# ══════════════════════════════════════════════════════════════════════════════
# 1.  QUICK START
# ══════════════════════════════════════════════════════════════════════════════

# Create sample: measured composition + known co-existing phases + their mass fractions.
# add_mass_weights takes mass fractions as returned by Rietveld refinement.
# Use add_molar_weights instead if you have molar fractions.
sample = Sample(
    "s1",           # name — label used in plots and titles
    "Fe2Mn4Ti4",    # composition — measured sample (pymatgen formula string or Composition)
)
sample.add_knowns(["FeMn", "FeTi"])          # known co-existing phases
sample.add_mass_weights([0.3, 0.7])          # mass fractions of each known phase (e.g. from Rietveld)
sample.set_predicted_error(0.3)              # uncertainty in the weights (0=precise, 1=rough)

# Initialise PICIP and add the sample
picip = PICIP(pf)
picip.add_sample(sample)

# Run inference — returns a Prediction object
pred = picip.run(
    n_l=50,   # points along the known simplex (more = smoother, diminishing returns >100)
    n_p=50,   # points per projected ray (20–50 sufficient)
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
# 2.  DETAIL VIEW — algorithm layers
#
#   a) Support assumption  — probability placed on the FeMn–FeTi line,
#                            peaking at the supplied mass weights.
#   b) Projected distribution — rays cast from each support point through
#                               the sample to the far boundary.
#   c) Interpolated distribution — final probability density in phase field space.
# ══════════════════════════════════════════════════════════════════════════════

s_detail = Sample("s1", "Fe2Mn4Ti4")
s_detail.add_knowns(["FeMn", "FeTi"])
s_detail.add_mass_weights([0.3, 0.7])
s_detail.set_predicted_error(0.3)

picip_d = PICIP(pf)
picip_d.add_sample(s_detail)
pred_d = picip_d.run()

pl = make_plotter(pf)
pl.plot_detail_view(pred_d)
pl.show()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  EFFECT OF predicted_error
#     Low (0.05) = trust the weights precisely.  High (0.6) = weights uncertain.
#     Typical starting range: 0.1–0.3.
# ══════════════════════════════════════════════════════════════════════════════

errors = [(0.05, "very sharp"), (0.3, "default"), (0.6, "broad")]
preds_err = []
for err, label in errors:
    s = Sample(f"σ={err}", "Fe2Mn4Ti4")
    s.add_knowns(["FeMn", "FeTi"])
    s.add_mass_weights([0.3, 0.7])
    s.set_predicted_error(err)
    p = PICIP(pf)
    p.add_sample(s)
    preds_err.append(p.run())

pl = make_plotter(pf)
pl.plot_prediction_results(preds_err[0], plot_average_known=True)
pl.plot_prediction_results(preds_err[1], plot_samples=False, visible='legendonly')
pl.plot_prediction_results(preds_err[2], plot_samples=False, visible='legendonly')
pl.show(title="3. Effect of predicted_error — toggle in legend")
# ↑ σ=0.05 on load (tight spike).  Toggle σ=0.3 and σ=0.6 in the legend.


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MULTIPLE SAMPLES
#     Each sample has an associated probability density.  When multiple samples
#     are combined, their densities are multiplied — the result is the region
#     consistent with all samples simultaneously, which is typically far narrower
#     than any individual prediction.
# ══════════════════════════════════════════════════════════════════════════════

s_a = Sample("a", Composition({"Fe": 0.300, "Mn": 0.357, "Ti": 0.343}))
s_a.add_knowns(["FeMn", "FeTi"])
s_a.add_mass_weights([0.3, 0.7])
s_a.set_predicted_error(0.3)

s_b = Sample("b", Composition({"Fe": 0.300, "Mn": 0.477, "Ti": 0.223}))
s_b.add_knowns(["FeMn", "FeTi"])
s_b.add_mass_weights([0.7, 0.3])
s_b.set_predicted_error(0.3)

picip_multi = PICIP(pf)
picip_multi.add_sample(s_a)
picip_multi.add_sample(s_b)

pred_a    = picip_multi.run(0)   # sample a only
pred_b    = picip_multi.run(1)   # sample b only
pred_both = picip_multi.run()    # combined probability density

pl = make_plotter(pf)
pl.plot_prediction_results(pred_a, plot_average_known=True)
pl.plot_prediction_results(pred_b, plot_samples=False, visible='legendonly')
pl.plot_prediction_results(pred_both, plot_samples=False, visible='legendonly', plot_individual=False)
pl.show(title="4. Multiple samples — toggle a, b, combined in legend")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESOLUTION — n_l and n_p
#     n_l : points along the known simplex.  50 is recommended; >100 = no gain.
#     n_p : points per projected ray for griddata.  20–50 is sufficient.
# ══════════════════════════════════════════════════════════════════════════════

def _make_res_sample():
    s = Sample("res", "Fe2Mn4Ti4")
    s.add_knowns(["FeMn", "FeTi"])
    s.add_mass_weights([0.3, 0.7])
    s.set_predicted_error(0.1)    # sharp density makes banding easier to see
    return s

def _run_nl(nl):
    p = PICIP(pf)
    p.add_sample(_make_res_sample())
    return p.run(n_l=nl, n_p=50, name=f"n_l={nl}")

pred_nl5   = _run_nl(5)
pred_nl20  = _run_nl(20)
pred_nl50  = _run_nl(50)
pred_nl200 = _run_nl(200)

pl = make_plotter(pf)
pl.plot_detail_view(pred_nl5)
pl.show()
# ↑ n_l=5: clear banding — 5 distinct dots on the simplex, 5 rays.

pl = make_plotter(pf)
pl.plot_detail_view(pred_nl50)
pl.show()
# ↑ n_l=50: smooth, no banding.

pl = make_plotter(pf)
pl.plot_prediction_results(pred_nl5)
pl.plot_prediction_results(pred_nl20,  plot_samples=False, visible='legendonly')
pl.plot_prediction_results(pred_nl50,  plot_samples=False, visible='legendonly')
pl.plot_prediction_results(pred_nl200, plot_samples=False, visible='legendonly')
pl.show(title="5. Effect of n_l — toggle in legend")
# ↑ n_l=5 on load.  n_l=50 is smooth — recommended default.


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SUGGESTIONS — what to synthesise next
#     suggest(pred, n) returns n compositions sampled from the probability density.
#     First = probability-weighted mean.  Remaining are sampled at high-density points.
#     min_dist: minimum separation between suggestions (in constrained-basis coords,
#               where 1.0 spans the full phase field).  Prevents clustered suggestions.
# ══════════════════════════════════════════════════════════════════════════════

picip_s = PICIP(pf)
picip_s.add_sample(s_detail)
pred_s = picip_s.run()

suggestions = picip_s.suggest(pred_s, n=5, min_dist=0.05)

print("\n6. Suggested next compositions:")
for label, row in zip(suggestions.labels, suggestions.standard):
    print(f"  {label:8s}  Fe={row[0]:.3f}  Mn={row[1]:.3f}  Ti={row[2]:.3f}")

suggestions.save("../output/tutorial_suggestions.csv")

pl = make_plotter(pf)
pl.plot_prediction_results(pred_s, plot_average_known=True)
pl.show(title="6. Suggestions overlay", save="../output/tutorial_suggestions")
# ↑ Suggestions are auto-plotted.  save= writes an interactive HTML.
