"""
PICIP Tutorial — 2-D Phase Fields
===================================
Sections:  1. Quick start  2. Detail view  3. predicted_error
           4. Multiple samples  5. Resolution  6. Suggestions
           7. Charged phase fields (setup_charged)
Run top to bottom — each section opens a browser figure.

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md
  Setup tutorial: tutorial_setup.py
  3-D tutorial  : tutorial_3d.py
  Evaluation    : tutorial_evaluation.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter
from pymatgen.core import Composition


# Phase_Field defines the composition space PICIP operates over.
# setup_uncharged: grid of all compositions summing to 1, no charge constraint.
# 3 elements → 2-D triangle — make_plotter returns a Square automatically.
#
# >>> See `tutorial_setup.py` for the full menu: setup_uncharged / setup_charged /
#     setup_charge_ranges / setup_custom, n_points/resolution, precursor restriction.
#     Section 7 below runs this whole workflow again on a CHARGED field and spells
#     out exactly what has to change. <<<
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

# ── Other 2-D phase fields you could use instead (uncomment to try) ───────────
#   Every option below is 2-D, so the rest of this tutorial runs unchanged.
#   The rule is: setup_uncharged and setup_charge_ranges give n_elements - 1
#   dimensions; setup_charged adds a second constraint and gives n_elements - 2.
#   For 3-D fields see tutorial_3d.py, and tutorial_setup.py for the full menu.
#
# # Charged system, fixed oxidation states — 4 elements → 2-D.  Section 7:
# pf.setup_charged({"Ba": +2, "Y": +3, "Si": +4, "O": -2})
#
# # Variable oxidation states — 3 elements → 2-D.  Mixed-valence elements take
# # a [min_charge, max_charge] pair; fixed ones take a scalar:
# pf.setup_charge_ranges({"Fe": [+2, +3], "Mn": [+2, +4], "O": -2})
#
# # Restrict the field to compositions accessible from a specific set of precursors:
# pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["FeO", "MnO2", "TiO2"])
#
# # Coarser/finer grid:
# pf.setup_uncharged(["Fe", "Mn", "Ti"], n_points=200)   # faster, lower resolution
# pf.setup_uncharged(["Fe", "Mn", "Ti"], n_points=2000)  # slower, higher resolution
# ───────────────────────────────────────────────────────────────────────────────


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
sample.set_predicted_error(0.3)              # uncertainty in the phase_fractions (0=precise, 1=rough)

# Initialise PICIP and add the sample
picip = PICIP(pf)
picip.add_sample(sample)

# Run inference — returns a Prediction object.
# version='gaussian' is the default: the original PICIP algorithm from the
# paper, and the well-tested path for any number of knowns in any dimension.
pred = picip.run(
    n_l=50,   # number of ray directions (more = smoother, diminishing returns >100)
    n_p=50,   # points per projected ray (20–50 sufficient)
)

# ── Opting into the Dirichlet method ──────────────────────────────────────────
#
# version='dirichlet' selects the Dirichlet method instead.  Rather than collapsing the
# knowns to their centroid, it puts a Dirichlet distribution over the mixing
# ratio of the known phases and projects that through the sample — so it uses
# the measured mass fractions as more than a weighted average.
#
# CAVEAT: it is implemented for exactly two configurations —
#     2-D constrained space with exactly 2 knowns   (this sample)
#     3-D constrained space with exactly 3 knowns   (see tutorial_3d.py)
# Ask for it anywhere else — 1 known, 3 knowns in 2-D, 2 or 4 knowns in 3-D —
# and PICIP warns that the method is untested there and falls back to Gaussian,
# so you still get a result, just not the one you asked for.  Watch for:
#
#   RuntimeWarning: The Dirichlet method (version='dirichlet') is implemented only for
#   2-D with 2 known phases and 3-D with 3 known phases ... falling back to the
#   Gaussian method.
#
# This sample has 2 knowns in a 2-D field, so it genuinely takes the Dirichlet
# path.  Everything downstream — plotting, suggestions, multiple samples — works
# identically; swap `pred_dirichlet` for `pred` below to compare them.
pred_dirichlet = picip.run(version='dirichlet', n_l=50, n_p=50, name='Dirichlet')
# ──────────────────────────────────────────────────────────────────────────────

# Get suggested next compositions to synthesise
suggestions = picip.suggest(
    pred,
    n=5,          # number of suggestions to return
    min_dist=0.05 # minimum spacing between suggestions in constrained-basis coordinates
)

# Experimental alternative — Bayes-optimal batch for min-distance-to-truth.
# Less tested than `suggest()`; see PICIP_manual.md §7.
# suggestions = picip.suggest_kmedoids(pred, n=5)

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DETAIL VIEW — algorithm layers
#
#   a) Support assumption  — probability placed on the FeMn–FeTi line,
#                            peaking at the supplied mass phase_fractions.
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
# version='dirichlet' so all three layers below are populated: the Gaussian
# default collapses the knowns to their centroid, so it has no support layer
# and its rays fan out from the sample rather than from the known simplex.
pred_d = picip_d.run(version='dirichlet')

pl = make_plotter(pf)
pl.plot_detail_view(pred_d)
pl.show()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  EFFECT OF predicted_error
#     Low (0.05) = trust the phase_fractions precisely.  High (0.6) = phase_fractions uncertain.
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

#   Sample a and its average known are drawn in one colour, sample b and its
#   average known in another.  The b markers start hidden so the figure opens
#   on sample a alone; toggle them in the legend to compare.  plot_samples is
#   left off in all three calls so the markers below are the only ones drawn —
#   otherwise plot_prediction_results adds its own white/amber pair per call.

C_A, C_B = '#4C9BE8', '#E8556B'      # sample a, sample b

pl = make_plotter(pf)
pl.plot_prediction_results(pred_a, plot_samples=False)
pl.plot_prediction_results(pred_b, plot_samples=False, visible='legendonly')
pl.plot_prediction_results(pred_both, plot_samples=False, visible='legendonly', plot_individual=False)

for samp, colour, vis in ((s_a, C_A, True), (s_b, C_B, 'legendonly')):
    pl.plot_points([samp.comp_constrained], color=colour, name=f'sample {samp.name}',
                   custom_labels=[f"<b>{samp.name}</b>"], s=14,
                   border_color='#222222', border_width=2, visible=vis)
    pl.plot_points([samp.average_k], color=colour, name=f'avg. known {samp.name}',
                   custom_labels=[f"<b>{samp.name}</b><br>avg. known"], s=14,
                   marker='cross', border_color='#222222', border_width=2, visible=vis)

pl.show(title="4. Multiple samples — toggle a, b, combined in legend")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESOLUTION — n_l and n_p
#     n_l : points along the known simplex.  50 is recommended; >100 = no gain.
#     n_p : points per projected ray for griddata.  20–50 is sufficient.
# ══════════════════════════════════════════════════════════════════════════════

def _make_res_sample():
    s = Sample("s1", "Fe2Mn4Ti4")
    s.add_knowns(["FeMn", "FeTi"])
    s.add_mass_weights([0.3, 0.7])
    s.set_predicted_error(0.1)    # sharp density makes banding easier to see
    return s

def _run_nl(nl):
    p = PICIP(pf)
    p.add_sample(_make_res_sample())
    return p.run(version='dirichlet', n_l=nl, n_p=50, name=f"n_l={nl}")

pred_nl5   = _run_nl(5)
pred_nl20  = _run_nl(20)
pred_nl50  = _run_nl(50)
pred_nl200 = _run_nl(200)

pl = make_plotter(pf)
pl.plot_detail_view(pred_nl5)
pl.show(title="5a. n_l=5 — banding: 5 support points, 5 rays")

pl = make_plotter(pf)
pl.plot_detail_view(pred_nl50)
pl.show(title="5b. n_l=50 — smooth, no banding (recommended default)")

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
#     min_dist: minimum separation between suggestions (in constrained-basis
#               coords).  Prevents clustered suggestions.
# ══════════════════════════════════════════════════════════════════════════════

picip_s = PICIP(pf)
picip_s.add_sample(s_detail)
pred_s = picip_s.run()

suggestions = picip_s.suggest(pred_s, n=5, min_dist=0.05)

print("\n6. Suggested next compositions:")
suggestions.print_table()

suggestions.save("../output/tutorial_suggestions.csv")

pl = make_plotter(pf)
pl.plot_prediction_results(pred_s, plot_average_known=True)
pl.show(title="6. Suggestions overlay", save="../output/tutorial_suggestions")
# ↑ Suggestions are auto-plotted.  save= writes an interactive HTML.


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
#   So this section's 4-element charged system is 2-D, exactly like the
#   3-element uncharged system used above — same Square plotter, same code.
#   (setup_charged with only 3 elements would be 1-D, which PICIP rejects.)
#
#   COMPOSITIONS.  The sample and every known phase must be charge neutral
#   under the charges you declared.  add_knowns warns about a known that is
#   not, but nothing checks the sample composition — get that arithmetic wrong
#   and you get a result that looks fine and means nothing.  Nothing else about
#   Sample changes.
#
#   NUMBER OF KNOWNS FOR version='dirichlet'.  The Dirichlet method needs
#   (constrained_dim) knowns — 2 in 2-D, 3 in 3-D — so what counts as a
#   supported configuration is set by the dimensionality, not by the setup
#   method.  Two knowns work here for the same reason they did above.
#
#   EVERYTHING ELSE IS IDENTICAL.  run(), suggest(), make_plotter() and the
#   plotting calls are unchanged — make_plotter picks Square or Cube from
#   constrained_dim on its own.
# ──────────────────────────────────────────────────────────────────────────────

pf_charged = Phase_Field()
pf_charged.setup_charged(
    {"Ba": +2, "Y": +3, "Si": +4, "O": -2},   # 4 elements + charge → 2-D
    n_points=600,
)
print(f"\n7. Charged Ba-Y-Si-O: constrained_dim={pf_charged.constrained_dim}, "
      f"{len(pf_charged.omega)} grid points")

# Charge-neutral sample and knowns.  PICIP does not check this for you, so
# verify the arithmetic yourself — a sample off the neutral plane sits outside
# the phase field and will silently give a meaningless result:
#   Ba33Y24Si42O153 → 33(+2) + 24(+3) + 42(+4) + 153(-2) = 66 + 72 + 168 - 306 = 0
#   Ba9Y2Si6O24     →  9(+2) +  2(+3) +  6(+4) +  24(-2) = 18 +  6 +  24 -  48 = 0
sample_c = Sample("s1", "Ba33Y24Si42O153")
sample_c.add_knowns(["Ba9Y2Si6O24", "BaSiO3"])   # 2 knowns in 2-D → Dirichlet available
sample_c.add_mass_weights([0.6, 0.4])
sample_c.set_predicted_error(0.3)

picip_c = PICIP(pf_charged)
picip_c.add_sample(sample_c)

pred_c = picip_c.run(n_l=50, n_p=50)                       # Gaussian (default)
pred_c_dir = picip_c.run(version='dirichlet', n_l=50, n_p=50,      # Dirichlet: 2-D + 2 knowns
                         name='Dirichlet')

pl = make_plotter(pf_charged)     # returns a Square — 2-D, same as sections 1-6
pl.plot_prediction_results(pred_c, plot_average_known=True)
pl.show(title="7. setup_charged — Ba-Y-Si-O (charge neutral, 2-D)",
        save="../output/tutorial_charged_2d")
