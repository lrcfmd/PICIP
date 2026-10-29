"""
PICIP quick start — uncharged phase field
=========================================
Fe-Mn-Ti with two known phases.

Both batch-selection methods are shown: suggest() (probability-weighted
sampling, stochastic) and suggest_kmedoids() (deterministic batch).

Run with:  python quickstart_uncharged.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space — 3 elements, no constraint → 2-D triangle.
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

# 2. The measured sample: overall composition, identified phases, mass fractions.
sample = Sample("s1", "Fe2Mn4Ti4")
sample.add_knowns(["FeMn", "FeTi"])
sample.add_mass_weights([0.3, 0.7])     # e.g. from Rietveld refinement
sample.set_predicted_error(0.3)         # 0 = precise weights, 1 = rough

# 3. Infer where the unknown phase sits.
picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run(version='dirichlet')   # 2-D + 2 knowns → Dirichlet supported

# 4a. Suggestions — probability-weighted sampling.
suggestions = picip.suggest(pred, n=5, min_dist=0.05)
print("\nsuggest() — probability-weighted sampling:")
suggestions.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Uncharged Fe-Mn-Ti — suggest()")

# 4b. Suggestions — deterministic k-medoids batch.
#     Overwrites pred.suggestions, so plot 4a before running this.
suggestions_km = picip.suggest_kmedoids(pred, n=5)
print("\nsuggest_kmedoids() — deterministic batch:")
suggestions_km.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Uncharged Fe-Mn-Ti — suggest_kmedoids()")
