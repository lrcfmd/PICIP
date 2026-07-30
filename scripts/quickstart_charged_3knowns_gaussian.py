"""
PICIP quick start — three knowns in 2-D, Gaussian method
========================================================
The same charged Ba-Y-Si-O field as quickstart_charged.py, but with three
known phases instead of two.

This is deliberately *not* a Dirichlet configuration.  Dirichlet needs the
known simplex to sit one dimension below the phase field — 2-D + 2 knowns, or
3-D + 3 knowns.  Three knowns in a 2-D field make a triangle of the same
dimension as the field, so many mixing ratios share one line through the
sample and the construction is not well posed.

The Gaussian method has no such restriction: it collapses the knowns to their
mass-weighted centroid and spreads probability around the line through it.  It
is the default, so no version argument is needed.  (Asking for
version='dirichlet' here would warn and fall back to exactly this.)

Run with:  python quickstart_charged_3knowns_gaussian.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space — 4 elements + charge neutrality → 2-D.
pf = Phase_Field()
pf.setup_charged({"Ba": +2, "Y": +3, "Si": +4, "O": -2}, n_points=600)

# 2. The measured sample, with three charge-neutral knowns.
#    Y2SiO5 → 2(+3) + (+4) + 5(-2) = 6 + 4 - 10 = 0
sample = Sample("s1", "Ba33Y24Si42O153")
sample.add_knowns(["Ba9Y2Si6O24", "BaSiO3", "Y2SiO5"])
sample.add_mass_weights([0.45, 0.35, 0.20])
sample.set_predicted_error(0.3)

# 3. Infer where the unknown phase sits — Gaussian is the default.
picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run()

# 4. Suggestions — probability-weighted sampling.
suggestions = picip.suggest(pred, n=5, min_dist=0.05)
print("\nsuggest() — probability-weighted sampling:")
suggestions.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Charged Ba-Y-Si-O, 3 knowns — Gaussian method")
