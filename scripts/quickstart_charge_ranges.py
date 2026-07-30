"""
PICIP quick start — mixed-valence phase field (charge ranges)
=============================================================
Fe-Mn-O where Fe can be +2/+3 and Mn +2/+4.  Elements with a variable
oxidation state take a [min, max] pair; fixed ones take a scalar.

There is no charge-neutrality *equality* constraint here — the grid is built as
if uncharged (so the constrained dimension is N - 1, giving 2-D for three
elements) and then filtered to compositions that can be balanced by some
allowed combination of oxidation states.  With two knowns this is a supported
Dirichlet configuration.

Run with:  python quickstart_charge_ranges.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space — 3 elements → 2-D, then filtered by charge range.
pf = Phase_Field()
pf.setup_charge_ranges({"Fe": [+2, +3], "Mn": [+2, +4], "O": -2}, n_points=2000)

# 2. The measured sample.  Each composition must be balanceable within the
#    allowed ranges — PICIP warns if it is not.  The sample must also lie OFF
#    the line joining the two knowns: if it is exactly a mixture of them there
#    is no unknown phase to solve for, and the Dirichlet method rejects it as
#    degenerate.  (Fe2O3 + MnO2 = Fe2MnO5 exactly, hence Fe3MnO5 here.)
sample = Sample("s1", "Fe3MnO5")
sample.add_knowns(["Fe2O3", "MnO2"])
sample.add_mass_weights([0.55, 0.45])
sample.set_predicted_error(0.3)

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
pl.show(title="Mixed-valence Fe-Mn-O — suggest()")

# 4b. Suggestions — deterministic k-medoids batch.
#     Overwrites pred.suggestions, so plot 4a before running this.
suggestions_km = picip.suggest_kmedoids(pred, n=5)
print("\nsuggest_kmedoids() — deterministic batch:")
suggestions_km.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Mixed-valence Fe-Mn-O — suggest_kmedoids()")
