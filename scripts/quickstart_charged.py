"""
PICIP quick start — charged phase field (fixed oxidation states)
================================================================
Ba-Y-Si-O with two known phases.  setup_charged adds charge neutrality on top
of the sum-to-one constraint, so four elements give a 2-D field.

Every composition must be charge neutral under the declared charges.  PICIP
warns if one is not, and projects it onto the field/
Run with:  python quickstart_charged.py
"""

from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space — 4 elements + charge neutrality → 2-D.
pf = Phase_Field()
pf.setup_charged({"Ba": +2, "Y": +3, "Si": +4, "O": -2}, n_points=600)

# 2. The measured sample.
sample = Sample("s1", "Ba33Y24Si42O153")
sample.add_knowns(["Ba9Y2Si6O24", "BaSiO3"])
sample.add_mass_weights([0.6, 0.4])
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
pl.show(title="Charged Ba-Y-Si-O — suggest()")

# 4b. Suggestions — deterministic k-medoids batch.
#     Overwrites pred.suggestions, so plot 4a before running this.
suggestions_km = picip.suggest_kmedoids(pred, n=5)
print("\nsuggest_kmedoids() — deterministic batch:")
suggestions_km.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Charged Ba-Y-Si-O — suggest_kmedoids()")
