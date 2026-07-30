"""
PICIP quick start — custom stoichiometry constraint
===================================================
Li-Fe-Mn-O restricted to spinel stoichiometry: 3 cations to 4 anions.

setup_custom takes any linear equality constraint c . x = 0 as a row of
custom_con.  Here 4(Li + Fe + Mn) - 3(O) = 0 enforces cation:anion = 3:4, so
LiFe2O4, LiMn2O4 and Fe3O4 all lie in the field.

Each custom constraint removes one dimension: four elements minus the
sum-to-one constraint minus this one gives a 2-D field, which with two knowns
is a supported Dirichlet configuration.

Run with:  python quickstart_custom_constraint.py
"""

import numpy as np

from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space — 4 elements, 1 custom constraint → 2-D.
#    Row order matches the element order passed to setup_custom.
spinel = np.array([4, 4, 4, -3.0])       # 4(Li + Fe + Mn) = 3(O)

pf = Phase_Field()
pf.setup_custom(["Li", "Fe", "Mn", "O"], custom_con=spinel, n_points=2000)

# 2. The measured sample.  Compositions off the constraint plane are reported
#    by PICIP and projected onto it, so keep the 3:4 ratio exact.  The sample
#    must also lie OFF the line joining the two knowns: if it is exactly a
#    mixture of them there is no unknown phase to solve for, and the Dirichlet
#    method rejects it as degenerate.  (LiFe2O4 + LiMn2O4 = LiFeMnO4 exactly,
#    hence the lower Li content here.)
sample = Sample("s1", "LiFe3Mn2O8")
sample.add_knowns(["LiFe2O4", "LiMn2O4"])
sample.add_mass_weights([0.4, 0.6])
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
pl.show(title="Spinel Li-Fe-Mn-O — suggest()")

# 4b. Suggestions — deterministic k-medoids batch.
#     Overwrites pred.suggestions, so plot 4a before running this.
suggestions_km = picip.suggest_kmedoids(pred, n=5)
print("\nsuggest_kmedoids() — deterministic batch:")
suggestions_km.print_table()

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(title="Spinel Li-Fe-Mn-O — suggest_kmedoids()")
