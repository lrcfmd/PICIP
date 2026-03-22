"""
PICIP Tutorial — Phase Field Setup
===================================
Probabilistic Isolation of Inorganic Crystalline Phases

This file covers everything to do with building the phase field:
the composition space that PICIP operates over.

Run from top to bottom.  Each section opens a figure in your browser
so you can see the result of each setup option as you read about it.

Sections
--------
  1.  setup_uncharged      — no charge constraint, any composition
  2.  setup_charged        — fixed formal charges, charge-neutral only
  3.  setup_charge_ranges  — mixed-valence species
  4.  setup_custom         — your own linear constraint  (spinel example)
  5.  Post-setup modifiers — minimum amounts, precursors (cut + dim-reduce)

Documentation
-------------
  Installation : ../PICIP_installation.md
  User manual  : ../PICIP_manual.md
  2-D tutorial : tutorial_2d.py
  3-D tutorial : tutorial_3d.py
"""

from setup_phase_field import Phase_Field
from visualise_cube import make_plotter
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1.  setup_uncharged
#     Use when there is no charge constraint — every composition that sums
#     to 1 is a valid point.
#
#     The constrained dimension equals (number of elements − 1):
#       3 elements → 2-D triangle    → use Square for plotting
#       4 elements → 3-D tetrahedron → use Cube  for plotting
#
#     n_points controls the grid density (default 500 for 2-D, 1000 for 3-D).
#     Higher → more grid points → smoother probability maps.
# ══════════════════════════════════════════════════════════════════════════════

pf_2d = Phase_Field()
pf_2d.setup_uncharged(["Fe", "Mn", "Ti"])
print(f"1  2-D uncharged (default): {pf_2d.omega.shape[0]} grid points")

pf_3d = Phase_Field()
pf_3d.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])
print(f"1  3-D uncharged (default): {pf_3d.omega.shape[0]} grid points")

pf_fine = Phase_Field()
pf_fine.setup_uncharged(["Fe", "Mn", "Ti"], n_points=2000)
print(f"1  2-D uncharged (n_points=2000): {pf_fine.omega.shape[0]} grid points")

# Show the 2-D field — just corners, edges, and the grid of valid points.
pl = make_plotter(pf_2d)
pl.plot_points(pf_2d.omega, color='#888888', name='grid', s=3)
pl.show(title="1. setup_uncharged — Fe-Mn-Ti  (2-D triangle)")
# ↑ Every grey dot is a composition on the grid that sums to 1.
#   The three corners are the pure elements Fe, Mn, Ti.

pl = make_plotter(pf_3d)
pl.plot_points(pf_3d.omega, color='#888888', name='grid', s=2)
pl.show(title="1. setup_uncharged — Fe-Mn-Ti-Cu  (3-D tetrahedron)")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  setup_charged
#     Use when species have fixed formal charges and only charge-neutral
#     compositions are physically meaningful.
#     Pass a dict of {element: formal_charge}.
#
#     One charge constraint is added on top of sum=1, so:
#       4 elements + charge → 2-D  → Square
#       5 elements + charge → 3-D  → Cube
#
#     Note: 3 elements + charge → 1-D, which PICIP cannot handle.
#     An error is raised if this occurs.
# ══════════════════════════════════════════════════════════════════════════════

pf_charged = Phase_Field()
pf_charged.setup_charged(
    {"Li": 1, "Ni": 3, "Mn": 4, "Co": 3, "O": -2},
)
print(f"2  charged 3-D:  {pf_charged.omega.shape[0]} grid points")

pl = make_plotter(pf_charged)
pl.plot_points(pf_charged.omega, color='#888888', name='grid', s=2)
pl.show(title="2. setup_charged — Li-Ni-Mn-Co-O  (charge neutral, 3-D)")
# ↑ The grid covers a much smaller region than an uncharged field of the
#   same system because the charge-neutrality constraint removes most compositions.
#   n_points ensures the same grid density regardless.


# ══════════════════════════════════════════════════════════════════════════════
# 3.  setup_charge_ranges
#     Use when one or more elements can take a range of formal charges
#     (e.g. Mn can be 3+ or 4+).
#     Pass a list [min_charge, max_charge] for mixed-valence elements.
#
#     The grid is filtered to compositions that are charge-reachable by
#     at least one valid charge assignment within those ranges.
# ══════════════════════════════════════════════════════════════════════════════

pf_ranges = Phase_Field()
pf_ranges.setup_charge_ranges(
    {"Li": 1, "Mn": [3, 4], "O": -2},
)
print(f"3  charge ranges: {pf_ranges.omega.shape[0]} grid points")

pl = make_plotter(pf_ranges)
pl.plot_points(pf_ranges.omega, color='#888888', name='grid', s=5)
pl.show(title="3. setup_charge_ranges — Li-Mn(3+/4+)-O")
# ↑ The valid region is a band: wider than a fixed-charge field because
#   Mn can vary between 3+ and 4+, relaxing the neutrality constraint.


# ══════════════════════════════════════════════════════════════════════════════
# 4.  setup_custom
#     Add your own linear equality constraint on top of sum=1.
#     custom_con is a 2-D array — each row is one constraint vector v
#     where  v · x = 0  for all valid compositions x.
#
#     Example: spinel oxides (M₃O₄) require exactly 3 cations per 4 anions:
#
#         4 × (Li + Fe + Mn) = 3 × O
#         →  v = [4, 4, 4, −3]
#
#     With 4 elements + sum=1 + this constraint → 2-D.
#     Oxygen is fixed at 4/7 ≈ 57 %; you freely explore the cation ratios.
# ══════════════════════════════════════════════════════════════════════════════

pf_spinel = Phase_Field()
pf_spinel.setup_custom(
    ["Li", "Fe", "Mn", "O"],
    custom_con=np.array([[4, 4, 4, -3]]),   # 3 cations : 4 anions
)
print(f"4  spinel custom: {pf_spinel.omega.shape[0]} grid points")

pl = make_plotter(pf_spinel)
pl.plot_points(pf_spinel.omega, color='#888888', name='grid', s=3)
pl.show(title="4. setup_custom — spinel Li-Fe-Mn-O  (O fixed at 4/7)")
# ↑ The three corners are the end-member spinels LiFe₂O₄, LiMn₂O₄, Fe₃O₄.
#   Every dot satisfies the 3:4 cation:anion stoichiometry.


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Post-setup modifiers
#     These can be applied after any of the setups above.
# ══════════════════════════════════════════════════════════════════════════════

# --- 5a. constrain_minimum_amount --------------------------------------------
# Remove compositions where any element falls below a minimum fraction.
# Useful for avoiding compositions that are experimentally inaccessible.

pf_min = Phase_Field()
pf_min.setup_uncharged(["Fe", "Mn", "Ti"])
n_before = pf_min.omega.shape[0]
pf_min.constrain_minimum_amount(0.05)   # every element must be ≥ 5 %
print(f"5a min-amount 5%: {n_before} → {pf_min.omega.shape[0]} points")

pl = make_plotter(pf_min)
pl.plot_points(pf_min.omega, color='#888888', name='grid', s=3)
pl.show(title="5a. constrain_minimum_amount(0.05) — corners removed")
# ↑ The corners and edges of the triangle are removed because they have
#   one or two elements at zero.

# --- 5b. Precursors: cutting the field ---------------------------------------
# Pass precursors= to restrict the grid to the convex hull of those phases.
# This is useful when your synthesis is limited to specific starting materials.
# The corners and edges shown are those of the precursor hull, not the full field.

pf_cut = Phase_Field()
pf_cut.setup_uncharged(
    ["Fe", "Mn", "Ti", "Cu"],
    precursors=["Fe3Mn", "FeTi2", "MnCu", "FeCu3"]   # 4 phases spanning 3-D but sub-region
)
print(f"5b precursors (cut):       {pf_cut.omega.shape[0]} points inside precursor hull")

pl = make_plotter(pf_cut)
pl.plot_points(pf_cut.omega, color='#888888', name='grid', s=2)
pl.show(title="5b. Precursors cut the 3-D field to a sub-region")
# ↑ The corners shown are the precursor compositions, not the pure elements.
#   Only compositions inside their convex hull are on the grid.

# --- 5c. Precursors: reducing the dimension ----------------------------------
# If all precursors lie on a lower-dimensional subspace, the field collapses
# automatically.  Here we use 4 elements (normally 3-D) but all precursors
# have Cu=0, so they only span the Fe-Mn-Ti face — a 2-D plane.
# The field reduces from 3-D to 2-D and Square should be used for plotting.

pf_reduced = Phase_Field()
pf_reduced.setup_uncharged(
    ["Fe", "Mn", "Ti", "Cu"],
    precursors=["FeMn", "FeTi", "MnTi"]   # all Cu=0 → lie on the Fe-Mn-Ti face
)
print(f"5c precursors (dim reduce): constrained_dim={pf_reduced.constrained_dim}  "
      f"({pf_reduced.omega.shape[0]} points)")
# ↑ constrained_dim=2 despite four elements — Cu is locked to zero.

pl = make_plotter(pf_reduced)
pl.plot_points(pf_reduced.omega, color='#888888', name='grid', s=3)
pl.show(title="5c. Precursors collapse 4-element field from 3-D to 2-D")
# ↑ A triangle even though we specified four elements.

