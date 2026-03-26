"""
Composition Spreading Tutorial
================================
Sections:  1. Quick start  2. Known phases  3. 3-D phase field
           4. Spread quality  5. Save output

Uses Lloyd's algorithm (Voronoi relaxation) to find maximally spread-out
compositions within a phase field — useful for planning exploratory synthesis
where you want to cover the composition space as uniformly as possible.

Run top to bottom — each section opens a browser figure.

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md
  Setup tutorial: tutorial_setup.py
  2-D PICIP tutorial: tutorial_2d.py
"""

from picip import Phase_Field, Spread, make_plotter


# ══════════════════════════════════════════════════════════════════════════════
# 1.  QUICK START — uniform coverage of a ternary phase field
# ══════════════════════════════════════════════════════════════════════════════

# 3 elements → 2-D triangle.  Spread finds n points as evenly spaced as possible
# across the interior, with corners fixed as boundaries.
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

spread = Spread(pf)
result = spread.run(
    n=10,            # number of compositions to suggest
    num_repeats=50,  # independent restarts (higher = better solution, slower)
)

pl = make_plotter(pf)
pl.plot_spread_result(result)
pl.show(title="1. Quick start — 10 spread compositions, no known phases")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  KNOWN PHASES — treat existing compositions as fixed points
#
#     add_known_phases() registers compositions that are already known.
#     The algorithm avoids placing spread points near these, so suggestions
#     cover the unexplored region of the phase field.
# ══════════════════════════════════════════════════════════════════════════════

pf2 = Phase_Field()
pf2.setup_uncharged(["Fe", "Mn", "Ti"])

spread2 = Spread(pf2)
spread2.add_known_phases(["FeMn", "FeTi", "MnTi"])   # three known phases
result2 = spread2.run(n=10, num_repeats=50)

pl2 = make_plotter(pf2)
pl2.plot_spread_result(result2)
pl2.show(title="2. With known phases — suggestions avoid FeMn, FeTi, MnTi")
# ↑ Green points cluster away from the amber known-phase diamonds.


# ══════════════════════════════════════════════════════════════════════════════
# 3.  3-D PHASE FIELD — quaternary system
#
#     4 elements → 3-D tetrahedron.  make_plotter handles this automatically.
# ══════════════════════════════════════════════════════════════════════════════

pf3 = Phase_Field()
pf3.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])

spread3 = Spread(pf3)
spread3.add_known_phases(["FeMn", "FeTi"])
result3 = spread3.run(n=10, num_repeats=30)

pl3 = make_plotter(pf3)
pl3.plot_spread_result(result3)
pl3.show(title="3. 3-D phase field — Fe-Mn-Ti-Cu with known phases")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SPREAD QUALITY — histogram diagnostic
#
#     evaluate_spread() plots two distributions:
#       - Worst coverage distance (dwcd): how far the most isolated omega point
#         is from its nearest suggested composition.  Smaller = better coverage.
#       - Minimum pairwise distance (dmpd): how close the nearest pair of
#         suggested compositions is.  Larger = better separation.
#     A good solution has small dwcd (every point well covered) and
#     large dmpd (sites well separated).
# ══════════════════════════════════════════════════════════════════════════════

pf4 = Phase_Field()
pf4.setup_uncharged(["Fe", "Mn", "Ti"])

spread4 = Spread(pf4)
result_few  = spread4.run(n=5,  num_repeats=20)   # coarse coverage
result_many = spread4.run(n=20, num_repeats=20)   # fine coverage

print("\n4. Spread quality for n=5:")
spread4.evaluate_spread(result_few)    # opens matplotlib figure

print("4. Spread quality for n=20:")
spread4.evaluate_spread(result_many)   # opens matplotlib figure


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

pf5 = Phase_Field()
pf5.setup_uncharged(["Fe", "Mn", "Ti"])

spread5 = Spread(pf5)
spread5.add_known_phases(["FeMn", "FeTi"])
result5 = spread5.run(n=10, num_repeats=50)

path = result5.save("../output/spread_suggestions")
print(f"\n5. Saved to: {path}")

pl5 = make_plotter(pf5)
pl5.plot_spread_result(result5)
pl5.show(
    title="5. Spread suggestions",
    save="../output/spread_suggestions",
)
# ↑ Writes spread_suggestions.html — open in any browser.
