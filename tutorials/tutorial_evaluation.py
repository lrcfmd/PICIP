"""
PICIP Tutorial — Evaluating PICIP on synthetic phase fields
============================================================
Sections:  1. A single 2-D one-shot trial   2. A single 3-D one-shot trial
           3. Dirichlet vs Gaussian over 50 one-shot 3-D trials
           4. Dirichlet vs Gaussian over many varied 2-D phase fields
Run top to bottom.  Figures are written to ../output/ as self-contained HTML.

Why this exists
---------------
Every other tutorial runs PICIP on a real problem, where nobody knows the right
answer.  `picip.evaluation` builds phase fields whose answer is known by
construction, hides one phase, and measures how close PICIP gets to it.  That
lets you check a change, compare two inference methods, or estimate out how many
experiments are required.

What you need
-------------
Nothing.  Unlike the other tutorials there is no measured sample to supply —
this one generates the phase fields it tests against, so it runs anywhere.

What it produces
----------------
  output/tutorial_eval_diagnostic_2d.html  §1  one 2-D trial, stage by stage
  output/tutorial_eval_diagnostic_3d.html  §2  the same in 3-D
  output/tutorial_eval_oneshot_3d.html     §3  Dirichlet vs Gaussian box plots
  output/tutorial_eval_many_fields_2d.html §4  the same, pooled over 12 2-D fields

Each is a self-contained HTML file — open it in any browser, offline, with no
plotly.min.js sidecar to keep alongside it.  Nothing opens automatically.

Runtime
-------
About 30 seconds in total — §3 runs 100 trials and §4 another 144, both using
every CPU core (`n_jobs=-1`).  Results are identical to a serial run, so drop to
`n_jobs=1` if you want to step through it in a debugger.

Key terms
---------
  unknown phase   the phase hidden from PICIP — the answer being recovered
  known phases    the co-existing phases PICIP is told about
  one shot        PICIP sees one composition and proposes once (`n_batches=1`)
  iterative       each batch is observed and fed back (`n_batches > 1`)
  closest hit     distance from the unknown phase to the nearest suggestion,
                  in constrained-basis units where the field spans about 1.0

Documentation
-------------
  Installation  : ../PICIP_installation.md
  User manual   : ../PICIP_manual.md  (§12 — full API reference)
  Setup tutorial: tutorial_setup.py
  2-D tutorial  : tutorial_2d.py
  3-D tutorial  : tutorial_3d.py
  Parameter sweeps: ../src/parameter_optimisation.py
"""

import os

# Import picip.evaluation explicitly.  It is deliberately NOT pulled in by
# `import picip`, because it caps BLAS/OpenMP threads at import time for its
# parallel trial running — that must stay something you opt into.
import numpy as np

from picip.evaluation import (
    generate_test_phase_field,   # one synthetic field
    generate_phase_field_set,    # a list of them
    evaluate,                    # run PICIP against a field, hiding one phase
    diagnostic_figure,           # one trial, stage by stage
    distance_vs_batch_figure,    # closest hit vs batch number
    save_figure,                 # write one self-contained HTML file
    summarise,                   # short text report of a run
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')


# ══════════════════════════════════════════════════════════════════════════════
# 1.  A SINGLE 2-D ONE-SHOT TRIAL, WITH THE DIAGNOSTIC FIGURE
#
#     "One shot" = PICIP sees one measured composition and proposes once.
#     That is `n_batches=1`, the default.
# ══════════════════════════════════════════════════════════════════════════════

# Dimensionality follows from charge neutrality: n elements with the sum-to-one
# and charge-balance constraints give constrained_dim = n - 2.  So a 4-element
# system is 2-D.  '3c1a' = 3 cations + 1 anion → a triangular polytope.
#
# n_internal=1 puts exactly one phase strictly inside the polytope.  Since
# `evaluate` hides every internal phase in turn, one internal phase plus
# num_repeats=1 gives exactly one trial — which is what we want to inspect.
field_2d = generate_test_phase_field(
    dim=2,              # constrained dimensionality
    structure='3c1a',   # 3 cations + 1 anion  (also valid in 2-D: '2c2a')
    n_internal=1,       # phases strictly inside the polytope — the hideable ones
    n_edge=2,           # phases sitting on polytope edges
    resolution=40,      # omega grid density (spacing = 1/resolution)
    seed=0,             # reproducible ion choice and phase placement
)
print("1. 2-D field")
print(field_2d.summary())

# Run PICIP against it.  Under the hood each trial:
#   - draws a random composition guaranteed to produce the unknown phase
#   - reads off which phases form there and in what molar ratio
#   - converts those to mass fractions and corrupts them with Dirichlet noise
#     of size `experimental_error` — i.e. simulates the Rietveld measurement
#   - hands the errored phase_fractions to PICIP and asks for `batch_size` suggestions
#
# version='dirichlet' is the Dirichlet method; 'gaussian' is the legacy one (§3 below).
# diagnostic=True keeps the per-sample stage-by-stage arrays that
# diagnostic_figure() needs — off by default because it retains every sample's
# projected ray points.
trial_2d = evaluate(
    field_2d,
    version='dirichlet',              # Dirichlet
    batch_size=5,             # suggestions per batch
    n_batches=1,              # ONE SHOT.  >1 makes it iterative — see §12 of the manual
    experimental_error=0.10,  # measurement noise on the observed phase_fractions
    num_repeats=1,            # trials per unknown phase
    diagnostic=True,          # retain what diagnostic_figure() needs
    seed=0,
)[0]                          # exactly one trial, so take it

print(f"\n   unknown phase : {trial_2d.unknown_formula}")
print(f"   knowns given : {', '.join(trial_2d.known_formulas)}")
print(f"   closest hit  : {trial_2d.min_distance:.4f}")
#   ^ distance from the unknown phase to the nearest of the 5 suggestions,
#     in constrained-basis units where the field spans roughly 1.0.

# The diagnostic figure puts every stage of the algorithm on one set of axes as
# separately toggleable layers.  Switching them on one at a time walks through
# what PICIP actually did:
#
#   Dirichlet support on known simplex — the Dirichlet likelihood over the
#       known simplex.  In 2-D there are 2 knowns, so this is the k1–k2 LINE.
#   projected p_density   — the raw ray points carrying that support outward
#                           through the sample, before interpolation.
#   interpolated p_density— that density on the omega grid.  Hidden on load.
#   true / estimated direction — lines from the average known through the
#       sample.  The TRUE line passes exactly through the unknown phase, so the
#       gap between the two lines is the measurement error made visible.
#   batch / unknown (truth) — what PICIP proposed, and the right answer.
save_figure(diagnostic_figure(trial_2d),
            os.path.join(OUT, 'tutorial_eval_diagnostic_2d'))
print("   figure       : ../output/tutorial_eval_diagnostic_2d.html")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  THE SAME THING IN 3-D
#
#     Identical call, identical figure layers.  The only difference is inherent
#     to the geometry: a 3-D sample decomposes into 4 phases, so hiding one
#     leaves 3 knowns and the known simplex is the k1–k2–k3 TRIANGLE rather
#     than a line.  That is the m == dim == 3 Dirichlet branch.
# ══════════════════════════════════════════════════════════════════════════════

field_3d = generate_test_phase_field(
    dim=3,
    structure='4c1a',   # 4 cations + 1 anion → tetrahedron  (also 3-D: '3c2a')
    n_internal=1,
    n_edge=2,
    n_face=1,           # 3-D only: phases sitting on polytope faces
    resolution=25,
    seed=0,
)
print("\n2. 3-D field")
print(field_3d.summary())

trial_3d = evaluate(field_3d, version='dirichlet', batch_size=5, n_batches=1,
                    experimental_error=0.10, num_repeats=1,
                    diagnostic=True, seed=0)[0]

print(f"\n   unknown phase : {trial_3d.unknown_formula}")
print(f"   knowns given : {len(trial_3d.known_formulas)} phases")
print(f"   closest hit  : {trial_3d.min_distance:.4f}")

save_figure(diagnostic_figure(trial_3d),
            os.path.join(OUT, 'tutorial_eval_diagnostic_3d'))
print("   figure       : ../output/tutorial_eval_diagnostic_3d.html")
# ↑ Open both figures side by side.  The layer names are identical, so you can
#   read the 2-D line case and the 3-D triangle case against each other.


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DIRICHLET vs GAUSSIAN — 50 ONE-SHOT 3-D TRIALS EACH
#
#     One trial tells you very little: the measurement noise is a random draw,
#     and a lucky or unlucky draw swamps any real difference between methods.
#     Repeat it 50 times per method and the comparison becomes readable.
# ══════════════════════════════════════════════════════════════════════════════

# 5 fields x 5 internal phases x 2 repeats = 50 trials.
fields_3d = generate_phase_field_set(
    dim=3, structure='4c1a', n_internal=5, n_edge=2, n_face=1,
    repeats=5,          # 5 independent fields
    resolution=25,
    seed=1,
)
print(f"\n3. Dirichlet vs Gaussian — {len(fields_3d)} fields x 5 unknown phases "
      f"x 2 repeats = 50 trials per method")

# Run both methods over the SAME fields with the SAME seed.  Each trial seeds
# its RNG from its own identity (field, unknown phase, repeat) rather than from a
# shared stream, so trial i is the *same physical experiment* under both
# methods — same composition, same measurement noise.  That makes this a paired
# comparison, which is far more sensitive than comparing two independent runs.
common = dict(fields=fields_3d, batch_size=5, n_batches=1,
              experimental_error=0.10, num_repeats=2,
              n_jobs=-1,        # all CPU cores; results identical to n_jobs=1
              seed=0)
dirichlet = evaluate(version='dirichlet', **common)
gaussian = evaluate(version='gaussian', **common)

print("\n   Dirichlet:")
print('   ' + summarise(dirichlet).replace('\n', '\n   '))
print("\n   Gaussian:")
print('   ' + summarise(gaussian).replace('\n', '\n   '))

# Pair the two lists index by index and keep the trials both methods scored.
pairs = [(d.min_distance, g.min_distance)
         for d, g in zip(dirichlet, gaussian)
         if d.min_distance is not None and g.min_distance is not None]
d_arr, g_arr = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
diff = d_arr - g_arr                        # negative ⇒ Dirichlet is closer

se = diff.std(ddof=1) / np.sqrt(len(diff))
t = diff.mean() / se if se > 0 else np.nan

print(f"\n   paired over {len(pairs)} trials scored by both methods")
print(f"     Dirichlet  mean {d_arr.mean():.4f}   median {np.median(d_arr):.4f}")
print(f"     Gaussian   mean {g_arr.mean():.4f}   median {np.median(g_arr):.4f}")
print(f"     difference {diff.mean():+.4f} +/- {se:.4f}  (t={t:+.2f})")
print(f"     Dirichlet closer in {int((diff < 0).sum())}/{len(diff)} trials")
# A |t| above about 2 is the usual threshold for "not just noise".  Below that,
# 50 trials were not enough to separate the two methods — run more repeats
# rather than reading anything into the sign of the difference.

# For a one-shot run there is only one batch, so there is nothing to plot
# against — distance_vs_batch_figure() draws a BOX PLOT of the per-trial
# distances instead (median, quartiles, whiskers, every trial as a point).
# Passing a dict puts both methods on the same axes, side by side, which is the
# picture that actually settles the comparison.
save_figure(
    distance_vs_batch_figure(
        {'dirichlet': dirichlet, 'gaussian': gaussian},
        title='One-shot 3-D — closest hit, Dirichlet vs Gaussian'),
    os.path.join(OUT, 'tutorial_eval_oneshot_3d'))
print("\n   figure       : ../output/tutorial_eval_oneshot_3d.html")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  DOES THAT RESULT HOLD OVER MANY DIFFERENT PHASE FIELDS?
#
#     Section 3 ran on 5 phase fields that all shared one structure and one set
#     of ions.  A result from a handful of near-identical fields is a result
#     about those fields, not about the method.  Here the same paired comparison
#     is run over a larger, more varied population of 2-D fields — different
#     ions, and both 2-D polytope shapes ('2c2a' quadrilateral and '3c1a'
#     triangle), chosen at random per field by leaving `structure=None`.
#
#     Staying in 2-D keeps one thing fixed that would otherwise confound the
#     comparison: a 2-D sample always decomposes into 3 phases, so hiding one
#     always leaves PICIP exactly 2 knowns to work from.
# ══════════════════════════════════════════════════════════════════════════════

# 12 fields x 3 unknown phases x 2 repeats = 72 trials per method.
fields_2d = generate_phase_field_set(
    dim=2,
    structure=None,     # let each field pick its own 2-D structure at random
    n_internal=3,
    n_edge=2,
    repeats=12,         # 12 independent fields, each with its own ions
    resolution=30,
    seed=2,
)
structures_used = sorted({f.structure for f in fields_2d})
print(f"\n4. Dirichlet vs Gaussian over {len(fields_2d)} varied 2-D phase fields "
      f"({', '.join(structures_used)})")

shared = dict(fields=fields_2d, batch_size=5, n_batches=1,
              experimental_error=0.10, num_repeats=2, n_jobs=-1, seed=0)
d_many = evaluate(version='dirichlet', **shared)
g_many = evaluate(version='gaussian', **shared)

# Same pairing as §3: trial i is the same physical experiment under both
# methods — same field, same unknown phase, same composition, same measurement
# noise — so the difference of pairs is far more sensitive than comparing the
# two means independently.
pairs = [(a.min_distance, b.min_distance) for a, b in zip(d_many, g_many)
         if a.min_distance is not None and b.min_distance is not None]
da = np.array([p[0] for p in pairs])
ga = np.array([p[1] for p in pairs])
dif = da - ga                                   # negative ⇒ Dirichlet closer
sem = dif.std(ddof=1) / np.sqrt(len(dif))
tstat = dif.mean() / sem if sem > 0 else np.nan

print(f"   paired over {len(pairs)} trials scored by both methods")
print(f"     Dirichlet  mean {da.mean():.4f}   median {np.median(da):.4f}")
print(f"     Gaussian   mean {ga.mean():.4f}   median {np.median(ga):.4f}")
print(f"     difference {dif.mean():+.4f} +/- {sem:.4f}  (t={tstat:+.2f})")
print(f"     Dirichlet closer in {int((dif < 0).sum())}/{len(dif)} trials")
# Pooling across many fields is what makes this the strongest statement in the
# tutorial: the comparison no longer depends on which field you happened to
# generate.  |t| > 2 is the usual bar for "not just noise".

# One box per method, pooled over every field — the summary picture.
save_figure(
    distance_vs_batch_figure(
        {'dirichlet': d_many, 'gaussian': g_many},
        title=f'One-shot closest hit over {len(fields_2d)} varied 2-D phase '
              f'fields — Dirichlet vs Gaussian'),
    os.path.join(OUT, 'tutorial_eval_many_fields_2d'))
print("\n   figure       : ../output/tutorial_eval_many_fields_2d.html")


# ── Where to go next ──────────────────────────────────────────────────────────
#
# Iterative runs: pass n_batches > 1 and every suggestion is observed, added as
# a further sample, and the next batch drawn from the product of all densities
# so far — so PICIP chooses where to look next.  Then
# distance_vs_batch_figure() shows a real convergence curve:
#
#   iterative = evaluate(field_3d, batch_size=3, n_batches=4, diagnostic=True)
#   save_figure(distance_vs_batch_figure(iterative), '../output/convergence')
#
# Sweeps over predicted_error, resolution and method live in
# parameter_optimisation.py, outside the package.
