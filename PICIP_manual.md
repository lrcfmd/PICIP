# PICIP User Manual

**Probabilistic Isolation of Inorganic Crystalline Phases**

PICIP uses Bayesian inference to predict the composition of an *unknown* phase present in a multi-phase XRD sample. Given the measured sample composition and the identities of the *known* co-existing phases (from Rietveld refinement), PICIP places a probability density over the entire composition space and returns the most likely compositions for the unknown phase, together with suggested next experiments.

---

> **New here? Start with the tutorials** (clone the repo to get these):
> - `tutorials/tutorial_setup.py` — phase field setup (all options)
> - `tutorials/tutorial_2d.py` — 2-D phase fields end to end
> - `tutorials/tutorial_3d.py` — 3-D phase fields end to end
>
> Run any tutorial top to bottom; each section opens an interactive figure in your browser.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Step 1 — Set up a phase field](#2-step-1--set-up-a-phase-field)
3. [Step 2 — Add samples](#3-step-2--add-samples)
4. [Step 3 — Run inference](#4-step-3--run-inference)
5. [Step 4 — Plot results](#5-step-4--plot-results)
6. [Suggestions — what to synthesise next](#6-suggestions--what-to-synthesise-next)
7. [Multiple samples](#7-multiple-samples)
8. [Single-known (original PICIP) inference](#8-single-known-original-picip-inference)
9. [Composition spreading](#9-composition-spreading)
10. [Parameter reference](#10-parameter-reference)
11. [Reference](#11-reference)

---

## 1. How it works

### The problem

After a Rietveld refinement you know:

- The **overall sample composition**
- The presence of an *unknown* phase
- The **identities** of the co-existing *known* phases
- The **estimated mass fractions** of each *known* phase

What you do not know is the composition or mass fraction of any *unknown* phase.

### Geometry

PICIP works in **constrained composition space** — the simplex of all charge-neutral (or stoichiometry-neutral) compositions in your element system. For a ternary uncharged system this is a 2-D triangle; for a quaternary system it is a 3-D tetrahedron.

All known phases occupy fixed points in this space. Their mass fractions define a probability distribution over the *known simplex* (the line between two knowns, or the triangle between three).

### Inference

For each point on the known simplex, PICIP casts a ray from that point through the measured sample composition toward the opposite boundary. Any composition on that ray is consistent with the sample being a mixture of the known phases at that support point *plus* the unknown at some composition along the ray.

The probability density on each ray is proportional to the probability of the support point. Stacking all rays and interpolating onto the composition grid gives the full probability density over the phase field — the set of unknown compositions consistent with the measurement.

When **multiple samples** share the same unknown phase, their individual densities are multiplied together. The combined density is far narrower than any individual prediction because only compositions consistent with *all* samples simultaneously survive.

---

## 2. Step 1 — Set up a phase field

A `Phase_Field` defines the grid of compositions that PICIP reasons over. Choose the setup method that matches your system.

```python
from picip import Phase_Field
pf = Phase_Field()
```

### Uncharged systems

Use when there is no charge constraint (e.g. alloys, or systems where you want to span all compositions without a neutrality requirement). The constrained dimension is `N − 1`.

```python
pf.setup_uncharged(["Fe", "Mn", "Ti"])           # 3 elements → 2-D triangle
pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])      # 4 elements → 3-D tetrahedron
```

> See `tutorial_2d.py` (uncharged ternary) and `tutorial_3d.py` (uncharged quaternary).

### Charged systems — fixed oxidation states

Use when every element has a single, well-defined oxidation state. The charge-neutrality constraint reduces the constrained dimension to `N − 2`.

```python
pf.setup_charged({"Fe": 3, "Mn": 2, "Ti": 4, "O": -2})            # 4 elements → 2-D
pf.setup_charged({"Fe": 3, "Mn": 2, "Ti": 4, "Cu": 2, "O": -2})   # 5 elements → 3-D
```

### Charged systems — mixed oxidation states

Use when an element can take a range of oxidation states (e.g. Fe²⁺/Fe³⁺). Supply a `[min, max]` list for mixed-valence elements and a single value for fixed. The constrained dimension is `N − 2`.

```python
pf.setup_charge_ranges({"Fe": [2, 3], "Mn": [2, 3], "Ti": [3, 4], "O": -2})
```

### Custom stoichiometry constraints

Use when your system has a non-charge constraint, such as a fixed cation:anion ratio in a spinel or perovskite. `custom_con` rows are normal vectors of equality constraints of the form `custom_con @ x = 0` (in addition to sum-to-one).

```python
# 4(Fe + Mn + Ti + Cu) = 3O  →  spinel M₃O₄ stoichiometry
import numpy as np
pf.setup_custom(
    ["Fe", "Mn", "Ti", "Cu", "O"],
    custom_con=np.array([[4, 4, 4, 4, -3]])
)
```

### Grid resolution

```python
pf.setup_uncharged(["Fe", "Mn", "Ti"], n_points=20000)   # exact target count
pf.setup_uncharged(["Fe", "Mn", "Ti"], resolution=100)   # spacing = 1/resolution
```

> [!NOTE]
> If neither argument is given, PICIP defaults to ~10 000 points for 2-D fields and ~6 000 for 3-D. `n_points` takes precedence over `resolution` if both are supplied.

### Precursors — restricting the phase field

Precursors are optional compositions that constrain which part of the phase field PICIP considers. Pass them to any `setup_*` method.

**Region restriction** — if the precursors span the full constrained dimensionality, `omega` is trimmed to the convex hull of the precursor compositions:

```python
pf.setup_uncharged(
    ["Fe", "Mn", "Ti", "Cu"],
    precursors=["Fe3Mn", "FeTi3", "MnCu", "FeMnTiCu"]   # restrict to sub-tetrahedron
)
```

**Dimension reduction** — if all precursors lie in a lower-dimensional sub-space (e.g. all have a particular element fixed to zero), PICIP detects this and reduces `constrained_dim` accordingly:

```python
# 5-element system — Zn=0 throughout → 3-D effective space
pf.setup_uncharged(
    ["Fe", "Mn", "Ti", "Cu", "Zn"],
    precursors=["Fe2Mn", "FeMnTi", "FeTiCu", "MnTiCu"]
)
```

> [!TIP]
> After dimension reduction, use `make_plotter(pf)` (step 4) — it automatically picks the correct plotter for the reduced space without you needing to track `constrained_dim`.

### Reproducible grids

The constrained basis is random by default. To fix it across runs:

```python
pf.setup_uncharged(["Fe", "Mn", "Ti"], print_basis=True)        # prints basis to stdout
pf.setup_uncharged(["Fe", "Mn", "Ti"], precalculated_basis=<array>)  # reuse it
```

---

## 3. Step 2 — Add samples

A `Sample` holds one measured XRD result: the overall composition, the identified co-existing phases, and their Rietveld mass fractions.

```python
from picip import PICIP, Sample

sample = Sample(
    "s1",           # name — used in plot titles and legends
    "Fe2Mn4Ti4",    # measured composition (pymatgen formula string or Composition)
)
sample.add_knowns(["FeMn", "FeTi"])        # phases identified by XRD
sample.add_mass_weights([0.3, 0.7])        # Rietveld mass fractions (must sum to 1)
sample.set_predicted_error(0.3)            # uncertainty: 0 = trust exactly, 1 = very rough
```

If you have molar fractions instead:

```python
sample.add_molar_weights([0.3, 0.7])       # converted internally
```

> [!TIP]
> `predicted_error` is the main tuning parameter — typical starting value is **0.2–0.3**. Lower values produce a tight, peaked probability cloud; higher values spread it out. The effect is demonstrated in section 3 of `tutorial_2d.py` and section 4 of `tutorial_3d.py`.

Register the sample with PICIP:

```python
picip = PICIP(pf)
picip.add_sample(sample)
```

`add_sample` validates the sample against the phase field, converts compositions into the constrained basis, and computes the weight-averaged known position. Multiple samples can be added for combined inference — see [Multiple samples](#7-multiple-samples).

---

## 4. Step 3 — Run inference

```python
pred = picip.run(n_l=50, n_p=50)
```

Returns a `Prediction` object containing the normalised probability density over the phase field grid.

### Resolution parameters

| Parameter | Controls | Recommended |
|-----------|----------|-------------|
| `n_l` | Points along the known simplex — more gives a smoother result | 50 for 2-D, 30–50 for 3-D |
| `n_p` | Interpolation points per projected ray | 50 for 2-D, 10–20 for 3-D |

> [!NOTE]
> Diminishing returns above `n_l = 100`. The effect is shown in section 5 of `tutorial_2d.py`.

### Running a subset of samples

```python
pred_a    = picip.run(0)      # sample at index 0 only
pred_b    = picip.run(1)      # sample at index 1 only
pred_both = picip.run()       # all samples — densities multiplied together
```

---

## 5. Step 4 — Plot results

```python
from picip import make_plotter

pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show()
```

`make_plotter` automatically returns the correct plotter for the phase field — 2-D or 3-D, and after any precursor-driven dimension reduction.

### Display modes (3-D)

Three ways to render the 3-D probability cloud — toggle between them in the Plotly legend. Demonstrated in section 2 of `tutorial_3d.py`.

```python
pl = make_plotter(pf)
pred.name = 'nonzero'
pl.plot_prediction_results(pred, plot_average_known=True)

pred.name = 'top 50%'
pl.plot_prediction_results(pred, p_mode='top', top=0.5,
                            plot_samples=False, visible='legendonly')

pred.name = 'surface 50%'
pl.plot_prediction_results(pred, p_mode='surface', top=0.5,
                            plot_samples=False, visible='legendonly')
pred.name = 's1'    # restore
pl.show(title="Display modes")
```

| Mode | Description |
|------|-------------|
| `'nonzero'` (default) | Every grid point with p > 0 |
| `'top'` | Scatter of the top X% of probability mass |
| `'surface'` | Convex-hull isosurface of the top X% region — cleanest for presentations |

### Detail view — algorithm layers

Shows the internal steps: support distribution on the known simplex, projected rays, and final interpolated density. Demonstrated in section 2 of `tutorial_2d.py` and section 3 of `tutorial_3d.py`.

```python
pl = make_plotter(pf)
pl.plot_detail_view(pred)
pl.show()

# 3-D with surface overlay:
pl = make_plotter(pf)
pl.plot_detail_view(pred, top_mode='surface', top=0.5)
pl.show()
```

### Overlaying multiple predictions

Pass `visible='legendonly'` for traces hidden on load but toggleable in the legend:

```python
pl = make_plotter(pf)
pl.plot_prediction_results(preds[0], plot_average_known=True)
pl.plot_prediction_results(preds[1], plot_samples=False, visible='legendonly')
pl.plot_prediction_results(preds[2], plot_samples=False, visible='legendonly')
pl.show(title="Comparison")
```

### Saving figures

```python
pl.show(save="my_figure")    # writes my_figure.html (interactive Plotly)
```

---

## 6. Suggestions — what to synthesise next

```python
suggestions = picip.suggest(
    pred,
    n=5,           # number of suggestions
    min_dist=0.05  # minimum spacing between suggestions in constrained-basis
                   # coordinates, where 1.0 spans the full phase field
)
```

Selection works in two steps:

1. **Mean** — the probability-weighted centroid of the density, snapped to the nearest grid point. This is the single most likely unknown composition and is always returned first (`label = 'mean'`).
2. **Sampled** — the remaining `n − 1` points are drawn sequentially by probability-weighted random sampling (`label = 'sampled'`). Before each draw, all grid points within `min_dist` of any already-chosen point (and of the measured sample compositions) are zeroed out, so each new point must lie in a fresh region of the density. This prevents clustering while still favouring high-probability areas.

If `min_dist` is large enough that no valid candidates remain, fewer than `n` points are returned with a warning.

```python
# Print
for label, row in zip(suggestions.labels, suggestions.standard):
    print(f"  {label:8s}  Fe={row[0]:.3f}  Mn={row[1]:.3f}  Ti={row[2]:.3f}")

# Save to CSV
suggestions.save("suggestions.csv")
```

> Suggestions are automatically overlaid on the next `plot_prediction_results` call. Demonstrated in section 6 of both tutorial files.

---

## 7. Multiple samples

Adding several samples that share the same unknown phase multiplies their probability densities together. The combined result is far narrower than any single prediction — only compositions consistent with *all* measurements simultaneously survive.

```python
picip = PICIP(pf)
picip.add_sample(s_a)
picip.add_sample(s_b)

pred_a    = picip.run(0)     # sample a only
pred_b    = picip.run(1)     # sample b only
pred_both = picip.run()      # combined

pl = make_plotter(pf)
pl.plot_prediction_results(pred_a, plot_average_known=True)
pl.plot_prediction_results(pred_b, plot_samples=False, visible='legendonly')
pl.plot_prediction_results(pred_both, plot_samples=False, visible='legendonly')
pl.show(title="Multiple samples")
```

> [!TIP]
> Samples with different known mass-fraction splits produce rays that cross at different angles, which helps narrow the combined prediction. This is not required — any samples consistent with the same unknown phase can be combined — but the narrowing effect is greatest when the splits differ.

Demonstrated in section 4 of `tutorial_2d.py` and section 5 of `tutorial_3d.py`.

---

## 8. Single-known (original PICIP) inference

The original PICIP algorithm described in the paper uses a **Gaussian method** rather than the Dirichlet simplex approach. It applies when you have identified only one co-existing known phase, so there is no simplex to discretise — just a single point in composition space.

It is triggered automatically when you supply one known phase, or explicitly via `version='gaussian'` on any run:

```python
# Automatic — one known
sample = Sample("s1", "Fe2Mn4Ti4")
sample.add_knowns(["FeMn"])
sample.add_mass_weights([1.0])
sample.set_predicted_error(0.3)
picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run()                          # Gaussian used automatically

# Explicit — force Gaussian even with 2+ knowns
pred_gaussian = picip.run(version='gaussian')   # original PICIP algorithm
pred_dirichlet = picip.run(version='b')         # Dirichlet (default)
```

### How it differs from the Dirichlet method

With 2+ knowns, the mass fractions define a known simplex (a line for two knowns, a triangle for three). The Dirichlet method places support across this simplex and casts rays from each support point through the sample.

With 1 known — or when `version='gaussian'` is used regardless of how many knowns there are — there is no simplex. Instead, support is a **Gaussian centred on the average known** (the mass-fraction-weighted centroid of all knowns, which for a single known is just that known composition itself). The algorithm then:

1. Fires `n_l` rays from the **sample point** outward on a hemisphere pointing away from the average known.
2. Weights each ray by a **Gaussian** in the perpendicular distance from the average known to that ray — rays that nearly pass through it get high weight; rays that miss it widely get low weight.
3. Interpolates the weighted ray values onto the omega grid as usual.

The result is a cone centred opposite the known from the sample. The cone narrows with lower `predicted_error`.

> [!TIP]
> If you have mass fractions available, always prefer the Dirichlet method (`version='b'`, 2+ knowns) — it gives a much tighter prediction.

---

## 9. Composition spreading

`Spread` finds a set of compositions that are as evenly distributed as possible across a phase field — useful for planning an exploratory synthesis campaign where you want maximum coverage of the composition space. It uses Lloyd's algorithm (Voronoi relaxation): each point is iteratively moved to the centroid of its Voronoi cell until the arrangement is as uniform as possible.

### Quick start

```python
from picip import Phase_Field
from picip import Spread

pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

spread = Spread(pf)
result = spread.run(
    n=10,            # number of compositions to suggest
    num_repeats=50,  # independent restarts — higher gives a better solution
)
```

`run` returns a `SpreadResult` with the suggested compositions. Plot with the usual plotter:

```python
from picip import make_plotter

pl = make_plotter(pf)
pl.plot_spread_result(result)
pl.show(title="Spread compositions")
```

`make_plotter` returns the correct plotter automatically — Square for 2-D fields, Cube for 3-D.

### Known phases

Register compositions that are already known so the algorithm avoids placing spread points near them:

```python
spread.add_known_phases(["FeMn", "FeTi", "MnTi"])
result = spread.run(n=10, num_repeats=50)
```

Suggestions cluster in the unexplored region of the phase field, away from the known-phase markers.

### 3-D phase fields

Works identically with four elements — `make_plotter` picks the Cube automatically:

```python
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])

spread = Spread(pf)
result = spread.run(n=10, num_repeats=30)

pl = make_plotter(pf)
pl.plot_spread_result(result)
pl.show()
```

### Spread quality — histogram diagnostic

`evaluate_spread` opens a matplotlib figure showing two distributions:

- **Worst coverage distance (dwcd):** how far the most isolated grid point is from its nearest suggestion. Smaller = better coverage.
- **Minimum pairwise distance (dmpd):** the closest pair of suggestions. Larger = better separation.

A good solution has similar dwcd and dmpd — the suggestions are well separated *and* they cover the field without leaving large gaps.

```python
spread.evaluate_spread(result_few)    # coarse coverage
spread.evaluate_spread(result_many)   # fine coverage
```

### Precursor rounding

When the phase field was set up with precursors, `simplify_to_precursors` snaps each spread point to the nearest composition achievable from rounded amounts of precursors:

```python
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])

spread = Spread(pf)
result = spread.run(n=8, num_repeats=30)
result_rounded = spread.simplify_to_precursors(result, accuracy=2)
```

`accuracy` sets the number of decimal places used when rounding the precursor amounts. This ensures that compositions satisfying the imposed constraints (such as charge neutrality) can still be generated despite rounding errors.

Inspect the rounded compositions and precursor amounts:

```python
for row, amounts in zip(result_rounded.points_standard, result_rounded.precursor_amounts):
    label = "  ".join(f"{l}={a:.2f}" for l, a in zip(result_rounded.precursor_labels, amounts))
    print(f"  Fe={row[0]:.3f}  Mn={row[1]:.3f}  Ti={row[2]:.3f}    ({label})")
```

### Saving results

```python
result.save("../output/spread_suggestions")    # writes a CSV
```

The CSV contains element mole fractions for each suggested composition. If `simplify_to_precursors` has been used, precursor formula-unit amounts are included as additional columns.

---

## 10. Parameter reference

### `Spread.run`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | — | Number of compositions to suggest |
| `num_repeats` | — | Independent restarts of Lloyd's algorithm; higher gives a better solution, slower |

### `Spread.simplify_to_precursors`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `accuracy` | 2 | Decimal places for precursor formula-unit amounts (normalized to sum to LCM of precursor formula sizes) |

### `Phase_Field.setup_*`

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_points` | int | Target grid size. Defaults: 10 000 (2-D), 6 000 (3-D). Overrides `resolution`. |
| `resolution` | int | Grid spacing = 1/resolution. Ignored if `n_points` is set. |
| `precalculated_basis` | ndarray | Fixed basis for reproducible grid layouts. |
| `print_basis` | bool | Print the basis to stdout so it can be reused. |
| `precursors` | list of str | Restrict (cut) or reduce the dimensionality of the phase field. |

### `Sample`

| Method | Description |
|--------|-------------|
| `add_knowns(list)` | Phase composition strings (pymatgen format) |
| `add_mass_weights(list)` | Rietveld mass fractions — primary input |
| `add_molar_weights(list)` | Molar fractions (converted internally) |
| `set_predicted_error(σ)` | Uncertainty in weights: 0 = precise, 1 = rough |

### `PICIP.run`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_indexes` | None (all) | int or slice to select a subset of samples |
| `version` | `'b'` | `'b'` = Dirichlet (default); `'gaussian'` = original PICIP algorithm |
| `n_l` | 50 | Points along the known simplex |
| `n_p` | 50 | Interpolation points per projected ray |
| `name` | auto | Label for the returned Prediction |

### `PICIP.suggest`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | — | Number of suggestion points |
| `min_dist` | None | Minimum spacing in constrained-basis units (1.0 = full field width) |
| `past_samples` | None | Extra exclusion points (default: the prediction's sample compositions) |

### `plot_prediction_results`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_mode` | `'nonzero'` | `'nonzero'` / `'top'` / `'surface'` (3-D only) |
| `top` | 0.5 | Fraction of probability mass shown for `'top'` and `'surface'` modes |
| `plot_samples` | True | Show measured sample composition markers |
| `plot_knowns` | True | Show known phase markers |
| `plot_average_known` | False | Show mass-fraction-weighted centroid of knowns |
| `visible` | True | `True` / `'legendonly'` — for overlaying multiple predictions |

### `show`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `title` | auto | Figure title |
| `save` | None | Filename (no extension) — exports an interactive HTML file |
| `show` | True | Display in browser; set False for export-only workflows |
| `return_fig` | False | Return the Plotly figure object instead of displaying |

---

## 11. Reference

If you use PICIP in your work, please cite:

> Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
