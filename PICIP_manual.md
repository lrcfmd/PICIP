# PICIP User Manual

**Probabilistic Isolation of Inorganic Crystalline Phases**

PICIP uses Bayesian inference to predict the composition of an *unknown* phase present in a multi-phase XRD sample. Given the measured sample composition and the identities of the *known* co-existing phases (from Rietveld refinement), PICIP places a probability density over the entire composition space and returns the most likely compositions for the unknown phase, together with suggested next experiments.

Not installed yet? See the [Installation Guide](PICIP_installation.md).

---

## Contents

1. [Quick start](#1-quick-start)
2. [How it works](#2-how-it-works)
3. [Step 1 — Set up a phase field](#3-step-1--set-up-a-phase-field)
4. [Step 2 — Add samples](#4-step-2--add-samples)
5. [Step 3 — Run inference](#5-step-3--run-inference)
6. [Step 4 — Plot results](#6-step-4--plot-results)
7. [Suggestions — what to synthesise next](#7-suggestions--what-to-synthesise-next)
8. [Multiple samples](#8-multiple-samples)
9. [The two inference methods](#9-the-two-inference-methods)
10. [Composition spreading](#10-composition-spreading)
11. [Parameter reference](#11-parameter-reference)
12. [Evaluating PICIP on synthetic phase fields](#12-evaluating-picip-on-synthetic-phase-fields)
13. [Reference](#13-reference)

---

## 1. Quick start

A complete run: define the composition space, describe one measured sample,
infer where the unknown phase sits, and get compositions to synthesise next.

```python
from picip import Phase_Field, PICIP, Sample, make_plotter

# 1. The composition space. Three elements, no charge constraint → a 2-D triangle.
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

# 2. One measured sample: what you refined, and what you identified in it.
sample = Sample("s1", "Fe2Mn4Ti4")     # overall measured composition
sample.add_knowns(["FeMn", "FeTi"])    # phases you identified
sample.add_mass_weights([0.3, 0.7])    # their Rietveld mass fractions, summing to 1
sample.set_predicted_error(0.3)        # how much you trust those fractions

# 3. Infer where the unknown phase is.
picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run()

# 4. Ask what to make next.
suggestions = picip.suggest(pred, n=5, min_dist=0.05)
suggestions.print_table()

# 5. Look at it.
pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show(save="quickstart")             # opens in your browser, writes quickstart.html
```

Printed output:

```
mean      Fe=0.099  Mn=0.562  Ti=0.339
sampled   Fe=0.053  Mn=0.633  Ti=0.314
sampled   Fe=0.000  Mn=0.740  Ti=0.260
sampled   Fe=0.161  Mn=0.447  Ti=0.392
sampled   Fe=0.217  Mn=0.782  Ti=0.001
```

`mean` is the single most likely composition for the unknown phase. The
`sampled` rows are spread across the rest of the probability density, at least
`min_dist` apart, so a batch of five experiments covers the plausible region
rather than clustering on one point.

The figure shows the probability density over the whole triangle, with the
measured sample and the two known phases marked. Hover any point for its
composition.

### What each piece is

| Object | Role |
|---|---|
| `Phase_Field` | The space of compositions PICIP reasons over, as a grid of points |
| `Sample` | One measured result: composition, identified phases, their mass fractions |
| `PICIP` | Runs the inference over one or more samples |
| `Prediction` | The answer — `pred.prob_density` is one probability per grid point |
| `Suggestions` | Compositions to synthesise next; `.print_table()` prints them, `.save("file.csv")` writes them out |

### Where to go next

- **Understand the method** → [§2 How it works](#2-how-it-works)
- **Your system isn't a simple ternary** → [§3 Set up a phase field](#3-step-1--set-up-a-phase-field)
  covers charged, mixed-valence and custom-stoichiometry systems
- **The prediction looks too tight or too diffuse** → tune `predicted_error`,
  [§4](#4-step-2--add-samples)
- **You have more than one sample of the same unknown** → [§8 Multiple samples](#8-multiple-samples)
- **Run it end to end yourself** → `tutorials/tutorial_2d.py` is this example
  expanded, with a section per parameter

> **Tutorials** (clone the repo to get these — see the Installation Guide):
> | File | Covers |
> |---|---|
> | `tutorial_setup.py` | All four phase-field setup methods |
> | `tutorial_2d.py` | 2-D fields end to end, uncharged and charged |
> | `tutorial_3d.py` | 3-D fields end to end, uncharged and charged |
> | `tutorial_spread.py` | Composition spreading |
> | `tutorial_evaluation.py` | Scoring PICIP against synthetic fields (§12) |

### Ready-made scripts

`scripts/` holds one short, self-contained script per phase-field type — the quick start above adapted to each. Copy the one closest to your system and edit the compositions.

| Script | Phase field | Knowns | Method |
|---|---|---|---|
| `quickstart_uncharged.py` | `setup_uncharged` — Fe-Mn-Ti | 2 | Dirichlet + both suggestion methods |
| `quickstart_charged.py` | `setup_charged` — Ba-Y-Si-O | 2 | Dirichlet + both suggestion methods |
| `quickstart_charge_ranges.py` | `setup_charge_ranges` — Fe-Mn-O, mixed valence | 2 | Dirichlet + both suggestion methods |
| `quickstart_custom_constraint.py` | `setup_custom` — spinel Li-Fe-Mn-O | 2 | Dirichlet + both suggestion methods |
| `quickstart_charged_3knowns_gaussian.py` | `setup_charged` — Ba-Y-Si-O | 3 | Gaussian (the default) |

The first four are all 2-D with two knowns, which is a supported Dirichlet configuration, and each shows `suggest()` and `suggest_kmedoids()` side by side ([section 7](#7-suggestions--what-to-synthesise-next)). The last has three knowns in a 2-D field, which Dirichlet cannot handle, so it uses the Gaussian default — see [section 9](#9-the-two-inference-methods) for why.

---

## 2. How it works

### The problem

After a Rietveld refinement you know:

- The **overall sample composition**
- The presence of an *unknown* phase
- The **identities** of the co-existing *known* phases
- The **estimated mass fractions** of each *known* phase

What you do not know is the composition or mass fraction of any *unknown* phase.

### Geometry

PICIP works in **constrained composition space** — the polytope of all compositions which satisfy the defined constraints. This is a simplex if no constraints are considered, or some slice or subregion of that simplex if constraints such as charge neutrality or restricted precursors are imposed. For a ternary uncharged system this is a 2-D triangle, for a quaternary uncharged system this is a 3-D tetrahedron. The dimension of this space is the *constrained dimension*, `d`.

All known phases occupy fixed points in this space. Their convex hull is the *known simplex*: a line for two knowns, a triangle for three, a tetrahedron for four. Their mass fractions define a point on the known simplex referred to as the average known composition. PICIP accounts for error in the calculation of these mass fractions and uses this to construct a probability distribution on the known simplex for the average known composition.

### Inference

For each composition on the known simplex, PICIP draws a line through the measured sample: any composition on the forward portion of that line is consistent with the sample being a mixture of the known phases at that composition *plus* the unknown at some composition along the line. Each line is weighted by the probability of its supporting composition under the distribution described above, and the weighted lines are interpolated onto the composition grid to give the posterior.

When the known simplex has the same dimension as the phase field — three knowns in a 2-D phase field, or four knowns in a 3-D one — many supporting compositions share the same line through the sample. This degeneracy is one reason the Gaussian method is the default; see [§9 The two inference methods](#9-the-two-inference-methods).

When **multiple samples** share the same unknown phase, their individual densities are multiplied together. The combined density is far narrower than any individual prediction because only compositions consistent with *all* samples simultaneously survive.

---

## 3. Step 1 — Set up a phase field

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

Use when an element can take a range of oxidation states (e.g. Fe²⁺/Fe³⁺). Supply a `[min, max]` list for mixed-valence elements and a single value for fixed. The constrained dimension is `N − 1`.

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
pf.setup_uncharged(["Fe", "Mn", "Ti"], precalculated_basis=saved_basis)  # reuse it
```

---

## 4. Step 2 — Add samples

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

`add_sample` validates the sample against the phase field, converts compositions into the constrained basis, and computes the weight-averaged known position. Multiple samples can be added for combined inference — see [Multiple samples](#8-multiple-samples).

---

## 5. Step 3 — Run inference

```python
pred = picip.run(n_l=50, n_p=50)
```

Returns a `Prediction` object containing the normalised probability density over the phase field grid.

### Default routing

`picip.run()` defaults to **`version='gaussian'`** — the original PICIP algorithm from the paper, the well-tested path, and valid for any number of knowns in any dimension. You do not need to set `version` for the standard workflow.

**`version='dirichlet'`** opts into the Dirichlet method, which is implemented for exactly two configurations:

| constrained dim | # knowns | `version='dirichlet'` gives |
|---|---|---|
| 2 | 2 | **Dirichlet** along the k1–k2 line segment |
| 3 | 3 | **Dirichlet** over the k1–k2–k3 triangle |
| anything else | anything else | warning, then **Gaussian** |

These are the two cases where the known simplex has one dimension fewer than the phase field, which is where the method has a clean geometric interpretation: the unknown's position along each projection line is fixed by a single mixing weight.

Requesting `version='dirichlet'` anywhere else — 1 known, 3+ knowns in 2-D, or any count other than exactly 3 in 3-D — emits a warning that the method is untested there and falls back to Gaussian, so you still get a result, just not the one you asked for:

**The warning only appears if you asked for `'dirichlet'`.** The default path never warns: Gaussian is valid for any number of knowns in any dimension, so there is nothing to fall back from. If you have not set `version`, you will not see this message whatever your dimension and known count.

```
RuntimeWarning: The Dirichlet method (version='dirichlet') is implemented only for 2-D
with 2 known phases and 3-D with 3 known phases. Sample 's1' is 3-D with 2 known
phases, which is untested — falling back to the Gaussian method. Pass
version='gaussian' to select it explicitly and silence this warning.
```

Worked examples of opting in are in the quick-start section of `tutorials/tutorial_2d.py` (2 knowns in 2-D) and `tutorials/tutorial_3d.py` (3 knowns in 3-D).

### Resolution parameters

| Parameter | Controls | Recommended |
|-----------|----------|-------------|
| `n_l` | Number of ray directions | 50 for 2-D, 30–100 for 3-D |
| `n_p` | Interpolation points per projected ray | 10–50 (effect is small) |

> [!NOTE]
> Diminishing returns above `n_l = 100`. The effect is shown in section 5 of `tutorial_2d.py`.

### Running a subset of samples

```python
pred_a    = picip.run(0)      # sample at index 0 only
pred_b    = picip.run(1)      # sample at index 1 only
pred_both = picip.run()       # all samples — densities multiplied together
```

---

## 6. Step 4 — Plot results

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

## 7. Suggestions — what to synthesise next

```python
suggestions = picip.suggest(
    pred,
    n=5,           # number of suggestions
    min_dist=0.05  # minimum spacing between suggestions in constrained-basis
                   # coordinates
)
```

Selection works in two steps:

1. **Mean** — the probability-weighted centroid of the density, snapped to the nearest grid point. This is the single most likely unknown composition and is always returned first (`label = 'mean'`).
2. **Sampled** — the remaining `n − 1` points are drawn sequentially by probability-weighted random sampling (`label = 'sampled'`). Before each draw, all grid points within `min_dist` of any already-chosen point (and of the measured sample compositions) are zeroed out, so each new point must lie in a fresh region of the density. This prevents clustering while still favouring high-probability areas.

If `min_dist` is large enough that no valid candidates remain, fewer than `n` points are returned with a warning.

```python
# Print
suggestions.print_table()

# Save to CSV
suggestions.save("suggestions.csv")
```

> Suggestions are automatically overlaid on the next `plot_prediction_results` call. Demonstrated in section 6 of both tutorial files.

### Experimental: `suggest_kmedoids`

```python
suggestions = picip.suggest_kmedoids(pred, n=5)
```

An alternative batch selection that picks points to minimise the posterior-weighted expected distance from the batch to the true unknown — i.e. it optimises *how close the best composition in the batch will be*, rather than sampling from the density. The batch is deterministic, with no random draws.

> [!WARNING]
> **Less tested than `suggest()`.** It has been benchmarked on synthetic phase fields ([section 12](#12-evaluating-picip-on-synthetic-phase-fields)) but not on real experimental workflows. Use `suggest()` routinely; reach for this when you want a deterministic batch, or are scoring batches by closest-hit distance.

---

## 8. Multiple samples

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

## 9. The two inference methods

[Section 2](#2-how-it-works) describes the method in general terms: the mass fractions fix an average known composition on the known simplex, measurement error turns that into a probability distribution, and lines drawn through the sample are weighted by it. The two methods differ in how they represent that distribution.

### The Gaussian method (the default)

The original PICIP algorithm from the paper, and what `picip.run()` uses unless you ask for otherwise. It works for any number of knowns in any dimension, so it needs no `version` argument.

For each sample, the **average known** is the mass-fraction-weighted centroid of all reported knowns (just the single known composition when only one is present). The Gaussian method then:

1. Fires `n_l` rays from the **sample point** outward on a hemisphere pointing away from the average known.
2. Weights each ray by a **Gaussian** in the perpendicular distance from the average known to that ray — rays that nearly pass through it get high weight; rays that miss it widely get low weight.
3. Interpolates the weighted ray values onto the omega grid.

The result is a cone centred opposite the known centroid as seen from the sample. The cone narrows with lower `predicted_error`.

### The Dirichlet method

Selected with `version='dirichlet'`. Rather than collapsing the knowns to one point, it keeps them distinct and works with their *mixing ratio* directly:

1. Discretises the known simplex into `n_l` support compositions — points along the k1–k2 line in 2-D, or across the k1–k2–k3 triangle in 3-D.
2. Weights each support composition by a **Dirichlet** likelihood over the mixing ratios, with concentration `weights / predicted_error`, so the measured mass fractions peak the distribution and `predicted_error` sets its width.
3. Projects each support composition through the sample along its own line, sampling `n_p` points per line, then interpolates onto the omega grid.

Because every mixing ratio is projected separately, the measured mass fractions do more than set a centroid — they shape the density along the whole known simplex.

### Choosing between them

The Dirichlet construction needs the known simplex to be one dimension below the phase field, so that each support composition gives a distinct line through the sample. That is why `version='dirichlet'` covers 2-D + 2 knowns (a k1–k2 line) and 3-D + 3 knowns (a k1–k2–k3 triangle) and nothing else. Where the known simplex has the *same* dimension as the field (3 knowns in 2-D, 4 in 3-D), many mixing ratios share one line through the sample and the construction is not well posed.

Gaussian has no such restriction and is the better-tested path, which is why it is the default. Reach for Dirichlet when you are in one of its two supported configurations and want the mass fractions to carry more weight than a centroid. See [section 5](#5-step-3--run-inference) for the dispatch rules and the fallback warning.

---

## 10. Composition spreading

`Spread` finds a set of compositions spread as evenly as possible across a phase field — for exploratory campaigns where you want coverage rather than a targeted prediction. It uses Lloyd's algorithm: each point is repeatedly moved to the centroid of its Voronoi cell until the arrangement stops changing.

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

`run` returns a `SpreadResult` with the suggested compositions. Plot it with the usual plotter:

```python
from picip import make_plotter

pl = make_plotter(pf)
pl.plot_spread_result(result)
pl.show(title="Spread compositions")
```

### Known phases

Register compositions that are already known so the algorithm avoids placing spread points near them:

```python
spread.add_known_phases(["FeMn", "FeTi", "MnTi"])
result = spread.run(n=10, num_repeats=50)
```

Suggestions cluster in the unexplored region of the phase field, away from the known-phase markers.

### 3-D phase fields

Identical code — only the element list changes, and `make_plotter` returns a Cube instead of a Square:

```python
pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])
```

### Spread quality — histogram diagnostic

`evaluate_spread` returns a matplotlib figure showing two distributions. It displays nothing by default — pass `show=True` to open it in a viewer, or `save=<path>` to write a PNG:

- **Worst coverage distance (dwcd):** how far the most isolated grid point is from its nearest suggestion. Smaller = better coverage.
- **Minimum pairwise distance (dmpd):** the closest pair of suggestions. Larger = better separation.

A good solution has similar dwcd and dmpd — the suggestions are well separated *and* they cover the field without leaving large gaps.

```python
spread.evaluate_spread(result_few,  save="coverage_few.png")    # coarse coverage
spread.evaluate_spread(result_many, save="coverage_many.png")   # fine coverage

# ...or pass show=True instead of save= to open it in your image viewer.
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

## 11. Parameter reference

### `Spread.run`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | — | Number of compositions to suggest |
| `num_repeats` | — | Independent restarts of Lloyd's algorithm; higher gives a better solution, slower |

### `Spread.simplify_to_precursors`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `accuracy` | 2 | Decimal places for precursor formula-unit amounts (normalized to sum to LCM of precursor formula sizes) |

### `Spread.evaluate_spread`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show` | False | Open the figure in the system image viewer |
| `save` | None | Path to write the figure to as a PNG |

Returns the matplotlib figure either way; displays nothing unless asked.

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
| `version` | `'gaussian'` | `'gaussian'` is the original algorithm and works for any dimension and known count. `'dirichlet'` opts into the Dirichlet method, which covers 2-D + 2 knowns and 3-D + 3 knowns; elsewhere it warns and falls back to Gaussian. See section 5. |
| `n_l` | 50 | Number of ray directions |
| `n_p` | 50 | Interpolation points per projected ray |
| `name` | auto | Label for the returned Prediction |

### `PICIP.suggest`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | — | Number of suggestion points |
| `min_dist` | None | Minimum spacing in constrained-basis units (1.0 = full field width) |
| `past_samples` | None | Extra exclusion points (default: the prediction's sample compositions) |

### `PICIP.suggest_kmedoids` (experimental)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | — | Number of suggestion points |
| `past_samples` | None | Compositions to treat as already-measured; batch is chosen to complement them |
| `exclude_radius` | None | Hard-exclude any omega point within this distance of a past sample |
| `max_candidates` | 1000 | Restrict the candidate pool to the top-K omega points by posterior probability (speedup; near-lossless) |

> Less tested than `suggest()`. See section 7.

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

## 12. Evaluating PICIP on synthetic phase fields

`picip.evaluation` builds phase fields whose answer is already known, hides one
phase, and measures how close PICIP gets to it. Use it to check a change, to
compare inference methods, or to decide how many experiments a question needs.

> **Worked example:** `tutorials/tutorial_evaluation.py` runs the whole workflow
> end to end — a single one-shot trial in 2-D and in 3-D with its diagnostic
> figure, then 50 one-shot 3-D trials comparing the Dirichlet and Gaussian
> methods as paired box plots, and finally the same comparison pooled over a
> larger population of varied 2-D fields. It takes about 25 seconds and writes
> four HTML files to `output/`.

Import it explicitly — it is not pulled in by `import picip`, because it caps
BLAS threads at import time for its parallel trial running:

```python
from picip.evaluation import (generate_test_phase_field, evaluate,
                              diagnostic_figure, distance_vs_batch_figure,
                              save_figure, summarise)
```

### Generate a field

```python
field = generate_test_phase_field(dim=3, structure='4c1a', n_internal=5, seed=0)
print(field.summary())
```

Dimensionality follows from charge neutrality: `n` elements give
`constrained_dim = n - 2`. Valid structures are `'2c2a'` and `'3c1a'` in 2-D,
`'4c1a'` and `'3c2a'` in 3-D. Generated phases are charge-balanced by
construction. `generate_phase_field_set` produces a list, sweeping phase counts.

### Run one-shot or iterative

```python
one_shot  = evaluate(field, batch_size=5)                 # propose once
iterative = evaluate(field, batch_size=3, n_batches=4)    # observe and repeat
```

Each internal phase is hidden in turn and `num_repeats` trials are run for each.
A trial draws a composition that produces the unknown phase, reads off what forms
and in what ratios, corrupts the weights with Dirichlet measurement error, and
asks PICIP for `batch_size` suggestions.

With `n_batches > 1` every suggestion is observed, added as a further sample, and
the next batch is drawn from the product of all densities so far — so PICIP
chooses where to look next. Suggestions that do not produce the unknown phase
teach nothing about it and are discarded (counted in `BatchResult.n_skipped`).

The score is `TrialResult.min_distance`: the distance from the unknown phase to
the closest suggestion in the final batch. `summarise(results)` prints the
per-batch means, and `n_jobs=-1` runs trials across all cores with results
identical to a serial run.

### Inspect one trial, stage by stage

```python
results = evaluate(field, batch_size=3, n_batches=4, diagnostic=True)
save_figure(diagnostic_figure(results[0]), 'trial.html')
```

One figure, one toggleable layer per stage of the algorithm — identical in 2-D
and 3-D, so the two can be read side by side:

| Layer | What it shows |
|-------|---------------|
| `Dirichlet support on known simplex` | The Dirichlet likelihood over the known simplex — the k1–k2 *line* in 2-D, the k1–k2–k3 *triangle* in 3-D. Belief about the knowns' mixing ratio, before projection |
| `projected p_density` | The raw ray points carrying that support outward through the sample, before interpolation |
| `interpolated p_density (sN)` | That sample's density on the omega grid. Hidden on load |
| `combined p_density` | Product-of-experts density over every sample so far. Hidden on load |
| `true direction` / `estimated direction` | Lines from the average known through the sample. The true line passes exactly through the unknown phase, so the gap between the two *is* the measurement error made visible |
| `batch (bN)` | The suggestions returned |
| `unknown (truth)` | Where the unknown phase actually is |

`diagnostic=True` is off by default because it retains every sample's projected
ray points. Pass `sample_index` / `batch_index` to draw an earlier round.

### Plot closest hit against batch number

```python
save_figure(distance_vs_batch_figure(iterative), 'convergence.html')
```

The closest-hit distance is the distance from the unknown phase to the nearest
point in a batch. The figure adapts to the run it is given:

- **Iterative runs** get a line of the mean against batch number with a ±1 SE
  band — a falling curve means the extra rounds pay for themselves. Every trial
  contributes one value per batch, so the comparison is paired within a trial.
- **One-shot runs** have a single batch, so there is nothing to plot against.
  The figure is a **box plot** of the per-trial distances instead — median,
  quartiles, whiskers, and every trial drawn as a point.

Pass a `dict` to put several runs on the same axes — side-by-side boxes for
one-shot runs, overlaid lines for iterative ones:

```python
save_figure(distance_vs_batch_figure({'dirichlet': d_runs, 'gaussian': g_runs}),
            'comparison.html')
```

### Saving figures

Figure functions return plotly `Figure` objects and write nothing on their own.
`save_figure(fig, path)` writes **one self-contained HTML file** with plotly
inlined — no `plotly.min.js` sidecar to keep beside it. Files are ~5 MB each;
that is the cost of a single portable artefact.

### Parameter tuning

Sweeps over `predicted_error`, resolution and inference method live in
`parameter_optimisation.py`, outside the package.

---

## 13. Reference

If you use PICIP in your work, please cite:

> Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
