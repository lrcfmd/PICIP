# PICIP

**Probabilistic Isolation of Inorganic Crystalline Phases** — Bayesian inference for predicting the composition of an unknown phase in a multi-phase XRD sample.

After a Rietveld refinement you know the overall sample composition, the identities of the phases you recognised, and their mass fractions — but nothing about the unknown phase that is also present. PICIP places a probability density over the whole composition space, returns the most likely compositions for that unknown phase, and suggests what to synthesise next.

## Installation

```bash
pip install picip
```

New to Python, git or the command line? The [Installation Guide](https://github.com/lrcfmd/PICIP/blob/main/PICIP_installation.md) starts from scratch.

## Quick start

```python
from picip import Phase_Field, PICIP, Sample, make_plotter

# The composition space: three elements, no charge constraint → a 2-D triangle
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

# One measured sample: what you refined, and what you identified in it
sample = Sample("s1", "Fe2Mn4Ti4")     # overall measured composition
sample.add_knowns(["FeMn", "FeTi"])    # phases you identified
sample.add_mass_weights([0.3, 0.7])    # their Rietveld mass fractions
sample.set_predicted_error(0.3)        # how much you trust those fractions

# Infer where the unknown phase is
picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run()

# Ask what to make next, and look at the result
suggestions = picip.suggest(pred, n=5, min_dist=0.05)
pl = make_plotter(pf)
pl.plot_prediction_results(pred, plot_average_known=True)
pl.show()
```

`suggestions.constrained[0]` is the single most likely composition for the unknown phase; the rest are spread across the density so a batch of experiments covers the plausible region. `suggestions.save("next.csv")` writes them out.

## Documentation

- [Installation Guide](https://github.com/lrcfmd/PICIP/blob/main/PICIP_installation.md) — setup from scratch, dependencies, troubleshooting
- [User Manual](https://github.com/lrcfmd/PICIP/blob/main/PICIP_manual.md) — quick start, then every option in full
- `tutorials/` — runnable scripts covering setup, 2-D, 3-D, composition spreading, and evaluation (clone the repo to get them)

## Paper code

The code used to generate results for the paper is preserved on the [`PICIP_paper`](https://github.com/lrcfmd/PICIP/tree/PICIP_paper) branch.

## Citation

Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
