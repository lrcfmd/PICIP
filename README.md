# PICIP

**Probabilistic Isolation of Inorganic Crystalline Phases** — Bayesian inference for predicting the composition of an unknown phase in a multi-phase XRD sample.

## Installation

```bash
pip install picip
```

## Quick start

```python
from picip import PICIP, Phase_Field, Sample, make_plotter

pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

sample = Sample("s1", "Fe2Mn4Ti4")
sample.add_knowns(["FeMn", "FeTi"])
sample.add_mass_weights([0.3, 0.7])
sample.set_predicted_error(0.3)

picip = PICIP(pf)
picip.add_sample(sample)
pred = picip.run()

pl = make_plotter(pf)
pl.plot_prediction_results(pred)
pl.show()
```

## Documentation

- [Installation Guide](https://github.com/lrcfmd/PICIP/blob/main/PICIP_installation.md) — pip install, clone for tutorials, verify setup
- [User Manual](https://github.com/lrcfmd/PICIP/blob/main/PICIP_manual.md) — full usage guide with worked examples

## Paper code

The code used to generate results for the paper is preserved on the [`PICIP_paper`](https://github.com/lrcfmd/PICIP/tree/PICIP_paper) branch.

## Citation

Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
