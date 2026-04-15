# PICIP Installation Guide

## Requirements

- Python 3.9 or later

## Option 1 — Install from PyPI

For use in your own scripts:

```bash
pip install picip
```

### Verify

```bash
python -c "from picip import PICIP, Phase_Field, Sample, make_plotter, Spread; print('PICIP installed successfully')"
```

---

## Option 2 — Clone the repository

This is only needed if you want to run the tutorials or contribute to development. If you installed via pip you can skip this.

```bash
git clone https://github.com/lrcfmd/PICIP.git
cd PICIP

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
```

The editable install (`-e`) means changes to files in `src/picip/` take effect immediately without reinstalling.

### Verify

```bash
python -c "from picip import PICIP, Phase_Field, Sample, make_plotter, Spread; print('PICIP installed successfully')"
```

### Tutorials

```bash
python tutorials/tutorial_setup.py   # phase field setup (all options)
python tutorials/tutorial_2d.py      # 2-D phase fields end to end
python tutorials/tutorial_3d.py      # 3-D phase fields end to end
python tutorials/tutorial_spread.py  # composition spreading
```

Each tutorial runs top to bottom and opens an interactive figure in your browser for each section.

---

## Reference

Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
