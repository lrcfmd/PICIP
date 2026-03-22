# PICIP Installation Guide

## Requirements

- Python 3.9 or later
- The following packages (all available on PyPI):

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy` | Interpolation, statistics, spatial algorithms |
| `pymatgen` | Composition parsing and conversion |
| `plotly` | Interactive figures |
| `pandas` | CSV export for suggestions |
| `matplotlib` | Auxiliary plotting |

## Install

Clone the repository and install dependencies into a virtual environment:

```bash
git clone <repo-url>
cd PICIP

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r src/requirements.txt
```

## Verify

From the project root (after activating the virtual environment):

```bash
cd src
python -c "from phase_field import Phase_Field; from picip import PICIP, Sample; from visualise_cube import make_plotter; print('PICIP installed successfully')"
```

> **Note:** this command must be run from the `src/` directory — the modules are not installed as a package and must be imported from their location.

## Usage

All scripts are run from the `src/` directory:

```bash
cd src
python tutorial_2d.py    # 2-D phase field tutorial
python tutorial_3d.py    # 3-D phase field tutorial
```

Each tutorial section opens an interactive figure in your browser.

## Reference

Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
