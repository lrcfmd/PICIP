# PICIP Installation Guide

**Requirements:** Python 3.9 or later. PICIP runs on Windows, macOS and Linux.

If you already have Python and know your way around a terminal, skip to
[Option 1](#option-1--install-from-pypi).

---

## Before you start

PICIP is a Python library, so you need a few standard tools first. Each one is
linked to its official guide — install them in this order.

| You need | Why | Guide |
|---|---|---|
| **Python 3.9+** | The language PICIP is written in | [python.org/downloads](https://www.python.org/downloads/) — on Windows, tick **Add Python to PATH** during install |
| **A terminal** | Where you type the commands below | [Terminal basics](https://www.freecodecamp.org/news/command-line-for-beginners/) — Terminal on macOS/Linux, PowerShell on Windows |
| **A code editor** | To read and edit `.py` files | [VS Code](https://code.visualstudio.com/) is the usual choice; install its [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) |
| **git** | To download the repository and get updates | [git-scm.com/downloads](https://git-scm.com/downloads) |

Two Python concepts worth ten minutes before you begin:

- **`pip`** installs Python packages from the internet.
  [Official quickstart](https://packaging.python.org/en/latest/tutorials/installing-packages/).
- **Virtual environments (`venv`)** keep each project's packages separate, so
  installing PICIP cannot break another project.
  [Python docs](https://docs.python.org/3/library/venv.html). Strongly
  recommended — every example below assumes one.

Check your Python is visible from the terminal before continuing:

```bash
python --version        # if this errors, try python3 --version
```

If it prints `Python 3.9.x` or higher, you are ready.

---

## Option 1 — Install from PyPI

Use this if you just want to use PICIP in your own scripts.

```bash
python -m venv .venv                    # create a virtual environment
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install picip
```

You will need to run the `activate` line again in each new terminal session.
Your prompt shows `(.venv)` when it is active.

### Verify

```bash
python -c "from picip import PICIP, Phase_Field, Sample, make_plotter, Spread; print('PICIP installed successfully')"
```

Then go to the [User Manual](PICIP_manual.md) — its Quick start section is a
complete working example.

### Getting the example scripts

`pip install picip` gives you the library but not the example files, which live
only in the repository. You do not need to clone it for them — browse either
folder on GitHub and copy-paste, or use the **Download raw file** button:

- **[`scripts/`](https://github.com/lrcfmd/PICIP/tree/main/scripts)** — one
  short, self-contained quick-start script per phase-field type (uncharged,
  charged, mixed-valence, custom constraint), available for a quick start or
  for reference.
- **[`tutorials/`](https://github.com/lrcfmd/PICIP/tree/main/tutorials)** —
  longer walkthroughs with a section per parameter.

Save a file anywhere and run it with your virtual environment active; they
import `picip` like any other package. The tutorials write figures to
`../output/`, so either create an `output/` folder alongside the script or
change those paths.

---

## Option 2 — Clone the repository

Use this if you want the tutorials, or want to modify PICIP itself.

```bash
git clone https://github.com/lrcfmd/PICIP.git
cd PICIP

python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate

pip install -e .
```

`git clone` downloads the repository into a new `PICIP` folder. `pip install -e .`
installs it in **editable** mode, so any change you make to files in `src/picip/`
takes effect immediately without reinstalling — which is what you want while
developing.

To get later updates:

```bash
git pull
```

### Verify

```bash
python -c "from picip import PICIP, Phase_Field, Sample, make_plotter, Spread; print('PICIP installed successfully')"
```

### Dependencies

`pip` installs these automatically; you only need to know the names when
looking up how something works:

| Package | Used for | Docs |
|---|---|---|
| **numpy** | Arrays and linear algebra — nearly every PICIP value is a numpy array | [numpy.org/doc](https://numpy.org/doc/stable/user/absolute_beginners.html) |
| **scipy** | Geometry (convex hulls, triangulation) and interpolation | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) |
| **pymatgen** | Chemical formulas and compositions — the `"Fe2Mn4Ti4"` strings | [pymatgen.org](https://pymatgen.org/) |
| **plotly** | The interactive figures that open in your browser | [plotly.com/python](https://plotly.com/python/) |
| **pandas** | Tabular output and CSV export | [pandas.pydata.org](https://pandas.pydata.org/docs/) |

---

## Running the scripts and tutorials

`scripts/` holds one short, self-contained quick-start script per phase-field
type, available for a quick start or for reference:

```bash
python scripts/quickstart_uncharged.py                  # no charge constraint
python scripts/quickstart_charged.py                    # fixed oxidation states
python scripts/quickstart_charge_ranges.py              # mixed valence
python scripts/quickstart_custom_constraint.py          # custom stoichiometry
python scripts/quickstart_charged_3knowns_gaussian.py   # 3 knowns → Gaussian method
```

The tutorials are longer, with a section per parameter:

```bash
python tutorials/tutorial_setup.py       # phase field setup — all four methods
python tutorials/tutorial_2d.py          # 2-D fields end to end, uncharged and charged
python tutorials/tutorial_3d.py          # 3-D fields end to end, uncharged and charged
python tutorials/tutorial_spread.py      # composition spreading
python tutorials/tutorial_evaluation.py  # scoring PICIP against synthetic fields
```

Run them from the repository root with your virtual environment active. Each
script runs top to bottom and is divided into numbered sections; read the file
alongside its output. Most open interactive figures in your browser as they go —
`tutorial_evaluation.py` writes its figures to `output/` instead, and needs no
input data of its own because it generates the phase fields it tests against.

`tutorial_2d.py` is the shortest complete path through the library.

---

## Finding out how something works

**Read the docstring.** Every public function and class in PICIP documents its
parameters and return value in a docstring, alongside comments explaining the
steps. Go to the function in the source and read it there — in VS Code,
hovering over a function name shows the docstring and <kbd>F12</kbd> (or
ctrl-click) jumps straight to its definition.

**Read the source.** PICIP is small and heavily commented. The files worth
knowing:

| File | Contains |
|---|---|
| `src/picip/phase_field.py` | `Phase_Field` — the composition grid and its geometry |
| `src/picip/picip.py` | `PICIP`, `Sample`, `Prediction` — the inference itself |
| `src/picip/visualise_square.py` / `visualise_cube.py` | The 2-D and 3-D plotters |
| `src/picip/spread.py` | `Spread` — composition spreading |
| `src/picip/evaluation.py` | Synthetic phase fields for scoring PICIP |

**Look up the library.** If a line uses something unfamiliar, it is usually
numpy or pymatgen rather than PICIP — see the dependency table above for links.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python: command not found` | Python is not installed, or not on PATH. Reinstall and tick *Add Python to PATH*, or try `python3`. |
| `ModuleNotFoundError: No module named 'picip'` | The virtual environment is not active (no `(.venv)` in your prompt), or the install failed. Re-run `activate`, then `pip install picip`. |
| `pip: command not found` | Use `python -m pip` instead of `pip`. |
| Figures do not appear | Plotly opens your default browser. If nothing happens, pass `save="name"` to `show()` and open the resulting `name.html` yourself. |
| `Error, point ... is not balanced` printed | A constraint check failed. On a charged field it means the composition really is not charge balanced. On an uncharged field with precursors that reduce the dimension, it can also indicate the point lies outside the reduced composition subspace. |

For anything else, copy the full error message into a search engine — nearly
every common Python installation problem has a clear answer on Stack Overflow.

---

## Reference

Ritchie, D.; Gaultois, M. W.; Gusev, V. V.; Kurlin, V.; Rosseinsky, M. J.; Dyer, M. S. Probabilistic Isolation of Crystalline Inorganic Phases. *Journal of Chemical Information and Modeling* **2025**, *65* (24), 13226–13237. https://doi.org/10.1021/acs.jcim.5c02256
