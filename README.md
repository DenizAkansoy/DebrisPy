[![PyPI version](https://img.shields.io/pypi/v/debrispy)](https://pypi.org/project/debrispy/)
[![Python versions](https://img.shields.io/pypi/pyversions/debrispy)](https://pypi.org/project/debrispy/)
[![Tests](https://github.com/DenizAkansoy/DebrisPy/actions/workflows/tests.yml/badge.svg)](https://github.com/DenizAkansoy/DebrisPy/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/debrispy/badge/?version=latest)](https://debrispy.readthedocs.io/)
[![License](https://img.shields.io/pypi/l/debrispy)](https://github.com/DenizAkansoy/DebrisPy/blob/main/LICENSE)

# **DebrisPy**

## *A Python Package for Computing the Radial Profiles of Surface Density in Debris Discs*

Welcome to **DebrisPy** — a lightweight Python package designed to compute the azimuthally averaged (radial) surface density profiles in debris discs using both semi-analytical and Monte Carlo approaches.

![Demo](assets/demo.gif)


## **Installation**

1. The package can be installed via PyPI directly:

```bash
pip install debrispy
```

2. For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/DenizAkansoy/DebrisPy.git
cd DebrisPy
pip install -e .
```

> **Important:** DebrisPy requires **Python 3.8 or higher**.


## **Features**

DebrisPy provides tools for:

- defining semi-major-axis surface-density profiles;
- specifying unique eccentricity profiles or eccentricity distributions;
- constructing eccentricity kernels for ASD calculations;
- computing azimuthally averaged surface-density profiles;
- validating and visualising results with Monte Carlo sampling;
- using built-in profiles or arbitrary user-defined functions;
- optional adaptive gridding, interpolation, and parallelisation for more demanding calculations.


### *Important Note: custom functions must be vectorised*

User-supplied functions should be vectorised. DebrisPy evaluates many profiles and distributions on NumPy arrays, so scalar Python conditionals such as `if`/`else` will usually fail or behave incorrectly.

For example, avoid:

```python
def bad_profile(a):
    if a < 50:
        return 0.0
    return a**-1
```

Use NumPy-aware operations instead:

```python
import numpy as np

def good_profile(a):
    return np.where(a < 50, 0.0, a**-1)
```

Boolean masks and array arithmetic are also suitable. 


## **Documentation**

The full documentation is available at [debrispy.readthedocs.io](https://debrispy.readthedocs.io).

The documentation includes worked examples, API references, implementation notes, and notebook-based tutorials.

The documentation source files are also located in:

```text
docs/source/
```

## Repository structure

```text
debrispy/              Core package code
docs/source/           Sphinx documentation source
examples/              Example notebooks and scripts
tests/                 Test suite
assets/                README/demo assets
```


## **Examples**

Example notebooks are provided in the `examples/` directory. These demonstrate how to define input profiles, construct eccentricity kernels, compute ASD profiles, and compare semi-analytic calculations with Monte Carlo realisations.

---

## **Dependencies**

Core dependencies are installed automatically when installing DebrisPy with:

```bash
pip install debrispy
```

These include:

- `numpy`
- `scipy`
- `matplotlib`
- `fast_histogram`
- `adaptive`
- `tqdm`
- `joblib`

Additional optional dependencies are needed for development, testing, and building the documentation.

For development and testing:

```bash
pip install -e ".[dev]"
```

This installs additional packages such as:

- `pytest`
- `ipykernel`
- `notebook`

For building the documentation locally:

```bash
pip install -e ".[docs]"
```

This installs additional packages such as:

- `sphinx`
- `sphinx-rtd-theme`
- `myst-parser`
- `nbsphinx`


## **Testing**

To run the test suite, clone the repository and install the optional development dependencies:

```bash
git clone https://github.com/DenizAkansoy/DebrisPy.git
cd DebrisPy
pip install -e ".[dev]"
pytest tests/
```

## **Acknowledgments**

If you use DebrisPy (or parts of its code) in your research, we would greatly appreciate it if you cited the paper introducing the package:

```bibtex
@article{Rafikov2026,
  author        = {Roman R. Rafikov and Deniz Akansoy and Antranik A. Sefilian},
  title         = {Debris Disc Substructures Induced by Secular Planetary Perturbations},
  journal       = {arXiv e-prints},
  year          = {2026},
  eprint        = {2607.08750},
  archiveprefix = {arXiv},
  primaryclass  = {astro-ph.EP},
  doi           = {10.48550/arXiv.2607.08750}
}
```

## **Contact**

For questions, bug reports, or feedback, please open an issue on GitHub or contact Deniz Akansoy at `da619@cam.ac.uk`.