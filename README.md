# **DebrisPy**
## *A Python Package for Computing the Radial Profiles of Surface Density in Debris Discs*

Welcome to **DebrisPy** — a lightweight package designed to compute the azimuthally averaged surface density (ASD) profiles in debris discs using both semi-analytical and Monte Carlo approaches.

![Demo](assets/demo.gif)


---

## **Installation**

1. The package can be installed via PyPI directly:

```bash
pip install debrispy
```

> **Important:** DebrisPy requires **Python 3.8 or higher**.


2. For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/DenizAkansoy/DebrisPy.git
cd DebrisPy
pip install -e .
```

---

## **Features**

DebrisPy provides tools for:

- defining semi-major-axis surface-density profiles;
- specifying unique eccentricity profiles or eccentricity distributions;
- constructing eccentricity kernels for ASD calculations;
- computing azimuthally averaged surface-density profiles;
- validating and visualising results with Monte Carlo sampling;
- using built-in profiles or arbitrary user-defined functions;
- optional adaptive gridding, interpolation, and parallelisation for more demanding calculations.

---

## **Documentation**

The full documentation is available online:

```text
https://debrispy.readthedocs.io
```

The documentation includes worked examples, API references, implementation notes, and notebook-based tutorials.

The documentation source files are located in:

```text
docs/source/
```

To build the documentation locally:

```bash
cd docs
make html
open build/html/index.html
```

---

## Repository structure

```text
debrispy/              Core package code
docs/source/           Sphinx documentation source
examples/              Example notebooks and scripts
tests/                 Test suite
assets/                README/demo assets
```

---

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

After cloning the repository, the test suite can be run with:

```bash
pytest tests/
```

For development, install the package with the optional development dependencies:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## **Testing**

To run the test suite, clone the repository and install the optional development dependencies:

```bash
git clone https://github.com/DenizAkansoy/DebrisPy.git
cd DebrisPy
pip install -e ".[dev]"
pytest tests/
```

---

## **Contact**

For questions, bug reports, or feedback, please open an issue on GitHub or contact Deniz Akansoy at `da619@cam.ac.uk`.