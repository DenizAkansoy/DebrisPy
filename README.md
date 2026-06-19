# **DebrisPy**
## *A Python Package for Computing the Radial Profiles of Surface Density in Debris Discs*

Welcome to **DebrisPy** — a lightweight package designed to compute the azimuthally averaged surface density (ASD) profiles in debris discs using both semi-analytical and Monte Carlo approaches ([see full documentation](#4-documentation) for usage and API reference).

![Demo](assets/demo.gif)

---

## **Table of Contents**

1. [Repository Structure](#1-repository-structure)  
2. [Installation](#2-installation)  
3. [Dependencies](#3-dependencies)  
4. [Documentation](#4-documentation)  
5. [Example Notebooks](#5-example-notebooks)  
6. [Testing](#6-testing)

---

## **1. Repository Structure**

Breakdown of the repository's folder structure and its purpose:

- **`debrispy/`**  
  Core package code, including all class implementations.

- **`tests/`**  
  Contains a test suite using `pytest`. See [Section 6](#6-testing) for further information.

- **`examples/`**  
  Example files (both `.py` and `.ipynb`) showcasing how to use the package.

- **`docs/`**  
  Contains the full Sphinx-generated documentation (both the source files and built html files). The documentation can easily be accessed online without requiring manual building or download. See [Section 4](#4-documentation) for viewing instructions.


---

## **2. Installation**

> **Important:** DebrisPy requires **Python 3.8 or higher**.

To install the `DebrisPy` package locally:

1. Clone the repository from the GitLab directory:


```bash
git clone <repository-url>
cd DebrisPy  # Navigate to the parent directory
```

2. Install the python package:


```bash
pip install . 
```

---

## **3. Dependencies**

All required dependencies are automatically installed when installing the package via `pip`.

**Numerical and Scientific Computing**

- `numpy`: fast array manipulation and vectorised math
- `scipy`: numerical integration, special functions, and interpolation
- `fast_histogram`: high-performance 1D/2D histogramming
- `adaptive`: optional grid refinement and adaptive sampling
- `matplotlib`: 1D and 2D surface density plotting
- `tqdm`: progress bars for long-running sampling routines
- `joblib`: parallel execution for kernel computations


---

## **4. Documentation**

The documentation contains information on each class within the package, and provides examples on various use cases. This can be accessed online via debrispy.readthedocs.io

The source files for the documentation are also provided in the main repository.



---

## **5. Example Notebooks**

This repository includes a collection of Jupyter notebooks (found in the `notebooks` directory) that demonstrate how to use the DebrisPy package in practice. These are organised into two subdirectories:

- `docs_notebooks/`
These notebooks form the main section of the documentation. They are well-annotated, with clear, step-by-step examples showing how to initialise and use each of the core classes (e.g., SigmaA, Kernel, ASD, MonteCarlo).*We recommend looking through these notebooks before the others, as they provide the most accessible introduction to the package.*


- `report_notebooks/`
These notebooks were used to generate all figures and results shown in the MPhil report.
While not structured as tutorials, they are still lightly commented and provide insight into how the package can be applied in research scenarios, including benchmarking, model comparison, and analysis of specific case studies.

---

## **6. Tests**

This package includes a set of automated tests using the `pytest` framework, located in the `tests/` directory.

> `pytest` is a lightweight Python framework for writing and running test functions to automatically verify that code behaves as expected.

After installing the package, we recommend running the test suite to ensure that the package has been installed correctly. This can be run from the root directory, by executing:

```bash
pytest tests/
```

---

For any further questions regarding usage, please see the documentation. Feel free to contact Deniz Akansoy via da619@cam.ac.uk, any feedback would be greately appreciated.