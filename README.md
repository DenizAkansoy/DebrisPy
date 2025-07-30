# **DebrisPy**
## *A Python Package for Computing the Radial Profiles of Surface Density in Debris Discs*

Welcome to **DebrisPy** — a lightweight package designed to compute the azimuthally averaged surface density (ASD) profiles in debris discs using both semi-analytical and Monte Carlo approaches ([see full documentation](#4-documentation) for usage and API reference).

![Demo](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHd5ZDRwdnFzY3c3ZDdueW4xMWJibXF2dWF6aDNqY2VqaHB4Y2RnYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/92kAjwIuxmnmhX3QN3/giphy.gif)

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
  Core package code, including all class implementations and the `__init__.py` interface.

- **`report/`**  
  Contains the full MPhil report and an executive summary outlining key results.

- **`docs/`**  
  Contains the full Sphinx-generated documentation (pre-built). See [Section 4](#4-documentation) for viewing instructions.

- **`tests/`**  
  Contains a test suite using `pytest`. See [Section 6](#6-testing) for information.

- **`notebooks/`**  
  Jupyter notebooks used for documentation, benchmarking, and generating figures for the final report.

---

## **2. Installation**

> **Important:** DebrisPy requires **Python 3.8 or higher** due to internal dependencies.

To install the `DebrisPy` package locally:

1. Clone the repository from the GitLab directory:


```bash
git clone <repository-url>
cd da619  # Navigate to the parent directory
```

2. Install the python package:


```bash
pip install -e . # Install in editable mode (make sure to be in the root directory)
```

This installs the package in editable mode, meaning any local code changes will immediately take effect without reinstalling. 

Installing in editable mode allows for rapid development and testing. However, note that some IDEs or code editors (like VS Code or PyCharm) may not immediately recognise the package (e.g. by underlining imports or showing unresolved references).

If you encounter issues when using `-e` (e.g. `ModuleNotFoundError` in Jupyter notebooks or virtual environments), try reinstalling without the editable flag.

> **Note on Python Version**
>
> `DebrisPy` requires **Python 3.8 or higher**.  
> You can check your current Python version by running:
>
> ```bash
> python --version
> ```
>
> If your version is lower than 3.8, consider creating a clean Python environment with a more recent version. Here are two easy options:
>
> - **Using Conda (recommended):**
>
>   ```bash
>   conda create -n debrispy_env python=3.10
>   conda activate debrispy_env
>   ```
>
> - **Using Homebrew (macOS only):**
>
>   ```bash
>   brew install python@3.10
>   echo 'export PATH="/opt/homebrew/opt/python@3.10/bin:$PATH"' >> ~/.zprofile
>   source ~/.zprofile
>   python3.10 --version  # check it's working
>   ```
>
> After that, you can install the package as described above.
>
> **Compatibility Note:**
> `DebrisPy` was developed using Python 3.11, and has been tested on additional Python versions 3.8+.
>
> If using Python 3.13+, you may see *DeprecationWarnings* (e.g., related to `np.trapz`). This does not affect results or performance at all, and will be addressed in future updates to the package.


---

## **3. Dependencies**

All required dependencies are automatically installed when using `pip`.

**Numerical and Scientific Computing**

- `numpy`: fast array manipulation and vectorised math
- `scipy`: numerical integration, special functions, and interpolation
- `fast_histogram`: high-performance 1D/2D histogramming
- `adaptive`: optional grid refinement and adaptive sampling
- `matplotlib`: 1D and 2D surface density plotting
- `tqdm`: progress bars for long-running sampling routines

**Parallelism and Utilities**

- `joblib`: parallel execution for kernel computations
- `functools`, `itertools`: functional programming utilities
- `typing`: static type annotations for clarity and linting
- `warnings`: controlled user warnings for numerical operations


---

## **4. Documentation**

This contains information on each class within the package, and provides examples on various use cases. The documentation is provided pre-built for convenience.

### Viewing Pre-Built Documentation

To view the pre-built documentation (for which you do not need to have installed the package),  run the following command from the root directory:

```bash
open docs/build/html/index.html
```

This serves as the main menu of the documentation.

### Manually Building Documentation (Optional)

If you prefer to manually re-build the documentation, follow these steps:

1.	Ensure both the Python version and Cythonized version are installed (see Section 2).

2.	Navigate to the docs directory:

```bash
cd ../docs
```

3. Clean any previous builds and re-build: 

```bash
make clean
make html
```

> Note: Building the documentation is only necessary if you’ve made changes to the source code or notebooks and want to regenerate the outputs.

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

To run the test suite from the root directory, simply execute:

```bash
pytest tests/
```

These tests are included as a proof of concept for best practices, currently there are 55 total tests, testing all of the classes within the ASD pipeline. The functionality of the package was manually tested throughout development, however, as the package itself grows in functionality, the test suite will also grow, becoming more relevant.

---

For any further questions regarding usage, please see the documentation (or report) for an in-depth explanation.