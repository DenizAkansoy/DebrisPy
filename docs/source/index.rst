Welcome to DebrisPy
===================

.. image:: https://img.shields.io/badge/GitHub-DebrisPy-blue?logo=github
   :target: https://github.com/DenizAkansoy/DebrisPy
   :alt: GitHub repository

.. image:: https://img.shields.io/pypi/v/debrispy
   :target: https://pypi.org/project/debrispy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/debrispy
   :target: https://pypi.org/project/debrispy/
   :alt: Python versions

.. image:: https://github.com/DenizAkansoy/DebrisPy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/DenizAkansoy/DebrisPy/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://img.shields.io/pypi/l/debrispy
   :target: https://github.com/DenizAkansoy/DebrisPy/blob/main/LICENSE
   :alt: License


``DebrisPy`` is a Python package for modelling the surface-density structure of
eccentric debris discs using a semi-analytic formalism.

Based on the framework presented in `Rafikov (2023) <https://academic.oup.com/mnras/article/519/4/5607/6845736>`_,
``DebrisPy`` computes azimuthally averaged surface-density profiles,
:math:`\bar{\Sigma}(r)`, from prescribed distributions of semi-major axis and
orbital eccentricity.

The package is designed for forward-modelling debris-disc structures and for
exploring how eccentric parent-body distributions map into observable radial
surface-density profiles.

Key features
------------

- Compute and visualise azimuthally averaged surface-density profiles,
  :math:`\bar{\Sigma}(r)`, from input :math:`\Sigma_a(a)` and
  :math:`\psi(e,a)` profiles.
- Support both deterministic eccentricity profiles, :math:`e(a)`, and full
  eccentricity distributions, :math:`\psi(e,a)`.
- Define built-in or user-supplied semi-major-axis and eccentricity profiles.
- Construct eccentricity kernels and evaluate the corresponding ASD integral.
- Perform Monte Carlo sampling as an independent check of the semi-analytic
  calculation.
- Generate 1D radial profiles and 2D surface-density realisations.
- Use optional adaptive gridding, interpolation, and parallelisation for more
  demanding calculations.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   quick_start
   theory
   package_design

.. toctree::
   :maxdepth: 1
   :caption: Package Structure

   sigma_a
   unique_ecc
   builtin_ecc_distribution
   custom_ecc_distribution
   kernel
   asd
   monte_carlo

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   benchmarks
   api_reference

Citing DebrisPy
---------------

If you use DebrisPy, or parts of its code, in your research, we would
greatly appreciate it if you cited the paper introducing the package:

`Debris Disc Substructures Induced by Secular Planetary Perturbations
<https://arxiv.org/abs/2607.08750>`_

.. code-block:: bibtex

   @article{Rafikov2026,
     author        = {Rafikov, Roman R. and Akansoy, Deniz and Sefilian, Antranik A.},
     title         = {Debris Disc Substructures Induced by Secular Planetary Perturbations},
     journal       = {arXiv e-prints},
     year          = {2026},
     eprint        = {2607.08750},
     archiveprefix = {arXiv},
     primaryclass  = {astro-ph.EP},
     doi           = {10.48550/arXiv.2607.08750}
   }

Contact
-------

This package was developed and is maintained by Deniz Akansoy.

For questions, feedback, or suggestions, please open an issue on GitHub or
contact Deniz Akansoy at ``da619@cam.ac.uk``.