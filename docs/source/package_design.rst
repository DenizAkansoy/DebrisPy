Package Design
==============

This section outlines the modular structure of ``DebrisPy`` and summarises how
the main components of the package interconnect.

ASD pipeline and data flow
--------------------------

The diagram below summarises the data flow and functional organisation of
``DebrisPy``. Each numbered step corresponds to a core component in the ASD
calculation pipeline.

.. image:: _static/diagram.jpeg
   :alt: DebrisPy package flowchart
   :width: 100%
   :align: center


1. **Mass distribution in semi-major axis** — :math:`\Sigma_a(a)`

   The pipeline begins with the user-supplied mass distribution in semi-major
   axis. This may be chosen from one of the built-in profiles, such as a power
   law, Gaussian, or step function, or supplied as a custom vectorised function
   (see :doc:`sigma_a`).

2. **Eccentricity model** — :math:`e(a)` or :math:`\psi_e(e,a)`

   The user then specifies the eccentricity structure of the disc. ``DebrisPy``
   supports both deterministic eccentricity profiles and full eccentricity
   distributions:

   - **Unique eccentricity**: a deterministic mapping :math:`e=e(a)`
     (see :doc:`unique_ecc`).
   - **Built-in distributions**: predefined eccentricity distributions, such
     as the Rayleigh distribution (see :doc:`builtin_ecc_distribution`).
   - **User-supplied distributions**: arbitrary vectorised functions
     :math:`\psi_e(e,a)`, with optional numerical normalisation
     (see :doc:`custom_ecc_distribution`).

   User-supplied functions must be vectorised, since ``DebrisPy`` evaluates
   profiles and distributions on NumPy arrays. Use NumPy-aware operations such
   as ``np.where``, boolean masks, and array arithmetic instead of scalar
   Python ``if``/``else`` statements.

3. **Kernel calculation** — :math:`\Phi_e(r,a)`

   For each eccentricity model, ``DebrisPy`` constructs the eccentricity kernel,
   :math:`\Phi_e(r,a)`. Analytic kernel expressions are used where available;
   otherwise the kernel is evaluated numerically using one of the supported
   integration schemes. The kernel can be thought of as encoding how material
   with semi-major axis :math:`a` contributes to the radial surface density at
   radius :math:`r` (see :doc:`kernel`).

4. **ASD integration** — :math:`\bar{\Sigma}(r)`

   Finally, the package computes the azimuthally averaged surface-density
   profile by evaluating

   .. math::

      \bar{\Sigma}(r)
      =
      \pi^{-1}
      \int_{r/2}^{\infty}
      a^{-1}
      \Sigma_a(a)
      \Phi_e(r,a)
      \,\mathrm{d}a .

   The calculation is performed on a user-specified radial grid, with optional
   CPU parallelisation and adaptive radial refinement for sharply structured
   profiles (see :doc:`asd`).


Each stage is handled by a dedicated module with a consistent interface. This
modular design allows individual components to be tested, validated, reused, or
replaced independently, while keeping the full ASD pipeline compact and
transparent.

Intermediate quantities, such as eccentricity kernels and interpolated profile
values, are cached where possible to avoid unnecessary recomputation. These
quantities remain accessible to the user, making it possible to inspect or
reuse intermediate stages of the calculation.

Additional functionality
------------------------

In addition to the main semi-analytic ASD pipeline, ``DebrisPy`` includes a
Monte Carlo sampler for generating particle realisations of the same underlying
orbital distributions (see :doc:`monte_carlo`). These samples can be used to
construct one-dimensional radial histograms or two-dimensional maps in
Cartesian or polar coordinates, providing an independent check on the
semi-analytic calculation and a useful visualisation of the orbital structure.

``DebrisPy`` also includes convolution utilities for comparing high-resolution
model profiles and maps with observationally resolved data. One-dimensional
profiles can be convolved with Gaussian kernels, while two-dimensional
Cartesian maps support Gaussian point-spread functions, including elliptical
and rotated beams.

Dependencies
------------

``DebrisPy`` requires **Python 3.8 or higher**. Core dependencies are installed
automatically when installing the package with:

.. code-block:: bash

   pip install debrispy

The core dependencies are:

- ``numpy``: array manipulation and vectorised numerical operations
- ``scipy``: numerical integration, interpolation, and scientific utilities
- ``matplotlib``: plotting and visualisation
- ``fast_histogram``: high-performance one- and two-dimensional histogramming
- ``adaptive``: adaptive sampling and grid refinement utilities
- ``tqdm``: progress bars for long-running calculations
- ``joblib``: optional CPU parallelisation

Optional dependencies are available for development, testing, and documentation.

For development and testing:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional packages such as ``pytest``, ``ipykernel``, and
``notebook``.

For building the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"

This installs additional packages such as ``sphinx``, ``sphinx-rtd-theme``,
``myst-parser``, and ``nbsphinx``.