.. DebrisPy documentation master file, created by
   sphinx-quickstart on Mon May  5 18:39:51 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DebrisPy!
====================

`DebrisPy` is a Python package for modeling the surface density structure of debris discs using a semi-analytic formalism.
Based on the semi-analytic framework presented in `Rafikov (2023) <https://academic.oup.com/mnras/article/519/4/5607/6845736>`_, `DebrisPy` models the azimuthally averaged surface density profile, :math:`\bar{\Sigma}(r)`, based on arbitrary distributions of semi-major axis and orbital eccentricity. As explained in this documentation, the package offers robust tools for forward-modelling and visualisation.


	•	Flexible: Supports both deterministic and eccentricity distribution models, and allows custom surface density profiles.
	•	Efficient: Offers vectorized and optionally parallelized computation with support for grid refinement.
	•	Validated: Includes an independently implemented Monte Carlo sampler to verify the accuracy of semi-analytic ASD predictions.


Key Features
------------

- Compute and visualise :math:`\bar{\Sigma}(r)` from input :math:`\Sigma_a(a)` and :math:`\psi(e, a)` profiles.
- Perform 1D sampling of :math:`\bar{\Sigma}(r)` and 2D sampling of :math:`\Sigma(r, \phi)`.
- Compare analytic and sampling-based results with built-in plotting tools

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


.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


Contact
-------

This package was developed and is maintained by Deniz Akansoy
If you have any questions, feedback, or suggestions, feel free to reach out via: da619@cam.ac.uk