Quick Start Guide
=================

This section helps you install and start using `DebrisPy` with a minimal example.

Installation
------------

.. note::

   `DebrisPy` requires **Python 3.8 or greater**.

At present, the code is available exclusively through the internal GitLab repository, in order to access it:

1. First, download the coursework repository (`da619`) from GitLab.

2. Then, navigate to the package directory and install in editable mode:

.. code-block:: bash

   cd da619/debrispy
   pip install -e .

This allows you to run and modify the package locally without reinstalling.

.. note::

   In the future, `DebrisPy` will be uploaded as a public release on both 
   `GitHub <https://github.com/denizakansoy/debrispy>`_ and the Python Package Index (PyPI), allowing open access and one-line installation via:

   .. code-block:: bash

      pip install debrispy

Basic Usage
-----------

Here is a minimal working example to get started.

.. code-block:: python

   from debrispy import SigmaA, RayleighEccentricity, Kernel, ASD
   import numpy as np

   # 1. Define surface density profile with respect to semi-major axis
   sigma_a = SigmaA(a_min=1.0, a_max=5.0, profile_type='powerlaw', sigma0 = 1.0, power=0.5)

   # 2. Define eccentricity distribution
   ecc = RayleighEccentricity(a_min=1.0, a_max=5.0, sigma0=0.05, alpha=0.0)

   # 3. Initialise and compute the kernel
   kernel = Kernel(eccentricity_profile=ecc)
   kernel.compute_kernel()

   # 4. Compute surface density for 300 radial points
   asd = ASD(kernel, sigma_a)
   asd.compute_quadvec(r_vals = np.linspace(0.1, 5.0, 300))

   # 5. Plot results
   asd.plot()

More details can be found in the following sections, which includes comprehensive examples and explanations for each component.