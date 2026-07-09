Quick Start Guide
=================

This section helps you install and start using `DebrisPy` with a minimal example.

Installation
------------

``DebrisPy`` is available on PyPI and can be installed with:

.. code-block:: bash

   pip install debrispy

For development, the package can also be installed from a local clone:

.. code-block:: bash

   git clone https://github.com/DenizAkansoy/DebrisPy.git
   cd DebrisPy
   pip install -e .

Basic Usage
-----------

Here is a minimal working example to get started.

.. code-block:: python

   import debrispy import dp
   import numpy as np

   # 1. Define surface density profile with respect to semi-major axis
   sigma_a = dp.SigmaA(a_min=1.0, a_max=5.0, profile_type='powerlaw', sigma0 = 1.0, power=0.5)

   # 2. Define eccentricity distribution (e.g. Rayleigh distribution of eccentricities)
   ecc = dp.RayleighEccentricity(a_min=1.0, a_max=5.0, sigma0=0.05, alpha=0.0)

   # 3. Initialise and compute the kernel
   kernel = dp.Kernel(eccentricity_profile=ecc)
   kernel.compute_kernel()

   # 4. Compute surface density for 300 radial points
   asd = dp.ASD(kernel, sigma_a)
   asd.compute_quadvec(r_vals = np.linspace(0.1, 5.0, 300))

   # 5. Plot results
   asd.plot()

More details can be found in the following sections, which includes comprehensive examples and explanations for each component.

Important note on custom functions
----------------------------------

User-supplied functions must be vectorised. ``DebrisPy`` evaluates many input
profiles and distributions on NumPy arrays, so scalar Python conditionals such
as ``if``/``else`` will usually fail or behave incorrectly. Use NumPy-aware
operations such as ``np.where``, boolean masks, and array arithmetic instead.

For example, avoid scalar conditionals:

.. code-block:: python

   def bad_profile(a):
       if a < 50:
           return 0.0
       return a**-1

Use a vectorised version instead:

.. code-block:: python

   import numpy as np

   def good_profile(a):
       return np.where(a < 50, 0.0, a**-1)