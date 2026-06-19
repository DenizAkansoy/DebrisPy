Theoretical Framework
=====================

The central goal of `DebrisPy` is to predict the azimuthally averaged surface density profile, :math:`\bar{\Sigma}(r)`, from a physically motivated distribution of orbital elements.

.. note::
   For a complete derivation and foundational formalism, see `Rafikov (2023) <https://academic.oup.com/mnras/article/519/4/5607/6845736>`_.


The surface density of a debris disc, :math:`\Sigma(r, \phi)`, defined in polar coordinates, is directly linked to the disc's observed luminosity. In this work, we focus on the *azimuthally averaged* surface density, :math:`\bar\Sigma(r)`, which captures the radial structure of the disc after integrating out angular dependence,

.. math::

    \bar\Sigma(r) = \frac{1}{2\pi} \int_0^{2\pi} \Sigma(r, \phi)\,\mathrm{d}\phi.

The theoretical foundation for this calculation is based on the framework introduced in Rafikov (2023), which we implement in a modular and extensible form in `DebrisPy`. In this section, we summarise the derivation of :math:`\bar\Sigma(r)` under this approach.

The radial position of a particle in orbit with semi-major axis :math:`a` and eccentricity :math:`e` is determined by the Keplerian relation,

.. math::

    r = \frac{a(1 - e^2)}{1 + e \cos(\phi - \varpi)},

where :math:`\varpi` is the apsidal angle relative to a fixed direction. In such a case, the orbit ranges between the periastron, :math:`r_p = a(1 - e)`, and apoastron, :math:`r_a = a(1 + e)`, distances. This range is particularly important when determining which particles contribute to the matter density at a specific orbital distance.

To proceed, we first define the mass distribution over semi-major axis. Specifically, we let :math:`\mu(a)` be the total mass per unit :math:`a`, such that :math:`\mathrm{d}m = \mu(a) \mathrm{d}a` is the mass of the particles which belong in the :math:`(a, \mathrm{d}a)` interval. This allows us to introduce the associated surface density profile in :math:`a`-space,

.. math::

    \Sigma_a(a) = \frac{1}{2\pi a} \frac{\mathrm{d}m}{\mathrm{d}a} = \frac{\mu(a)}{2\pi a}.

Although this is not a physical quantity, it plays a central role in defining and calculating :math:`\bar\Sigma(r)`. The second quantity of central importance is the eccentricity profile of the particles.

The remainder of the derivation depends on whether eccentricity is specified uniquely, :math:`e = e(a)`, or instead follows a distribution :math:`\psi_e(e, a)`. In both cases, we focus solely on the azimuthally averaged density, so the results are independent of the apsidal angle, :math:`\varpi`. This is because the time-averaged spatial distribution of particles at an orbital distance :math:`r` (i.e. the time a particle spends at a radial distance) is unaffected by orbital orientation which is defined via :math:`\varpi`.

Eccentricity as a Unique Function of Semi-Major Axis
----------------------------------------------------

We now consider the case where :math:`e = e(a)`, which is equivalent to assuming the eccentricity distribution is a delta function in :math:`a`-space, :math:`\psi_e(e,a) = \delta(e - e(a))`.

It is helpful to define the boundary condition that determines whether or not particles on orbits with given :math:`(a, e)` can contribute to the surface density at a particular radial location :math:`r`, arising from :math:`r_p \leq r \leq r_a`.

This requirement can be rewritten as a lower bound on the eccentricity :math:`e(a)` for a given semi-major axis :math:`a`. Specifically,

- For :math:`r < a`, the constraint becomes :math:`e(a) > 1 - \frac{r}{a}`,
- For :math:`r > a`, it becomes :math:`e(a) > \frac{r}{a} - 1`.

Both of these cases can be unified using the relation,

.. math::

    e(a) > \kappa(r,a), \quad \text{where} \quad \kappa(r,a) = \left|1 - \frac{r}{a} \right|.

In other words, only orbits with :math:`e(a)` above this threshold can physically reach :math:`r`. Following this, we introduce an auxiliary mass function :math:`\eta(r|a)`, such that :math:`\eta(r|a)\,\mathrm{d}r\,\mathrm{d}a` is the mass contributed at :math:`(r, \mathrm{d}r)` from particles with semi-major axis in the :math:`(a, a + \mathrm{d}a)` range. Naturally, this auxiliary mass function can be related back to :math:`\mathrm{d}m(a)` using the condition

.. math::

    \mathrm{d}a \int_{r_p(a)}^{r_a(a)} \eta(r|a)\,\mathrm{d}r = \mathrm{d}m(a) = 2\pi a \Sigma_a(a)\,\mathrm{d}a.

Time-averaging over particle orbits implies the contribution to the matter density at :math:`r` is proportional to :math:`\mathrm{d}t/\mathrm{d}r \propto 1/v_r`, where :math:`v_r` is the radial velocity. Using this, it can be shown that

.. math::

    \eta(r|a) = \frac{C(a) r}{\sqrt{(r - r_p)(r_a - r)}} = \frac{2 \Sigma_a(a) r}{\sqrt{(r - r_p)(r_a - r)}}.

The azimuthally averaged surface density, which is our quantity of interest, is then defined as

.. math::

    \bar\Sigma(r) = \frac{\eta(r)}{2\pi r}, \quad \text{with} \quad \eta(r) = \int \eta(r|a)\,\mathrm{d}a.

Putting this together gives

.. math::

    \bar\Sigma(r) = \frac{1}{2\pi r} \int_{0}^{\infty} \eta(r|a)\,\mathrm{d}a = \pi^{-1} \int_{0}^{\infty} \frac{\Sigma_a(a) \theta(r,a)}{\sqrt{(r - r_p)(r_a - r)}}\,\mathrm{d}a,

where :math:`\theta(r,a)` is a Heaviside step function ensuring the integrand vanishes for :math:`r` outside the orbital domain :math:`r_p(a) \leq r \leq r_a(a)`.

Due to this constraint, the lower limit of the integral must be :math:`a > r/2`, rather than zero. To further simplify the integrand, we define the dimensionless kernel function

.. math::

    \Phi_e(r,a) = \frac{\theta(r,a)}{\sqrt{e^2(a) - \kappa^2(r,a)}}.

This kernel function fully maps the :math:`a - r` relationship of particles. As a result, we arrive at the final expression used in this case:

.. math::

    \bar\Sigma(r) = \pi^{-1} \int_{r/2}^{\infty} \frac{\Sigma_a(a)}{a} \Phi_e(r,a)\,\mathrm{d}a.

This is the central result computed by the `DebrisPy` package in the case of unique eccentricities.

Distribution of Eccentricities per Semi-Major Axis
--------------------------------------------------

We now consider the general case where particles at fixed :math:`a` follow a distribution :math:`\psi_e(e,a)`. The main constraint we have on this function is that it must be normalised:

.. math::

    \int_0^1 \psi_e(e,a)\,\mathrm{d}e = 1.

In this case, the auxiliary function becomes :math:`\eta(r|a,e)`, since we now need to consider many eccentricities for a particle with a given semi-major axis. The mass conservation condition becomes

.. math::

    \mathrm{d}a\,\mathrm{d}e \int \eta(r|a,e)\,\mathrm{d}r = \psi_e(e,a) \,\mathrm{d}m(a)\,\mathrm{d}e.

Following similar steps as before, one can show that

.. math::

    \bar\Sigma(r) = \pi^{-1} \int_0^{\infty} \mathrm{d}a \int_0^1 \mathrm{d}e\, \frac{\Sigma_a(a)\,\psi_e(e,a)\,\theta(r,e,a)}{\sqrt{(r - r_p)(r_a - r)}}.

Conveniently, this can be written in the same form as in the unique eccentricity case, where the kernel is now defined by

.. math::

    \Phi_e(r,a) = \int_{\kappa}^{1} \frac{\psi_e(e,a)}{\sqrt{e^2 - \kappa^2(r,a)}}\,\mathrm{d}e.

Substituting this kernel into the expression for :math:`\bar\Sigma(r)` again gives

.. math::

    \bar\Sigma(r) = \pi^{-1} \int_{r/2}^{\infty} \frac{\Sigma_a(a)}{a} \Phi_e(r,a)\,\mathrm{d}a.

This is the central result computed by the `DebrisPy` package in the case of a distribution of eccentricities.
