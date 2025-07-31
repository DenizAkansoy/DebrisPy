from debrispy import Kernel, UniqueEccentricity, RayleighEccentricity, EccentricityDistribution
import pytest
import numpy as np

def test_kernel_initialisation():
    """
    Check that Kernel is initialised with correct properties.
    """
    ecc = UniqueEccentricity(a_min=10, a_max=100, e0=0.1, power=0.0)
    kernel = Kernel(r_min = 10, r_max = 100, eccentricity_profile = ecc, num_a_points = 100, num_r_points = 100)
    assert kernel.a_min == 10
    assert kernel.a_max == 100
    assert kernel.r_min == 10
    assert kernel.r_max == 100
    assert kernel.num_a_points == 100
    assert kernel.num_r_points == 100

def test_kernel_compute():
    """
    Check that the kernel is computed correctly for a unique eccentricity profile
    """
    ecc = UniqueEccentricity(a_min=10, a_max=100, e0=0.1, power=0.0)
    kernel = Kernel(r_min = 10, r_max = 100, eccentricity_profile = ecc, num_a_points = 100, num_r_points = 100)
    kernel.compute()
    values = kernel.get_values(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert values.shape == (3, 3)

def test_kernel_compute_rayleigh():
    """
    Check that the kernel is computed correctly for a Rayleigh eccentricity profile
    """
    ecc = RayleighEccentricity(a_min=10, a_max=100, sigma0=0.1, power=0.0)
    kernel = Kernel(r_min = 10, r_max = 100, eccentricity_profile = ecc, num_a_points = 100, num_r_points = 100)
    kernel.compute()
    values = kernel.get_values(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert values.shape == (3, 3)

def test_kernel_compute_rayleigh_approx():
    """
    Check that the kernel is computed correctly for a Rayleigh eccentricity profile
    """
    ecc = RayleighEccentricity(a_min=10, a_max=100, sigma0=0.1, power=0.0)
    kernel = Kernel(r_min = 10, r_max = 100, eccentricity_profile = ecc, num_a_points = 100, num_r_points = 100)
    kernel.compute(rayleigh_approx=True)
    values = kernel.get_values(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert values.shape == (3, 3)

def test_kernel_small_eccentricity():
    """
    Kernel should still return finite values with small eccentricity.
    """
    ecc = UniqueEccentricity(a_min=10, a_max=100, e0=1e-5, power=0.0)
    kernel = Kernel(r_min = 10, r_max = 100, eccentricity_profile = ecc, num_a_points = 100, num_r_points = 100)
    kernel.compute()
    values = kernel.get_values(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)

def test_kernel_invalid_range():
    """
    If using general eccentricity distribution, values outside the grid return 0.
    """
    def psi_raw(e, a):
        return 10 * e**2

    ecc = EccentricityDistribution(
        a_min=10, a_max=50,
        distribution_func=psi_raw,
        auto_normalise=True,
        num_e_points=500,
        num_a_points=500,
        grid_type='uniform',
        interpolation_method='linear'
    )
    kernel = Kernel(r_min=10, r_max=50, eccentricity_profile=ecc,
                    num_a_points=100, num_r_points=100)
    kernel.compute()

    values = kernel.get_values(a_vals=[5], r_vals=[5])
    assert np.allclose(values, 0.0)

def test_kernel_output_shape():
    """
    Ensure that get_values returns the correct shape (len(a_vals), len(r_vals)).
    """
    def psi_raw(e, a): return e * a + 1
    ecc = EccentricityDistribution(a_min=10, a_max=50, distribution_func=psi_raw, auto_normalise=True, grid_type='uniform', interpolation_method='linear')
    kernel = Kernel(r_min=10, r_max=100, eccentricity_profile=ecc, num_a_points=100, num_r_points=100)
    kernel.compute(n_jobs = 1)

    a_vals = np.linspace(10, 50, 5)
    r_vals = np.linspace(10, 100, 4)
    values = kernel.get_values(a_vals, r_vals)

    assert values.shape == (5, 4)

def test_adaptive_gridding():
    """
    Test that adaptive gridding works (and returns a non-zero number of adaptive points).
    """
    def psi_raw(e, a):
        return 10 * e**2

    ecc = EccentricityDistribution(
        a_min=10, a_max=50,
        distribution_func=psi_raw,
        auto_normalise=True,
        num_e_points=500,
        num_a_points=500,
        grid_type='uniform',
        interpolation_method='linear'
    )

    kernel = Kernel(r_min=10, r_max=50, eccentricity_profile=ecc,
                    num_a_points=100, num_r_points=100)
    kernel.compute(adaptive_grid=True, tol = 1e-2)

    num_adaptive_points = kernel.num_adaptive_points()
    assert num_adaptive_points > 0

def test_phi_grid_before_compute_raises():
    """
    Accessing phi_grid before compute() should raise RuntimeError.
    """
    ecc = UniqueEccentricity(a_min=10, a_max=100, e0=0.1, power=0.0)
    kernel = Kernel(r_min=10, r_max=100, eccentricity_profile=ecc, num_a_points=100, num_r_points=100)

    with pytest.raises(RuntimeError, match="Phi_grid is not computed"):
        _ = kernel.phi_grid()

def test_phi_samples_before_compute_raises():
    """
    Accessing phi_samples before adaptive compute() should raise RuntimeError.
    """
    def psi_raw(e, a): return e**2

    ecc = EccentricityDistribution(
        a_min=10, a_max=50,
        distribution_func=psi_raw,
        auto_normalise=True,
        num_e_points=100,
        num_a_points=100,
        grid_type='uniform',
        interpolation_method='linear'
    )

    kernel = Kernel(r_min=10, r_max=50, eccentricity_profile=ecc, num_a_points=100, num_r_points=100)
    kernel.compute(adaptive_grid=False) 

    with pytest.raises(RuntimeError, match="Phi_samples is not computed"):
        _ = kernel.phi_samples()