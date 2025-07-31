from debrispy import SigmaA, UniqueEccentricity, RayleighEccentricity, Kernel, ASD
import numpy as np
import pytest

def test_asd_init():
    """
    Test that the ASD object is initialised without errors.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    assert asd.sigma_a.a_min == 1
    assert asd.sigma_a.a_max == 4
    assert asd.sigma_a.sigma0 == 1
    assert asd.sigma_a.power == 0.5
    assert asd.kernel == kernel
    assert asd.sigma_a == sigma_a

def test_asd_integrand():
    """
    Test that the ASD integrand has the correct shape.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    a_vals = np.linspace(1, 4, 1000)
    r_vals = np.linspace(1, 4, 1000)
    integrand = asd.integrand(a_vals, r_vals)
    assert integrand.shape == (1000, 1000)

def test_asd_compute_quadvec():
    """
    Test that the ASD compute_quadvec method works.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    r_vals = np.linspace(1, 4, 100)
    asd.compute_quadvec(r_vals)
    assert asd._sigma_r_vals is not None
    assert asd._r_vals is not None
    assert asd._sigma_r_vals.shape == (100,)
    assert asd._r_vals.shape == (100,)

def test_asd_compute_gl():
    """
    Test that the ASD compute_gl method works (with and without adaptive limits)
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    r_vals = np.linspace(1, 4, 100)

    asd.compute_gl(r_vals, n_points = 32, adaptive_limits = False)
    assert asd._sigma_r_vals is not None
    assert asd._r_vals is not None
    assert asd._sigma_r_vals.shape == (100,)
    assert asd._r_vals.shape == (100,)

    asd.compute_gl(r_vals, n_points = 32, adaptive_limits = True)
    assert asd._sigma_r_vals is not None
    assert asd._r_vals is not None
    assert asd._sigma_r_vals.shape == (100,)
    assert asd._r_vals.shape == (100,)

def test_asd_refine():
    """
    Test that the ASD refine method works.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)
    
    r_vals = np.linspace(1, 4, 100)
    asd.compute_gl(r_vals, n_points = 32, adaptive_limits = False)
    asd.refine(curvature_factor = 1.0, max_rounds = 2, subdiv = 2, n_jobs = -1, show_progress = False, n_points = 32, tol_rel = 1e-8, tol_abs = 1e-8, max_level = 25, adaptive_limits = False, rf = 10.0, pad = 0.05, batch_size = 10)

    r_vals, sigma_r_vals = asd.get_values()
    assert asd._sigma_r_vals is not None
    assert asd._r_vals is not None
    assert asd._sigma_r_vals.shape >= (100,)
    assert asd._r_vals.shape >= (100,)
    assert np.allclose(r_vals, asd._r_vals)
    assert np.allclose(sigma_r_vals, asd._sigma_r_vals)

def test_asd_convolve():
    """
    Test that the ASD convolve method works.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = unique_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    r_vals = np.linspace(1, 4, 100)
    asd.compute_gl(r_vals, n_points = 32, adaptive_limits = False)
    sigma_r_conv = asd.convolve(width = 0.1, M = 2048)

    assert sigma_r_conv is not None
    assert sigma_r_conv.shape == (100,)

def test_asd_adaptive_limits():
    """
    Test that the ASD adaptive limits do not work with an eccentric distribution that is not unique.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0 = 1, power = 0.5)
    ray_ecc = RayleighEccentricity(a_min=1, a_max=4, sigma0 = 0.4, power = 0.5)
    kernel = Kernel(r_min = 1, r_max = 4, eccentricity_profile = ray_ecc, num_a_points = 1000, num_r_points = 1000)
    kernel.compute()

    asd = ASD(kernel, sigma_a)

    r_vals = np.linspace(1, 4, 100)
    try:
        asd.compute_gl(r_vals, n_points = 32, adaptive_limits = True)
    except ValueError:
        assert True
    else:
        assert False

def test_asd_get_values_fails_if_not_computed():
    """
    Test that the ASD get_values method fails if the ASD has not been computed.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0=1, power=0.5)
    ecc = UniqueEccentricity(a_min=1, a_max=4, e0=0.1, power=0.0)
    kernel = Kernel(r_min=1, r_max=4, eccentricity_profile=ecc, num_a_points=100, num_r_points=100)
    kernel.compute()
    asd = ASD(kernel, sigma_a)

    with pytest.raises(RuntimeError, match="must calculate the ASD"):
        asd.get_values()

def test_asd_convolve_fails_without_sigma_r_vals():
    """
    Test that the ASD convolve method fails if the ASD has not been computed.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0=1, power=0.5)
    ecc = UniqueEccentricity(a_min=1, a_max=4, e0=0.1, power=0.0)
    kernel = Kernel(r_min=1, r_max=4, eccentricity_profile=ecc, num_a_points=100, num_r_points=100)
    kernel.compute()
    asd = ASD(kernel, sigma_a)

    with pytest.raises(RuntimeError, match="must calculate the ASD"):
        asd.convolve(width=0.1)

def test_asd_get_values_after_compute():
    """
    Test that the ASD get_values method works after the ASD has been computed.
    """
    sigma_a = SigmaA(a_min=1, a_max=4, sigma0=1, power=0.5)
    unique_ecc = UniqueEccentricity(a_min=1, a_max=4, e0=0.4, power=0.5)
    kernel = Kernel(r_min=1, r_max=4, eccentricity_profile=unique_ecc, num_a_points=100, num_r_points=100)
    kernel.compute()

    asd = ASD(kernel, sigma_a)
    r_vals = np.linspace(1, 4, 100)
    asd.compute_gl(r_vals, n_points=32, adaptive_limits=False)

    r_out, sigma_out = asd.get_values()
    assert isinstance(r_out, np.ndarray)
    assert isinstance(sigma_out, np.ndarray)
    assert r_out.shape == sigma_out.shape == (100,)
    assert np.allclose(r_out, r_vals)

    