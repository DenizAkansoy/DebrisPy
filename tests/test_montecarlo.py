import numpy as np
import pytest
from debrispy import (
    UniqueEccentricity, 
    SigmaA, 
    MonteCarlo, 
    RayleighEccentricity
)


@pytest.fixture
def setup_montecarlo():
    """
    Setup a MonteCarlo object with a UniqueEccentricity profile.
    """
    sigma = SigmaA(10, 50, Sigma_0=1.0, power=1.0)
    ecc = UniqueEccentricity(10, 50, e0=0.1, power=0.0)
    mc = MonteCarlo(sigma, ecc, n_samples=10000)
    return mc

def test_sample_a_outputs_correct_size(setup_montecarlo):
    """
    Test that the sample_a method outputs an array of the correct size.
    """
    mc = setup_montecarlo
    a_vals = mc.sample_a(use_jacobian=True)
    assert isinstance(a_vals, np.ndarray)
    assert len(a_vals) == mc.n_samples
    assert np.all((a_vals >= mc.sigma_a.a_min) & (a_vals <= mc.sigma_a.a_max))

def test_sampler_output_shapes(setup_montecarlo):
    """
    Test that the sampler method outputs arrays of the correct shape.
    """
    mc = setup_montecarlo
    a_vals, r_vals, e_vals, f_vals = mc.sampler(use_jacobian=True, verbose=False)
    assert a_vals.shape == r_vals.shape == e_vals.shape == f_vals.shape
    assert len(a_vals) == mc.n_samples

def test_get_1d_histogram_runs(setup_montecarlo):
    """
    Test that the get_1d_histogram method runs without errors.
    """
    mc = setup_montecarlo
    mc.sampler(use_jacobian=True, verbose=False)
    a_centers, hist_a, r_centers, hist_r = mc.get_1d_histogram(bins=100, scale=True, verbose=False)
    assert len(a_centers) == len(hist_a)
    assert len(r_centers) == len(hist_r)

def test_get_cartesian_histogram(setup_montecarlo):
    """
    Test that the get_cartesian_histogram method runs without errors.
    """
    mc = setup_montecarlo
    mc.sampler(verbose=False)
    hist, x_edges, y_edges = mc.get_cart_histogram(bins=100, verbose=False)
    assert hist.shape == (100, 100)
    assert len(x_edges) == 101
    assert len(y_edges) == 101

def test_get_polar_histogram(setup_montecarlo):
    """
    Test that the get_polar_histogram method runs without errors.
    """
    mc = setup_montecarlo
    mc.sampler(verbose=False)
    hist, r_edges, phi_edges = mc.get_polar_histogram(bins=100, verbose=False)
    assert hist.shape == (100, 100)
    assert len(r_edges) == 101
    assert len(phi_edges) == 101

def test_sample_eccentricities_rayleigh():
    """
    Test that the sample_eccentricities method runs without errors.
    """
    ecc = RayleighEccentricity(a_min=10, a_max=50, sigma0=0.4, power=0.5)
    sigma = SigmaA(a_min=10, a_max=50, Sigma_0=1.0, power=0.5)
    mc = MonteCarlo(sigma, ecc, n_samples=10_000)

    a_samples = np.linspace(10, 50, 2000)
    e_samples = mc.sample_eccentricities(a_samples)

    assert isinstance(e_samples, np.ndarray)
    assert e_samples.shape == (2000,)
    assert np.all((e_samples >= 0) & (e_samples <= 1))


def test_invalid_ecc_profile_type():
    """
    Test that the sample_eccentricities method raises an error if the eccentricity profile is not a valid type.
    """
    class BadEcc:
        pass

    sigma = SigmaA(a_min=10, a_max=50, sigma_0=1.0, power=1.0)
    mc = MonteCarlo(sigma, BadEcc(), n_samples=100)

    with pytest.raises(AttributeError):
        mc.sample_eccentricities(np.linspace(10, 50, 100))

def test_no_negative_radii():
    """
    Test that the sampler method raises an error if the radial position is negative.
    """
    sigma = SigmaA(a_min=10, a_max=50, Sigma_0=1.0, power=1.0)
    ecc = UniqueEccentricity(a_min=10, a_max=50, e0=0.1, power=1.0)
    mc = MonteCarlo(sigma, ecc, n_samples=10_000)
    _, r, _, _ = mc.sampler(verbose=False)
    assert np.all(r > 0)

def test_plot_1d_raises_without_asd():
    """
    Test that the plot_1d method raises an error if the asd is not provided.
    """
    sigma = SigmaA(a_min=10, a_max=50, Sigma_0=1.0, power=1.0)
    ecc = UniqueEccentricity(a_min=10, a_max=50, e0=0.1, power=1.0)
    mc = MonteCarlo(sigma, ecc, n_samples=1000)
    mc.sampler(verbose=False)

    with pytest.raises(ValueError, match="`asd` must be provided if overlay=True."):
        mc.plot_1d(overlay=True)

def test_reproducibility():
    """
    Test that the MonteCarlo object is reproducible.
    """
    sigma = SigmaA(10, 50, Sigma_0=1.0, power=1.0)
    ecc = UniqueEccentricity(10, 50, e0=0.1, power=0.0)

    np.random.seed(22)
    mc1 = MonteCarlo(sigma, ecc, n_samples=1000)
    a1 = mc1.sample_a()

    np.random.seed(22)
    mc2 = MonteCarlo(sigma, ecc, n_samples=1000)
    a2 = mc2.sample_a()

    np.testing.assert_allclose(a1, a2)