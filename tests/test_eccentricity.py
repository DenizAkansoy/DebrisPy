import numpy as np
from debrispy import (UniqueEccentricity, 
                      RayleighEccentricity, 
                      TopHatEccentricity, 
                      PowerLawEccentricity, 
                      TriangularEccentricity, 
                      TruncGaussEccentricity, 
                      EccentricityDistribution)
from scipy.integrate import quad

# Unique eccentricity tests

def test_unique_eccentricity():
    """
    Check that the eccentricity object is initialized with the correct values
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, e0 = 0.4, power = 0.5)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    assert ecc.e0 == 0.4
    assert ecc.power == 0.5

def test_unique_eccentricity_nonzero():
    """
    Check that the eccentricity is non-zero for a non-zero eccentricity profile
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, e0 = 0.4, power = 0.5)
    a = np.array([10, 20, 30])
    assert np.all(ecc(a) > 0)

def test_unique_eccentricity_callable():
    """
    Check that the eccentricity is constant for a constant eccentricity profile
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, e0 = 0.4, power = 0)
    a = np.array([10, 20, 30])
    assert np.all(ecc(a) == 0.4)

def test_unique_eccentricity_values():
    """
    Check that the returned eccentricity values are the same for the callable and the eccentricity function
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, e0 = 0.4, power = 1)
    a = np.array([10, 20, 30])
    vals1 = ecc(a)
    vals2 = ecc.eccentricity(a)
    assert np.all(vals1 == vals2)

def test_unique_eccentricity_derivative():
    """
    Check that the derivative of the eccentricity is 0 for a constant eccentricity
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, e0 = 0.4, power = 0)
    a = np.array([10, 20, 30])
    assert np.all(ecc.derivative(a) == 0)

def test_unique_eccentricity_custom_profile():
    """
    Check that the eccentricity is non-zero for a custom eccentricity profile
    """
    ecc = UniqueEccentricity(a_min = 10, a_max = 100, eccentricity_func = lambda a: 0.4*a**-0.5)
    a = np.array([10, 20, 30])
    assert np.all(ecc(a) > 0)
    
# Built-in eccentricity distributions

def test_rayleigh_eccentricity():
    """
    Check that the Rayleigh eccentricity distribution is initialized with the correct values
    """
    ecc = RayleighEccentricity(a_min = 10, a_max = 100, sigma0 = 0.4, power = 0.5)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    assert ecc.sigma0 == 0.4
    assert ecc.power == 0.5

def test_rayleigh_eccentricity_shape():
    """
    Check that the output eccentricity distributions have the correct shape
    """
    ecc = RayleighEccentricity(a_min = 10, a_max = 100, sigma0 = 0.4, power = 0.5)
    a_vals = np.linspace(10, 100, 1000)
    e_vals = np.linspace(0, 1, 1000)
    assert ecc.distribution(e_vals, a_vals).shape == (1000, 1000)
    assert ecc.distribution([0.5], a_vals).shape == (1000,)
    assert ecc.distribution(e_vals, [10]).shape == (1000,)

def test_rayleigh_eccentricity_bounds():
    """
    Check that the eccentricity distribution becomes negligible outside expected support
    """
    ecc = RayleighEccentricity(a_min=10, a_max=100, sigma0=0.1, power=0.0)
    val_high = ecc.distribution(2, 10)
    val_low = ecc.distribution(-1, 10)
    assert val_high < 1e-20
    assert val_low < 1e-20  

def test_tophat_eccentricity():
    """
    Check that the TopHatEccentricity eccentricity distribution is initialized with the correct values
    """
    ecc = TopHatEccentricity(a_min = 10, a_max = 100, lam = 0.4)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    a_vals = np.linspace(10, 100, 1000)
    assert np.all(ecc.lambda_func(a_vals) == 0.4)

def test_power_law_eccentricity():
    """
    Check that the PowerLawEccentricity eccentricity distribution is initialized with the correct values
    """
    ecc = PowerLawEccentricity(a_min = 10, a_max = 100, lam = 0.4, zeta = 0.5)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    assert ecc.zeta == 0.5
    a_vals = np.linspace(10, 100, 1000)
    assert np.all(ecc.lambda_func(a_vals) == 0.4)

def test_triangular_eccentricity():
    """
    Check that the TriangularEccentricity eccentricity distribution is initialized with the correct values
    """
    ecc = TriangularEccentricity(a_min = 10, a_max = 100, lam = 0.4)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    a_vals = np.linspace(10, 100, 1000)
    assert np.all(ecc.lambda_func(a_vals) == 0.4)

def test_trunc_gauss_eccentricity():
    """
    Check that the TruncGaussEccentricity eccentricity distribution is initialized with the correct values
    """
    ecc = TruncGaussEccentricity(a_min = 10, a_max = 100, lam = 0.4, sigma = 0.5)
    assert ecc.a_min == 10
    assert ecc.a_max == 100
    a_vals = np.linspace(10, 100, 1000)
    assert np.all(ecc.lambda_func(a_vals) == 0.4)
    assert np.all(ecc.sigma_func(a_vals) == 0.5)

# Check auto-normalisation for general eccentricity distributions

def test_eccentricity_distribution_autonorm_1():
    """
    Check that the auto-normalised eccentricity distribution integrates to ~1
    """
    # A simple unnormalised PDF
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

    # Integrate the normalised version at a few values of a
    for a_val in [10, 30, 40]:
        integral, _ = quad(lambda e: float(ecc.distribution(np.array([e]), a_val)[0]), 0, 1)
        assert np.isclose(integral, 1.0, atol=1e-3)

def test_eccentricity_distribution_autonorm_2():
    """
    Check that the auto-normalised eccentricity distribution integrates to ~1
    """
    # A simple unnormalised PDF
    def psi_raw(e, a):
        return 10 * a * np.sin(e) + 50

    ecc = EccentricityDistribution(
        a_min=10, a_max=50, 
        distribution_func=psi_raw,
        auto_normalise=True, 
        num_e_points=500,
        num_a_points=500,
        grid_type='uniform',
        interpolation_method='linear'
    )

    # Integrate the normalised version at a few values of a
    for a_val in [10, 30, 40]:
        integral, _ = quad(lambda e: float(ecc.distribution(np.array([e]), a_val)[0]), 0, 1)
        assert np.isclose(integral, 1.0, atol=1e-3)