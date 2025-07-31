from debrispy import SigmaA
import numpy as np
import pytest

def test_sigma_a_init_values():
    """
    Test that the SigmaA object is initialized with the correct values
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    assert s.a_min == 10 and s.a_max == 100
    assert s.profile_type == 'power_law'
    assert s.sigma0 == 1
    assert s.power == 0.5

def test_sigma_a_callable():
    """
    Test that the SigmaA object is callable
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    a = np.array([20, 40, 60])
    assert np.all(s(a) >= 0)

def test_sigma_a_boundaries():
    """
    Test that the SigmaA object returns 0 for values outside the boundaries
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    assert s(5) == 0
    assert s(150) == 0

def test_sigma_a_get_values():
    """
    Test that the get_values method returns the same values as the callable
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    a = np.array([20, 40, 60])
    vals1 = s.get_values(a)
    vals2 = s(a)
    assert np.all(vals1 == vals2)

def test_sigma_a_string_repr():
    """
    Test that the string representation of the SigmaA object is a string
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    assert isinstance(str(s), str)


def test_sigma_a_output():
    """
    Test that the output of the SigmaA object is a float or a scalar (each element of the output vector)
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    a = np.array([15, 20, 25])
    vals = s(a)
    assert isinstance(vals[0], float) or np.isscalar(vals[0])

def test_sigma_a_vector_output():
    """
    Test that the output of the SigmaA object is a vector when the input is a vector
    """
    s = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    a = np.array([15, 20, 25])
    vals = s(a)
    assert vals.shape == a.shape


def test_sigma_a_profiles():
    """
    Test that the SigmaA object returns different values for different profiles
    """
    s1 = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=1, power = 0.5)
    s2 = SigmaA(a_min = 10, a_max = 100, profile_type = 'power_law', sigma0=0.3, power = 1)
    a = np.array([15, 20, 25])
    vals1 = s1(a)
    vals2 = s2(a)
    assert np.all(vals1 != vals2)


def test_sigma_a_valid_profiles():
    """
    Test that the SigmaA object returns non-negative values for all valid profiles
    """
    for profile in SigmaA.VALID_PROFILES:
        kwargs = {}
        if profile == 'power_law':
            kwargs = {'power': 1.0}
        elif profile == 'gaussian':
            kwargs = {'gauss_center': 50, 'gauss_width': 10}
        elif profile in ['step_up', 'step_down']:
            kwargs = {'step': 60}
        elif profile == 'custom':
            kwargs = {'sigma_func': lambda a: np.ones_like(a)}

        s = SigmaA(a_min=10, a_max=100, profile_type=profile, sigma0=1, **kwargs)
        a = np.array([20, 30, 40])
        vals = s(a)
        assert np.all(vals >= 0)

def test_sigma_a_invalid_profile():
    """
    Test that the SigmaA object raises an error for an invalid profile
    """
    with pytest.raises(ValueError, match="Unknown profile_type"):
        SigmaA(a_min=10, a_max=100, profile_type="not_a_profile", sigma0=1)