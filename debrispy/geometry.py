# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
# ------------------------------------------------------------------------------------------------ #

# These are helper functions that the user may choose to use for easy calculations.
# They are not called by the remainder of the code.

def r(a, e, phi):
    """Compute radius r given semi-major axis a, eccentricity e, and true anomaly phi."""
    return a * (1 - e**2) / (1 + e * np.cos(phi))

def periapsis(a, e):
    """Compute periapsis distance, r_p."""
    return a * (1 - e)

def apoapsis(a, e):
    """Compute apoapsis distance, r_a."""
    return a * (1 + e)

def kappa(r, a):
    """Compute kappa."""
    return np.abs(1 - r/a)