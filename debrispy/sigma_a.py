# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.integrate import quad
from typing import Callable, Optional, Tuple, Union
# ------------------------------------------------------------------------------------------------ #

class SigmaA:
    """
    Represents a surface density profile, :math:`\Sigma(a)`, as a function of semi-major axis.

    Attributes
    ----------
    a_min (float): Minimum value of the semi-major axis range.
    a_max (float): Maximum value of the semi-major axis range.
    sigma0 (float): Base amplitude (normalization) of the profile. Defaults to 1.0.
    profile_type (str): Type of profile ('power_law', 'gaussian', 'step_up', 'step_down', 'custom').
    """

    VALID_PROFILES = ['power_law', 'gaussian', 'step_up', 'step_down', 'custom']

    def __init__(
        self,
        a_min: float,
        a_max: float,
        sigma0: float = 1.0,
        profile_type: str = 'power_law',
        **kwargs,
    ) -> None:
        """
        Initialize a profile object.

        Attributes
        ----------
        a_min (float): Minimum value of the domain range.
        a_max (float): Maximum value of the domain range.
        sigma0 (float, optional): Base amplitude of the profile. Defaults to 1.0.
        profile_type (str, optional): Type of profile. Defaults to 'power_law'.
        **kwargs: Additional parameters depending on profile_type:
            - 'power_law':
                - power (float): The exponent of the power law.
            - 'gaussian':
                - gauss_center (float): The center of the Gaussian.
                - gauss_width (float): The standard deviation (width) of the Gaussian.
            - 'step_up'/'step_down':
                - step (float): The position of the step.
            - 'custom':
                - sigma_func (Callable[[Union[float, npt.NDArray[np.float64]]], npt.NDArray[np.float64]]):
                    A function that takes a single float or array as input (the semi-major axis 'a')
                    and returns the *unnormalized* surface density value. This output will then
                    be multiplied by the `sigma0` attribute of the class.

        Raises
        ------
        TypeError: If a_min, a_max, or sigma0 are not numeric.
        ValueError: If a_min >= a_max or if profile_type is invalid or required parameters are missing.
        """
        # Validate basic inputs
        if not isinstance(a_min, (int, float)) or not isinstance(a_max, (int, float)):
            raise TypeError("a_min and a_max must be numeric values")
        if a_min >= a_max:
            raise ValueError("a_min must be less than a_max")
        if not isinstance(sigma0, (int, float)):
            raise TypeError("sigma0 must be a numeric value")

        # Store basic attributes
        self.a_min = a_min
        self.a_max = a_max
        self.sigma0 = sigma0

        # Validate profile type
        if profile_type not in self.VALID_PROFILES:
            raise ValueError(f"Unknown profile_type. Choose from {self.VALID_PROFILES}")
        self.profile_type = profile_type

        # Set up sigma function based on profile type
        self._setup_sigma_func(profile_type, **kwargs)

        # Cache last a_vals and sigma_a values for plotting
        self._last_a_vals: Optional[npt.NDArray[np.float64]] = None
        self._last_sigma_a: Optional[npt.NDArray[np.float64]] = None

    def _setup_sigma_func(self, profile_type: str, **kwargs) -> None:
        """
        Configure the sigma function based on profile type and parameters.
        
        Parameters
        ----------
        profile_type (str): The type of profile to use.
        **kwargs: Additional keyword arguments.
        
        Raises
        ------
        ValueError: If the profile type is invalid or required parameters are missing.
        """

        if profile_type == 'custom':
            if 'sigma_func' not in kwargs or not callable(kwargs['sigma_func']):
                raise ValueError("Custom profile requires 'sigma_func' parameter to be callable")
            self.sigma_func: Callable[[Union[float, npt.NDArray[np.float64]]], npt.NDArray[np.float64]] = kwargs['sigma_func']

        elif profile_type == 'power_law':
            if 'power' not in kwargs:
                raise ValueError("Power profile requires 'power' parameter")
            self.power: float = kwargs['power']
            self.sigma_func = lambda a: self.sigma0 * (self.a_min / a)**self.power

        elif profile_type == 'gaussian':
            if 'gauss_center' not in kwargs or 'gauss_width' not in kwargs:
                raise ValueError("Gaussian profile requires 'gauss_center' and 'gauss_width' parameters")

            self.gauss_center: float = kwargs['gauss_center']  
            self.gauss_width: float = kwargs['gauss_width']    

            if self.gauss_width <= 0:
                raise ValueError("Gaussian width must be positive")

            norm = 1 / (np.sqrt(2 * np.pi) * self.gauss_width)
            self.sigma_func = lambda a: self.sigma0 * norm * np.exp(-(a - self.gauss_center)**2 / (2 * self.gauss_width**2))

        elif profile_type == 'step_up':
            if 'step' not in kwargs:
                raise ValueError("Step profile requires 'step' parameter")
            self.step: float = kwargs['step']  
            self.sigma_func = lambda a: self.sigma0 * (a >= self.step)

        elif profile_type == 'step_down':
            if 'step' not in kwargs:
                raise ValueError("Step profile requires 'step' parameter")
            self.step: float = kwargs['step']  
            self.sigma_func = lambda a: self.sigma0 * (a < self.step)

    def get_values(self, a_vals: Union[float, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Calculate surface density at the given semi-major axis values.

        Parameters
        ----------
        a_vals (float or array-like): Semi-major axis value(s) to evaluate.

        Returns
        -------
        ndarray: Surface density values at the specified semi-major axis values.
        """
        a_vals = np.atleast_1d(a_vals)
        sigma = np.where(
            (a_vals >= self.a_min) & (a_vals <= self.a_max),
            self.sigma_func(a_vals),
            0.0
        )

        # Cache values for potential later use in plotting
        self._last_a_vals = a_vals
        self._last_sigma_a = sigma
        return sigma

    def __str__(self) -> str:
        """
        Return a string representation of the surface density profile.

        Returns
        -------
        str: String representation of the surface density profile.
        """
        info = f"SigmaA(type={self.profile_type}, a_min={self.a_min}, a_max={self.a_max}, sigma0={self.sigma0}"
        if self.profile_type == 'power_law':
            if hasattr(self, 'power'):
                info += f", power={self.power}"
        elif self.profile_type == 'gaussian':
            if hasattr(self, 'gauss_center') and hasattr(self, 'gauss_width'):
                info += f", gauss_center={self.gauss_center}, gauss_width={self.gauss_width}"
        elif self.profile_type in ['step_up', 'step_down']:
            if hasattr(self, 'step'):
                info += f", step={self.step}"
        return info + ")"

    def __call__(self, a_vals: Union[float, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Evaluate the surface density at specified semi-major axis values.
        This method provides a convenient function-like interface for the class.

        Parameters
        ----------
        a_vals (float or array-like): Semi-major axis value(s) to evaluate.

        Returns
        -------
        ndarray: Surface density values at the specified semi-major axis values.
        """
        return self.get_values(a_vals)

    def plot(
        self,
        a_vals: Optional[npt.NDArray[np.float64]] = None,
        num_points: int = 500,
        save: bool = False,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the surface density profile with flexible matplotlib customization.

        Parameters
        ----------
        a_vals (array-like, optional): Specific semi-major axis values to plot.
            If None, use cached values or generate new ones. Defaults to None.
        num_points (int, optional): Number of points to use if generating new values.
            Defaults to 500.
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        filename (str, optional): Filename to save the figure. If None, a default name
            will be generated. Defaults to None.
        figsize (tuple, optional): Figure size (width, height).
            Defaults to (8, 6).
        show (bool, optional): Whether to display the plot immediately.
            Defaults to True.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None,
            a new figure and axes will be created. Defaults to None.
        **plot_kwargs: Additional keyword arguments passed to plt.plot().
            Examples include:
                - color: Color of the line
                - linestyle: Style of the line ('-', '--', '-.', ':')
                - linewidth or lw: Width of the line
                - marker: Point marker style ('o', 's', '^', etc.)
                - alpha: Transparency of the line
                - label: Label for the legend.

        Returns
        -------
        tuple: Figure and axes objects for further customization if needed.
        """
        # Calculate or retrieve sigma values
        if a_vals is not None:
            sigma_vals = self.get_values(a_vals)
        else:
            if self._last_a_vals is not None and self._last_sigma_a is not None:
                a_vals = self._last_a_vals
                sigma_vals = self._last_sigma_a
            else:
                a_vals = np.linspace(self.a_min, self.a_max, num_points)
                sigma_vals = self.get_values(a_vals)

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Set default plot parameters that can be overridden
        default_kwargs = {
            'linewidth': 2,
        }

        # Update with user-provided kwargs
        default_kwargs.update(plot_kwargs)

        # Create the plot
        line = ax.plot(a_vals, sigma_vals, **default_kwargs)

        # Set default labels and grid (unless overridden later by user)
        ax.set_xlabel('Semi-Major Axis, $a$', fontsize=14)
        ax.set_ylabel('Surface Density, $\\Sigma(a)$', fontsize=14)
        ax.grid(True)

        plt.tight_layout()

        # Save if requested
        if save:
            if filename is None:
                filename = f"sigma_a_{self.profile_type}.png"
            plt.savefig(filename)

        # Show if requested
        if show:
            plt.show()

    def compute_area(self) -> float:
        """
        Compute the integral of :math:`\Sigma(a)` over [a_min, a_max].

        Returns
        -------
        float: Total area under the surface density curve.
        """
        result, _ = quad(self.sigma_func, self.a_min, self.a_max)
        return result