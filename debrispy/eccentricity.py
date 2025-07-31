# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

from typing import Callable, Optional, Tuple, Union
import numpy.typing as npt
from functools import partial
# ------------------------------------------------------------------------------------------------ #

class EccentricityProfile():
    """
    Abstract base class for eccentricity profiles.
    """

    def eccentricity(self, a):
        """
        Return the eccentricity e(a) at given semi-major axis a (for unique eccentricity profiles).
        Should be overridden by subclasses if applicable.
        """
        raise NotImplementedError("This profile does not define a deterministic e(a).")

    def distribution(self, e, a):
        """
        Return the eccentricity distribution ψ_e(e,a) (for distribution-based eccentricity profiles).
        Should be overridden by subclasses if applicable.
        """
        raise NotImplementedError("This profile does not define a distribution ψ_e(e,a).")

    def plot(self, *args, **kwargs):
        """
        Plot the eccentricity profile.
        """
        raise NotImplementedError("Subclasses must implement a plot method.")

class UniqueEccentricity(EccentricityProfile):
    """
    Fixed eccentricity profile where eccentricity is a unique function of semi-major axis: e = e(a).
    
    By default implements a power-law profile e(a) = e0 * (a_min/a)^{power}, but accepts custom user-defined functions.
    
    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    eccentricity_func : callable, optional
        Custom function e(a). If None, uses a default power-law form.
    e0 : float, optional
        Normalization constant. Default is 1.0.
    power : float, optional
        Power-law exponent for built-in power-law e(a) (required if eccentricity_func is None).
    
    Raises
    ------
    ValueError
        If the eccentricity function produces values outside the range [0, 1) 
        across the specified semi-major axis range.
    """

    def __init__(
            self, 
            a_min: float, 
            a_max: float, 
            eccentricity_func: Optional[Callable[[Union[float, npt.NDArray[np.float64]]], npt.NDArray[np.float64]]] = None,
            e0: float = 1.0, 
            power: Optional[float] = None
        ) -> None:
        if a_min <= 0 or a_max <= 0:
            raise ValueError("Semi-major axes must be positive.")
        if a_min >= a_max:
            raise ValueError("a_min must be less than a_max.")
            
        self.a_min: float = a_min
        self.a_max: float = a_max
        self.e0: float = e0
        
        if power is not None:
            self.power: Optional[float] = power
        else:
            self.power: Optional[float] = None

        if eccentricity_func is None:
            if power is None:
                raise ValueError("power argument must be provided for the default power-law profile.")
            self.eccentricity_func: Callable[[Union[float, npt.NDArray[np.float64]]], npt.NDArray[np.float64]] = lambda a: self.e0 * (self.a_min / np.asarray(a, dtype=float))**self.power
            self.default_profile: bool = True
        else:
            self.eccentricity_func: Callable[[Union[float, npt.NDArray[np.float64]]], npt.NDArray[np.float64]] = lambda a: self.e0 * eccentricity_func(a)
            self.default_profile: bool = False

        # Verify eccentricity range validity
        self._check_range()
    
    def _check_range(self):
        """
        Verify that 0 ≤ e(a) < 1 over the full [a_min, a_max] range.
        
        Raises
        ------
        ValueError
            If eccentricity values fall outside the valid range [0, 1).
        """
        a_vals: npt.NDArray[np.float64] = np.linspace(self.a_min, self.a_max, 1000)  # Use geomspace for better sampling
        e_vals: npt.NDArray[np.float64] = self.eccentricity_func(a_vals)
        
        if np.any(e_vals < 0) or np.any(e_vals >= 1):
            raise ValueError("Eccentricity values must satisfy 0 ≤ e(a) < 1 over the full semi-major axis range.")

    def eccentricity(
            self, 
            a: Union[float, npt.NDArray[np.float64]]
        ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Return eccentricity value(s) at given semi-major axis/axes.
        
        Parameters
        ----------
        a : float or array-like
            Semi-major axis value(s).
            
        Returns
        -------
        ndarray
            Eccentricity value(s) corresponding to the input semi-major axis/axes.
        """
        a_array: npt.NDArray[np.float64] = np.asarray(a, dtype=float)
        e_vals: npt.NDArray[np.float64] = self.eccentricity_func(a_array)
        
        return e_vals

    def distribution(
            self, 
            e: Union[float, npt.NDArray[np.float64]], 
            a: Union[float, npt.NDArray[np.float64]]
        ) -> None:
        """
        Distribution function ψ_e(e,a) is not defined for deterministic profiles.
        
        Raises
        ------
        NotImplementedError
            Always raised since fixed eccentricity profiles do not have a distribution.
        """
        raise NotImplementedError("Fixed eccentricity profiles do not define a distribution ψ_e(e,a).")

    def __str__(self) -> str:
        """
        Return a string representation of the eccentricity profile.
        """
        info: str = f"UniqueEccentricity(a_min={self.a_min}, a_max={self.a_max}"
        if self.default_profile:
            info += f", e0={self.e0}, power={self.power})"
        else:
            info += f", e0={self.e0},  custom_func=True)"
        return info
    
    def __call__(self, a_vals: Union[float, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Evaluate the eccentricity at specified semi-major axis values.
        This method provides a convenient function-like interface for the class.

        Parameters
        ----------
        a_vals (float or array-like): Semi-major axis value(s) to evaluate.

        Returns
        -------
        ndarray: Eccentricity values at the specified semi-major axis values.
        """
        return self.eccentricity(a_vals)

    def derivative(
            self, 
            a: Union[float, npt.NDArray[np.float64]]
        ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Calculate the derivative de/da of the eccentricity profile using analytical or finite differences.
        
        Parameters
        ----------
        a : float or array-like
            Semi-major axis value(s).
            
        Returns
        -------
        float or ndarray
            Derivative value(s) at the specified semi-major axis/axes.
        """
        a_array: npt.NDArray[np.float64] = np.asarray(a, dtype=float)

        if self.default_profile:
            # Analytical derivative for the built-in power-law profile
            result: npt.NDArray[np.float64] = -self.power * self.e0 * (self.a_min ** self.power) * a_array ** (-self.power - 1)
            return result[0] if a_array.size == 1 else result
        else:
            # Central difference method for user-defined functions
            h: float = 1e-5
            f_plus = self.eccentricity_func(a_array + h)
            f_minus = self.eccentricity_func(a_array - h)
            result = (f_plus - f_minus) / (2 * h)
            return result[0] if a_array.size == 1 else result
        
    def plot(
            self, 
            a_vals: Optional[npt.NDArray[np.float64]] = None, 
            num_points: int = 500, 
            save: bool = False, 
            filename: Optional[str] = None, 
            figsize: Tuple[int, int] = (8, 6),
            ax: Optional[plt.Axes] = None, 
            show: bool = True,
            **plot_kwargs
        ):
        """
        Plot the eccentricity profile e(a) with flexible matplotlib customization.

        Parameters
        ----------
        a_vals (array-like, optional): Specific semi-major axis values to plot.
            If None, generate new ones. Defaults to None.
        num_points (int, optional): Number of points to use if generating new values.
            Defaults to 500.
        save (bool, optional): Whether to save the figure to a file. Defaults to False.
        filename (str, optional): Filename to save the figure. If None, a default name
            will be generated. Defaults to None.
        figsize (tuple, optional): Figure size (width, height).
            Defaults to (8, 6).
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None,
             a new figure and axes will be created. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        **plot_kwargs: Additional keyword arguments passed to plt.plot() and ax.set().
            Examples include:
                - color: Color of the line
                - linestyle: Style of the line ('-', '--', '-.', ':')
                - linewidth or lw: Width of the line
                - marker: Point marker style ('o', 's', '^', etc.)
                - alpha: Transparency of the line
                - label: Label for the legend
                - log (bool): Whether to use a logarithmic scale for the y-axis.
                - xlim (tuple): Limits for the x-axis.
                - ylim (tuple): Limits for the y-axis.
                - xlabel (str): Custom x-axis label.
                - ylabel (str): Custom y-axis label.
                - title (str): Plot title.
                - grid (bool): Whether to show the grid.

        Raises
        ------
        ValueError: If save=True but no filename is provided.
        """
        # Calculate or retrieve eccentricity values
        if a_vals is None:
            # Use logarithmic spacing for better visualization of power-law behavior
            a_vals: npt.NDArray[np.float64] = np.linspace(self.a_min, self.a_max, num_points)
        e_vals: npt.NDArray[np.float64] = self.eccentricity(a_vals)

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Set default plot parameters for the line
        default_line_kwargs = {
            'linewidth': 2
        }
        line_kwargs = {**default_line_kwargs, **{k: v for k, v in plot_kwargs.items() if k in plt.rcParams['axes.prop_cycle'].by_key() or k in ['linestyle', 'linewidth', 'marker', 'alpha', 'label']}}
        ax.plot(a_vals, e_vals, **line_kwargs)

        if 'label' in plot_kwargs:
            ax.legend()

        # Set default labels and grid
        ax.set_xlabel(r'Semi-Major Axis, $a$', fontsize=14)
        ax.set_ylabel(r'Eccentricity, $e(a)$', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Apply other plot kwargs to the axes
        ax.set(**{k: v for k, v in plot_kwargs.items() if k not in line_kwargs and k in ['xscale', 'yscale', 'xlim', 'ylim', 'xlabel', 'ylabel', 'title', 'aspect', 'adjustable', 'anchor', 'autoscale_on', 'autoscalex_on', 'autoscaley_on', 'dataLim', 'in_layout', 'label_outer', 'legend_', 'navigate', 'navigate_mode', 'path_effects', 'picker', 'rasterized', 'renderer', 'sketch_params', 'snap', 'stale', 'sticky_edges', 'transform', 'transformed_clip_path_and_transform', 'visible', 'zorder', 'agg_filter', 'alpha', 'animated', 'artist_kwargs', 'axes', 'children', 'clipbox', 'figure', 'frame_on', 'gid', 'in_axes', 'mouseover', 'name', 'path_patch', 'pickable', 'prop_cycle', 'rasterization_zorder', 'renderer_cache', 'sharex', 'sharey', 'tight_layout', 'transform_set', 'url', 'viewLim', 'xaxis', 'yaxis']})

        if 'log' in plot_kwargs and plot_kwargs['log']:
            ax.set_yscale('log')

        plt.tight_layout()

        # Save if requested
        if save:
            if filename is None:
                filename = f"eccentricity_profile.png"  # Default filename
            plt.savefig(filename),
        
        if show:
            plt.show()


class EccentricityDistribution(EccentricityProfile):
    """
    Defines an eccentricity distribution ψ_e(e,a).
    Gridding and interpolation is required if auto_normalise is True, 
    otherwise the distribution is determined directly via the distribution_func.

    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    distribution_func : callable
        Function ψ_e(e,a) to sample; must accept array inputs for both e and a.
    num_e_points : int, optional
        Number of points in e (default 1000).
    num_a_points : int, optional
        Number of points in a (default 1000).
    interpolation_method : str, optional
        'nearest', 'linear' or 'cubic'.
    auto_normalise : bool, optional
        If True, normalise ψ_e along e for each a, this is done via gridding and interpolation.
    grid_type : str, optional
        'uniform', 'warped' or 'adaptive'. (If auto_normalise is True, this must be provided.)
    grid_spread : float, optional
        This is a parameter used in warped and adaptive gridding. 
        The larger the grid_spread, the less concentrated the grid is around sharp features.
    """

    def __init__(
            self, 
            a_min: float, 
            a_max: float, 
            distribution_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            num_e_points: int = 1000, 
            num_a_points: int = 1000,
            interpolation_method: Optional[str] = None,
            auto_normalise: bool = False, 
            grid_type: Optional[str] = None,
            grid_spread: Optional[float] = 1.0,
        ) -> None:

        # Interpolation and grid types:
        interpolation_types = ['nearest', 'linear', 'cubic']
        grid_types = ['uniform', 'warped', 'adaptive']

        # Store parameters
        self.a_min: float = a_min
        self.a_max: float = a_max
        self.num_e_points: int = num_e_points
        self.num_a_points: int = num_a_points
        self.distribution_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]] = distribution_func
        self.interpolation_method: Optional[str] = interpolation_method
        self.auto_normalise: bool = auto_normalise
        self.grid_type: Optional[str] = grid_type
        self.grid_spread: Optional[float] = grid_spread
        # Check if auto_normalise is True:
        if auto_normalise:
            if interpolation_method is None:
                raise ValueError("interpolation_method must be provided if auto_normalise is True.")
            if grid_type is None:
                raise ValueError("grid_type must be provided if auto_normalise is True.")
            if interpolation_method not in interpolation_types:
                raise ValueError("Choose an interpolation method from:", interpolation_types)
            if grid_type not in grid_types:
                raise ValueError("Choose a grid type from:", grid_types)
            
            # Create the grid for normalisation:
            self._create_grid()

            # Evaluate the distribution (ψ_e(e,a)) on the grid:
            if self.e_grid.ndim == 1:
                self.E = np.tile(self.e_grid[:, None], (1, self.a_grid.size))
            else:
                self.E = self.e_grid
            self.A = np.tile(self.a_grid[None, :], (self.E.shape[0], 1))

            self.psi_grid = self.distribution_func(self.E, self.A)
            if self.psi_grid.shape != self.E.shape:
                raise ValueError(f"distribution_func returned shape {self.psi_grid.shape}, expected {self.E.shape}")

            # Normalise the distribution (ψ_e(e,a)) over e for each a:
            integrals = np.trapz(self.psi_grid, x=self.e_grid, axis=0)
            if np.any(integrals == 0):
                raise ValueError("Zero integral found during ψ_e normalization.")
            self.psi_grid /= integrals

            self.N_array = integrals.copy()

            # Build the interpolator:
            # self._build_interpolator()
            self._N_interpolator = interp1d(
                self.a_grid,           
                self.N_array,           
                kind=self.interpolation_method,       
                bounds_error=False,
                fill_value=0
            )
    
    def _create_grid(self) -> None:
        """
        Helper function to create the grid for autoamtic normalisation.
        The user can choose between uniform, warped or adaptive grids.
        """

        if self.grid_type == 'uniform':
            """
            Create a simple uniform grid over the domain.
            """
            self.e_grid = np.linspace(0, 1, self.num_e_points)
            self.a_grid = np.linspace(self.a_min, self.a_max, self.num_a_points)

        elif self.grid_type == 'warped':
            """
            The warped grid is created by summing the magnitude of the gradients over the domain,
            either directly or using a Gaussian filter (to spread out clumped points).
            The cumulative distribution function (CDF) is then built over the sum, and the inverse CDF is used to 
            determine where to place the points in the warped grid.

            Regions with higher gradients will cause large increases in the CDF, 
            hence the points will be spaced more closely in these regions.
            """
            # Create coarse grids over the domain
            coarse_e_points = int(min(self.num_e_points/5, 1000))
            coarse_a_points = int(min(self.num_a_points/5, 1000))

            coarse_e = np.linspace(0, 1, coarse_e_points)
            coarse_a = np.linspace(self.a_min, self.a_max, coarse_a_points)

            # Evaluate the distribution on the coarse grid
            E_coarse, A_coarse = np.meshgrid(coarse_e, coarse_a, indexing='ij')
            psi_coarse = self.distribution_func(E_coarse, A_coarse)

            # Compute the magnitude of the gradient over the coarse grid
            dpsi_de, dpsi_da = np.gradient(psi_coarse, coarse_e, coarse_a, edge_order=2)
            grad_mag = np.sqrt(dpsi_de**2 + dpsi_da**2)

            # Normalise the gradient magnitude to [0, 1] for CDF-based remapping
            grad_norm = grad_mag / np.max(grad_mag) 
            
            # Sum the gradient magnitude over the grid, either directly or using a Gaussian filter (to spread out clumped points)
            if self.grid_spread is None or self.grid_spread == 0:
                grad_e_mean = grad_norm.sum(axis=1)
                grad_a_mean = grad_norm.sum(axis=0)
            else:
                grad_e_mean = gaussian_filter1d(grad_norm.sum(axis=1), sigma=self.grid_spread)
                grad_a_mean = gaussian_filter1d(grad_norm.sum(axis=0), sigma=self.grid_spread)

            # Build cumulative distribution functions for the two grids
            cdf_e = np.cumsum(grad_e_mean)
            cdf_e /= cdf_e[-1]

            cdf_a = np.cumsum(grad_a_mean)
            cdf_a /= cdf_a[-1]

            # Invert to get adaptive grids (spacings is determined by where significant changes happen in the distribution)
            self.e_grid = np.interp(np.linspace(0, 1, self.num_e_points), cdf_e, coarse_e)
            self.a_grid = np.interp(np.linspace(0, 1, self.num_a_points), cdf_a, coarse_a)

            # Ensure unique and sorted grids
            self.e_grid = np.unique(np.sort(self.e_grid))
            self.a_grid = np.unique(np.sort(self.a_grid))

        elif self.grid_type == 'adaptive':
            """
            The adaptive grid is made effectively in the same way as the warped grid, 
            however, the e-grid is built per column (a) of the distribution, instead of over the whole domain.

            This allows for a structured grid over e for each a, while still allowing the grid spacings in e 
            to change for each a.
            """
            # Create coarse grids over the domain
            coarse_e_points = int(min(self.num_e_points / 5, 1000))
            coarse_a_points = int(min(self.num_a_points / 5, 1000))

            coarse_e = np.linspace(0, 1, coarse_e_points)
            coarse_a = np.linspace(self.a_min, self.a_max, coarse_a_points)

            # Evaluate the distribution on the coarse grid
            E_coarse, A_coarse = np.meshgrid(coarse_e, coarse_a, indexing='ij')
            psi_coarse = self.distribution_func(E_coarse, A_coarse)

            # Compute the magnitude of the gradient over the coarse grid
            dpsi_de, dpsi_da = np.gradient(psi_coarse, coarse_e, coarse_a, edge_order=2)
            grad_mag = np.sqrt(dpsi_de**2 + dpsi_da**2)
            grad_norm = grad_mag / np.max(grad_mag)

            # Adapt a-grid via CDF over a (summing the gradient magnitude over e, either directly or using a Gaussian filter)
            if self.grid_spread is None or self.grid_spread == 0:
                grad_a_prof = grad_norm.sum(axis=0)  
            else:
                grad_a_prof = gaussian_filter1d(grad_norm.sum(axis=0), sigma=self.grid_spread)

            # Build the cumulative distribution function for the a-grid
            cdf_a = np.cumsum(grad_a_prof)
            if cdf_a[-1] == 0:
                cdf_a = np.linspace(0, 1, coarse_a_points)
            else:
                cdf_a /= cdf_a[-1]

            # Invert to get adaptive a_grid
            p_a = np.linspace(0, 1, self.num_a_points)
            self.a_grid = np.interp(p_a, cdf_a, coarse_a)

            # Build per-column (a) CDFs over e and invert
            if self.grid_spread is None or self.grid_spread == 0:
                smooth_e = grad_norm.copy()  
            else:
                # apply a Gaussian filter _along_ the e‐axis (axis=0),
                # so that each column in 'a' is smoothed independently:
                smooth_e = gaussian_filter1d(grad_norm, sigma=self.grid_spread, axis=0)

            # Build the cumulative distribution function for the e-grid (one CDF per a-column)
            cdf_e = np.cumsum(smooth_e, axis=0)  

            # Identify any columns that sum to zero
            zero_mask = (cdf_e[-1, :] == 0)

            # Normalize each non-zero column so that its last entry --> 1
            cdf_e[:, ~zero_mask] /= cdf_e[-1, ~zero_mask]

            # For any column that was all zeros, fill it with a uniform ramp 0 --> 1
            for j in np.where(zero_mask)[0]:
                cdf_e[:, j] = np.linspace(0, 1, self.num_e_points)

            # Invert each column's CDF to get the final e_grid
            pct_e = np.linspace(0, 1, self.num_e_points)
            idx_coarse = np.clip(np.searchsorted(coarse_a, self.a_grid), 0, coarse_a_points - 1)

            self.e_grid = np.zeros((self.num_e_points, self.num_a_points))
            for j, jc in enumerate(idx_coarse):
                # For the j-th target 'a', find its coarse‐a index jc,
                # then invert that column’s CDF:
                self.e_grid[:, j] = np.interp(pct_e, cdf_e[:, jc], coarse_e)

            # Sort each column in e (ensuring monotonicity)
            self.e_grid = np.sort(self.e_grid, axis=0)
            
        else:
            raise ValueError("Invalid grid type. Please choose from 'uniform', 'warped', or 'adaptive'.")
    
    def distribution(
            self, 
            e: Union[float, npt.NDArray[np.float64]], 
            a: Union[float, npt.NDArray[np.float64]]
        ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Evaluate the distribution at the provided points.

        If auto_normalise is True, the distribution is evaluated using the interpolator.
        If auto_normalise is False, the distribution is evaluated using analytic distribution function.

        Parameters
        ----------
        e : float or array-like
            Eccentricity values.
        a : float or array-like
            Semi-major axis values.

        Returns
        -------
        psi_e : float or array-like
            Distribution values.
        """
        e_arr = np.atleast_1d(e)
        a_arr = np.atleast_1d(a)

        if not self.auto_normalise:
            if e_arr.ndim > 1 or a_arr.ndim > 1:
                raise ValueError("e and a should be at most 1-dimensional for direct function evaluation.")

            if e_arr.size > 1 and a_arr.size > 1:
                E, A = np.meshgrid(e_arr, a_arr, indexing='ij')
                values = self.distribution_func(E, A)
            else:
                e_broadcast, a_broadcast = np.broadcast_arrays(e_arr, a_arr)
                values = self.distribution_func(e_broadcast, a_broadcast)
            
            return values
        else:
            if e_arr.size > 1 and a_arr.size > 1:
                E, A = np.meshgrid(e_arr, a_arr, indexing='ij')
            else:
                E, A = np.broadcast_arrays(e_arr, a_arr)

            raw_vals = self.distribution_func(E, A)

            N_vals = self._N_interpolator(A)  # A has correct shape now
            normed = raw_vals / N_vals
            return normed
    
    def get_sampled_distribution(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get the sampled distribution (e_grid, a_grid, psi_grid).

        If auto_normalise is True, the sampled distribution is the same as the grid data.
        If auto_normalise is False, the sampled distribution is the distribution function evaluated on the grid.

        Returns
        -------
        e_grid : npt.NDArray[np.float64]
            Eccentricity grid.
        a_grid : npt.NDArray[np.float64]
            Semi-major axis grid.
        psi_grid : npt.NDArray[np.float64]
            Distribution values.
        """
        if self.auto_normalise:
            return self.e_grid, self.a_grid, self.psi_grid
        else:
            self.e_grid = np.linspace(0, 1, self.num_e_points)
            self.a_grid = np.linspace(self.a_min, self.a_max, self.num_a_points)
            return self.e_grid, self.a_grid, self.distribution(self.e_grid, self.a_grid)

    def eccentricity(self, a):
        raise NotImplementedError("This profile defines a distribution, not a unique eccentricity.")

    def plot(
        self, 
        save: bool = False, 
        filename: Optional[str] = None, 
        log: bool = False, 
        vmin: Optional[float] = None, 
        vmax: Optional[float] = None, 
        points: bool = False, 
        point_size: float = 10,
        cmap: str = 'viridis',
    ) -> None:
        """
        Plot the eccentricity distribution.

        Parameters
        ----------
        save : bool
            If True, save figure to `filename`.
        filename : str
            Path to save the figure.
        show_grid : bool
            Overlay the computational grid (only if auto_normalise=True).
        log : bool
            Use logarithmic color scale.
        vmin, vmax : float
            Color scale limits.
        points : bool
            Plot the distribution as points instead of a surface.
        point_size : float
            Size of the points if points=True.
        cmap : str
            Colormap to use.
        """
        plt.figure(figsize=(8, 6))

        if self.auto_normalise:
            # Make sure all necessary arrays exist:
            if (self.E is None or self.A is None 
                or self.psi_grid is None 
                or self.grid_type is None 
                or self.e_grid is None 
                or self.a_grid is None):
                raise RuntimeError("Cannot plot as auto_normalise is set to True but grid data is missing.")

            if self.grid_type == 'adaptive' and not points:
                E_coords = self.e_grid
                A_coords = np.repeat(self.a_grid[np.newaxis, :], self.e_grid.shape[0], axis=0)
                norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
                mesh = plt.pcolormesh(
                    E_coords, 
                    A_coords, 
                    self.psi_grid, 
                    cmap=cmap, 
                    norm=norm, 
                    shading='auto'  # since X, Y have same shape as C
                )
                cbar = plt.colorbar(mesh)

            elif points:
                # If the user specifically wants a scatter‐point plot:
                norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
                plt.scatter(
                    self.E.ravel(), 
                    self.A.ravel(), 
                    c=self.psi_grid.ravel(), 
                    cmap=cmap, 
                    norm=norm, 
                    s=point_size, 
                    edgecolors='none'
                )
                cbar = plt.colorbar()
            else:
                # Fallback: if it's adaptive but not the “not points” case, or if grid_type!='adaptive',
                # just use pcolormesh on the full (E, A) regular mesh you already have:
                norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
                mesh = plt.pcolormesh(
                    self.E, 
                    self.A, 
                    self.psi_grid, 
                    cmap=cmap, 
                    norm=norm, 
                    shading='auto'
                )
                cbar = plt.colorbar(mesh)

        else:
            e_plot = np.linspace(0, 1, self.num_e_points)
            a_plot = np.linspace(self.a_min, self.a_max, self.num_a_points)
            E_plot, A_plot = np.meshgrid(e_plot, a_plot, indexing='ij')
            psi_plot = self.distribution(e_plot, a_plot)
            norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
            if points:
                plt.scatter(
                    E_plot.ravel(), 
                    A_plot.ravel(), 
                    c=psi_plot.ravel(), 
                    cmap=cmap, 
                    norm=norm, 
                    s=point_size, 
                    edgecolors='none'
                )
                cbar = plt.colorbar()
            else:
                mesh = plt.pcolormesh(
                    E_plot, 
                    A_plot, 
                    psi_plot, 
                    cmap=cmap, 
                    norm=norm, 
                    shading='auto'
                )
                cbar = plt.colorbar(mesh)

        if 'cbar' in locals():
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_ylabel(r'$\psi_e(e,a)$', fontsize=14)

        plt.xlabel(r'Eccentricity, $e$', fontsize=14)
        plt.ylabel(r'Semi-Major Axis, $a$', fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(self.a_min, self.a_max)
        plt.gca().autoscale_view()
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()

        if save:
            if filename is None:
                raise ValueError("Filename must be provided if save is True.")
            plt.savefig(filename)
        plt.show()

    def plot_slice(
        self,
        *,
        fix_a: Optional[float] = None,
        fix_e: Optional[float] = None,
        num_points: int = 500,
        save: bool = False,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> None:
        """
        Plot a 1D slice of the 2D eccentricity distribution psi(e, a).

        Parameters
        ----------
        fix_a : float, optional
            If provided, plots psi(e) at fixed semi-major axis a.
        fix_e : float, optional
            If provided, plots psi(a) at fixed eccentricity e.
        num_points : int, optional
            Number of points for plotting. Defaults to 500.
        save : bool, optional
            If True, saves the figure instead of displaying. Defaults to False.
        filename : str, optional
            Filename to save the figure. Required if `save` is True.
        figsize : tuple, optional
            Figure size in inches. Defaults to (8, 6).
        show : bool, optional
            Whether to show the plot. Defaults to True.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        **plot_kwargs : dict
            Additional keyword arguments for `plt.plot`.

        Raises
        ------
        ValueError
            If neither or both of `fix_a` and `fix_e` are provided.
        """
        if (fix_a is None and fix_e is None) or (fix_a is not None and fix_e is not None):
            raise ValueError("Provide exactly one of fix_a or fix_e.")

        if fix_a is not None:
            x_vals = np.linspace(0, 1, num_points)
            y_vals = self.distribution(x_vals, fix_a)
            label = r"$a = {:.2f}$".format(fix_a)
            xlabel = r"Eccentricity, $e$"
            ylabel = r"$\psi_e(e | a)$"
        else:
            x_vals = np.linspace(self.a_min, self.a_max, num_points)
            y_vals = self.distribution(fix_e, x_vals)
            label = r"$e = {:.2f}$".format(fix_e)
            xlabel = r"Semi-Major Axis, $a$"
            ylabel = r"$\psi_e(e | a)$"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        line_kwargs = {
            'linewidth': 2,
        }
        line_kwargs.update(plot_kwargs)

        ax.plot(x_vals, y_vals, **line_kwargs, label=label)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        plt.tight_layout()

        if save:
            if filename is None:
                raise ValueError("Filename must be provided if save=True.")
            plt.savefig(filename)

        if show:
            plt.show()
    
    def _check_lambda_range(self) -> None:
        """
        Helper function to check if the lambda function is valid.
        """
        a_vals: npt.NDArray[np.float64] = np.linspace(self.a_min, self.a_max, 1000)
        lam_vals: npt.NDArray[np.float64] = self.lambda_func(a_vals)
        if np.any(lam_vals <= 0):
            raise ValueError("λ(a) must be strictly positive (λ(a) > 0).")
        if np.any(lam_vals > 1):
            raise ValueError("λ(a) must be ≤ 1 to ensure eccentricities remain < 1.")
        

class RayleighEccentricity(EccentricityDistribution):
    """
    Eccentricity distribution assuming a properly normalized Rayleigh distribution.
    
    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    sigma0 : float, optional
        Amplitude of the power-law scale function sigma(a). Required if sigma_func is not provided.
    power : float, optional
        Power-law slope of sigma(a). Required if sigma_func is not provided.
    sigma_func : callable, optional
        Custom function sigma(a). If provided, sigma0 and power must be omitted.
    num_e_points : int
        Number of eccentricity grid points.
    num_a_points : int
        Number of semi-major axis grid points.
    """
    def __init__(
            self,
            a_min: float, 
            a_max: float,
            sigma0: Optional[float] = None, 
            power: Optional[float] = None, 
            sigma_func: Optional[Callable[[float], float]] = None,
            num_e_points: int = 1000, 
            num_a_points: int = 1000,
        ):
        
        self.a_min: float = a_min
        self.a_max: float = a_max

        # Validate mutually exclusive inputs
        if sigma_func is not None:
            if sigma0 is not None or power is not None:
                raise ValueError("Provide either (sigma0, power) or sigma_func, not both.")
            self.sigma_func: Callable[[float], float] = sigma_func
        else:
            if sigma0 is None or power is None:
                raise ValueError("If sigma_func is not provided, both sigma0 and power must be specified.")
            self.sigma0: float = sigma0
            self.power: float = power
            self.sigma_func: Callable[[float], float] = lambda a: sigma0 * (a_min / a)**power

        # Define Rayleigh distribution
        def rayleigh_dist(
                e: Union[float, npt.NDArray[np.float64]], 
                a: Union[float, npt.NDArray[np.float64]]
            ) -> Union[float, npt.NDArray[np.float64]]:
            """
            Rayleigh distribution:
            psi(e,a) = (e / sigma(a)^2) * exp(-e^2 / (2 * sigma(a)^2)) / (1 - exp(-1 / (2 * sigma(a)^2)))
            """
            sigma: float = self.sigma_func(a)
            norm_factor: float = 1 - np.exp(-1 / (2 * sigma**2))
            return (e / sigma**2) * np.exp(-e**2 / (2 * sigma**2)) / norm_factor

        super().__init__(
            a_min=a_min,
            a_max=a_max,
            distribution_func=rayleigh_dist,
            num_e_points=num_e_points,
            num_a_points=num_a_points,
            auto_normalise=False,
        )

class TopHatEccentricity(EccentricityDistribution):
    """
    Eccentricity distribution :math:`\psi(e,a) = (1 / \lambda(a)) * e / \sqrt(\lambda(a)^2 - e^2)`,
    which yields a step-function kernel :math:`\phi(kappa,a) = (\pi / 2\lambda(a)) · 1_{kappa <= \lambda(a)}`.

    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    lam : callable or float
        The user can either provide a callable lambda(a) or a constant lambda0.
    num_e_points : int, optional
        Number of eccentricity grid points.
    num_a_points : int, optional
        Number of semi-major axis grid points.
    """

    def __init__(
            self, 
            a_min: float, 
            a_max: float, 
            lam: Union[float, Callable[[float], float]],
            num_e_points: int = 1000, 
            num_a_points: int = 1000,
        ):
        
        self.a_min: float = a_min
        self.a_max: float = a_max

        if callable(lam):
            self.lambda_func: Callable[[float], float] = lam
        elif isinstance(lam, (int, float)):
            self.lambda_func: Callable[[float], float] = lambda a: lam * np.ones_like(a)
        else:
            raise ValueError("lambda must be a callable or a float.")

        # Check validity of lambda(a)
        self._check_lambda_range()

        def psi(
                e: Union[float, npt.NDArray[np.float64]], 
                a: Union[float, npt.NDArray[np.float64]]
            ) -> Union[float, npt.NDArray[np.float64]]:
            """
            Eccentricity distribution psi(e,a) = (1 / lambda(a)) * e / sqrt(lambda(a)^2 - e^2)
            """
            e: npt.NDArray[np.float64] = np.atleast_1d(e)
            a: npt.NDArray[np.float64] = np.atleast_1d(a)
            lam: npt.NDArray[np.float64] = self.lambda_func(a)

            out: npt.NDArray[np.float64] = np.zeros_like(e)
            mask: npt.NDArray[np.bool_] = (e >= 0) & (e < lam)
            out[mask] = (1 / lam[mask]) * e[mask] / np.sqrt(lam[mask]**2 - e[mask]**2)
            return out

        super().__init__(
            a_min=a_min,
            a_max=a_max,
            distribution_func=psi,
            num_e_points=num_e_points,
            num_a_points=num_a_points,
            auto_normalise=False
        )

class TriangularEccentricity(EccentricityDistribution):
    """
    Eccentricity distribution:
    :math:`\psi(e,a) = (2e / \lambda(a)^2) * \ln[ (\lambda(a) + \sqrt(\lambda(a)^2 - e^2)) / e ]`,
    valid for :math:`0 \leq e < \lambda(a)`

    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    lam : callable or float
        The user can either provide a callable lambda(a) or a constant lambda0.
    num_e_points : int, optional
        Number of eccentricity grid points.
    num_a_points : int, optional
        Number of semi-major axis grid points.
    """

    def __init__(
            self,
            a_min: float, 
            a_max: float, 
            lam: Union[float, Callable[[float], float]],
            num_e_points: int = 1000, 
            num_a_points: int = 1000,
        ):
        
        self.a_min: float = a_min
        self.a_max: float = a_max

        if callable(lam):
            self.lambda_func: Callable[[float], float] = lam
        elif isinstance(lam, (int, float)):
            self.lambda_func: Callable[[float], float] = lambda a: lam * np.ones_like(a)
        else:
            raise ValueError("lambda must be a callable or a float.")

        self._check_lambda_range()

        def psi_triangular(
                e: Union[float, npt.NDArray[np.float64]], 
                a: Union[float, npt.NDArray[np.float64]]
            ) -> Union[float, npt.NDArray[np.float64]]:
            """
            Eccentricity distribution
            psi(e,a) = (2e / lambda(a)^2) * ln[ (lambda(a) + sqrt(lambda(a)^2 - e^2)) / e ]
            """
            e: npt.NDArray[np.float64] = np.atleast_1d(e)
            a: npt.NDArray[np.float64] = np.atleast_1d(a)
            lam: npt.NDArray[np.float64] = self.lambda_func(a)
            out: npt.NDArray[np.float64] = np.zeros_like(e)

            mask = (e > 0) & (e < lam)
            lam_masked = lam[mask]
            e_masked = e[mask]

            numerator: npt.NDArray[np.float64] = lam_masked + np.sqrt(lam_masked**2 - e_masked**2)
            log_term: npt.NDArray[np.float64] = np.log(numerator / e_masked)

            out[mask] = (2 * e_masked / lam_masked**2) * log_term
            return out

        super().__init__(
            a_min=a_min,
            a_max=a_max,
            distribution_func=psi_triangular,
            num_e_points=num_e_points,
            num_a_points=num_a_points,
            auto_normalise=False
        )

class PowerLawEccentricity(EccentricityDistribution):
    """
    Eccentricity distribution
    psi(e,a) = (2*zeta + 1) * lambda(a)^(-(2*zeta + 1)) * e * (lambda(a)^2 - e^2)^(zeta - 1/2)
    defined for 0 ≤ e < lambda(a), with zeta > 0.

    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    zeta : float
        Power-law shape parameter (must satisfy ζ > 0).
    lam : callable or float
        Function lambda(a). User can provide a callable or a constant lambda.
    num_e_points : int, optional
        Number of eccentricity grid points.
    num_a_points : int, optional
        Number of semi-major axis grid points.
    """

    def __init__(
            self,
            a_min: float,
            a_max: float,
            zeta: float,
            lam: Union[float, Callable[[float], float]],
            num_e_points: int = 1000,
            num_a_points: int = 1000,
        ):

        if zeta <= 0:
            raise ValueError("zeta must be strictly positive (ζ > 0).")
        self.zeta: float = zeta
        self.a_min: float = a_min
        self.a_max: float = a_max

        if callable(lam):
            self.lambda_func: Callable[[float], float] = lam
        elif isinstance(lam, (int, float)):
            self.lambda_func: Callable[[float], float] = lambda a: lam * np.ones_like(a)
        else:
            raise ValueError("lambda must be a callable or a float.")

        self._check_lambda_range()

        def psi_powerlaw(
                e: Union[float, npt.NDArray[np.float64]], 
                a: Union[float, npt.NDArray[np.float64]]
            ) -> Union[float, npt.NDArray[np.float64]]:
            """
            Eccentricity distribution
            psi(e,a) = (2*zeta + 1) * lambda(a)^(-(2*zeta + 1)) * e * (lambda(a)^2 - e^2)^(zeta - 1/2)
            """
            e: npt.NDArray[np.float64] = np.atleast_1d(e)
            a: npt.NDArray[np.float64] = np.atleast_1d(a)
            lam: npt.NDArray[np.float64] = self.lambda_func(a)
            out: npt.NDArray[np.float64] = np.zeros_like(e)

            mask: npt.NDArray[np.bool_] = (e > 0) & (e < lam)
            lam_mask: npt.NDArray[np.float64] = lam[mask]
            e_mask: npt.NDArray[np.float64] = e[mask]

            prefactor: npt.NDArray[np.float64] = (2 * zeta + 1) / lam_mask**(2 * zeta + 1)
            out[mask] = prefactor * e_mask * (lam_mask**2 - e_mask**2)**(zeta - 0.5)
            return out

        super().__init__(
            a_min=a_min,
            a_max=a_max,
            distribution_func=psi_powerlaw,
            num_e_points=num_e_points,
            num_a_points=num_a_points,
            auto_normalise=False,
        )

class TruncGaussEccentricity(EccentricityDistribution):
    """
    Truncated Gaussian eccentricity distribution with a normalisation term.

    psi(e, a) = sqrt(2/pi) * C(a) * [
        exp(-e² / (2sigma_k(a)²)) / sigma_k(a) * erf( sqrt((lambda(a)² - e²) / (2sigma_k(a)²)) )
        + sqrt(2/pi) * exp(-lambda(a)² / (2sigma_k(a)²)) / sqrt(lambda(a)² - e²)
    ]

    Parameters
    ----------
    a_min : float
        Minimum semi-major axis.
    a_max : float
        Maximum semi-major axis.
    sigma : float or callable
        Constant sigma or a function sigma(a). Must be > 0.
    lam : float or callable
        Constant lambda or a function lambda(a). Must be 0 < lambda ≤ 1.
    num_e_points : int
        Number of eccentricity grid points.
    num_a_points : int
        Number of semi-major axis grid points.
    """

    def __init__(
            self, 
            a_min: float, 
            a_max: float, 
            sigma: Union[float, Callable[[float], float]],
            lam: Union[float, Callable[[float], float]],
            num_e_points: int = 1000, 
            num_a_points: int = 1000,
        ):

        self.a_min: float = a_min
        self.a_max: float = a_max

        if callable(sigma):
            self.sigma_func: Callable[[float], float] = sigma
        elif isinstance(sigma, (int, float)):
            self.sigma_func: Callable[[float], float] = lambda a: sigma * np.ones_like(a)
        else:
            raise ValueError("sigma must be a callable or a float.")

        if callable(lam):
            self.lambda_func: Callable[[float], float] = lam
        elif isinstance(lam, (int, float)):
            self.lambda_func: Callable[[float], float] = lambda a: lam * np.ones_like(a)
        else:
            raise ValueError("lambda must be a callable or a float.")

        # Ensure that both lambda and sigma are valid
        self._check_lambda_range()
        self._check_sigma_range()

        def psi(
                e: Union[float, npt.NDArray[np.float64]], 
                a: Union[float, npt.NDArray[np.float64]]
            ) -> Union[float, npt.NDArray[np.float64]]:
            """
            Truncated Gaussian eccentricity distribution with a normalisation term.
            """
            e: npt.NDArray[np.float64] = np.atleast_1d(e)
            a: npt.NDArray[np.float64] = np.atleast_1d(a)
            lam: npt.NDArray[np.float64] = self.lambda_func(a)
            sig: npt.NDArray[np.float64] = self.sigma_func(a)

            out: npt.NDArray[np.float64] = np.zeros_like(e)
            mask: npt.NDArray[np.bool_] = (e >= 0) & (e < lam)

            e_mask: npt.NDArray[np.float64] = e[mask]
            lam_mask: npt.NDArray[np.float64] = lam[mask]
            sig_mask: npt.NDArray[np.float64] = sig[mask]

            # Compute normalization factor C(a)
            C: npt.NDArray[np.float64] = np.sqrt(np.pi / 2) * (1/sig_mask) * 1 / erf(lam_mask / ((np.sqrt(2) * sig_mask)))

            # Compute both terms inside the brackets
            arg1: npt.NDArray[np.float64] = np.sqrt((lam_mask**2 - e_mask**2) / (2 * sig_mask**2))
            term1: npt.NDArray[np.float64] = np.exp(-e_mask**2 / (2 * sig_mask**2))/sig_mask * erf(arg1)
            
            term2: npt.NDArray[np.float64] = np.sqrt(2 / np.pi) * np.exp(-lam_mask**2 / (2 * sig_mask**2)) / np.sqrt(lam_mask**2 - e_mask**2)

            result: npt.NDArray[np.float64] = np.sqrt(2 / np.pi) * C * e_mask * (term1 + term2)
            out[mask] = result
            return out

        super().__init__(
            a_min=a_min,
            a_max=a_max,
            distribution_func=psi,
            num_e_points=num_e_points,
            num_a_points=num_a_points,
            auto_normalise=False
        )

    def _check_sigma_range(self):
        """
        Helper function to check that sigma(a) is strictly positive.
        """
        a_vals: npt.NDArray[np.float64] = np.linspace(self.a_min, self.a_max, 1000)
        sig_vals: npt.NDArray[np.float64] = self.sigma_func(a_vals)
        if np.any(sig_vals <= 0):
            raise ValueError("sigma(a) must be strictly positive.")