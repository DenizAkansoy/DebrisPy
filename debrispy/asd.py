# Import necessary modules
# ------------------------------------------------------------
import numpy as np
from numpy.polynomial.legendre import leggauss
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize

from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import islice
from typing import Optional, Tuple, Union, TypeVar, Iterable, Iterator, List
from functools import partial


from .eccentricity import EccentricityDistribution, UniqueEccentricity
# ------------------------------------------------------------

# Utility functions for adaptive Gauss-Legendre quadrature
# ------------------------------------------------------------

# Type variables
T = TypeVar("T")

def chunked(
        iterable: Iterable[T], 
        batch_size: int
    ) -> Iterator[List[T]]:
    """
    Return successive chunks of a given size from an iterable.

    Parameters
    ----------
    iterable : Iterable[T]
        Input iterable to be split into chunks.
    batch_size : int
        Maximum size of each chunk.

    Returns
    -------
    Iterator[List[T]]
        Batches of the original iterable as lists of size at most `batch_size`.
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def _sigma_batch(
    batch: List[Tuple[int, float, float, float]],
    kernel,
    sigma_a,
    n_points: int,
    tol_rel: float,
    tol_abs: float,
    max_level: int
    ) -> List[Tuple[int, float]]:
    """
    Compute ASD for a batch of (j, r, a_lo, a_hi) tasks.

    Parameters
    ----------
    batch : List of tuples (j, r, a_lo, a_hi)
        Each task includes an index `j`, a radius value `r`, and integration limits `a_lo`, `a_hi`.
    kernel : Kernel object
        The kernel object, Φ(r, a).
    sigma_a : SigmaA object
        The Sigma(a) surface density profile object.
    n_points : int
        Number of Gauss-Legendre points for integration.
    tol_rel : float
        Relative tolerance for adaptive integration.
    tol_abs : float
        Absolute tolerance for adaptive integration.
    max_level : int
        Maximum recursion depth for adaptivity.

    Returns
    -------
    List of (j, ASD) tuples.
        Index j and the corresponding computed value of ASD.
    """
    return [
        _sigma_at_r_gl(j, r, a_lo, a_hi, kernel, sigma_a, n_points=n_points,
                       tol_rel=tol_rel, tol_abs=tol_abs, max_level=max_level)
        for j, r, a_lo, a_hi in batch
    ]

def gl_quad(
    a_lo: float,
    a_hi: float,
    nodes: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    kernel,
    sigma_a,
    r_j: float
    ) -> float:
    """
    Perform Gauss-Legendre quadrature for ASD over [a_lo, a_hi].

    Parameters
    ----------
    a_lo : float
        Lower integration bound.
    a_hi : float
        Upper integration bound.
    nodes : (N,) array
        Gauss-Legendre nodes in [-1, 1].
    weights : (N,) array
        Corresponding Gauss-Legendre weights.
    kernel : Kernel object
        Phi(r, a) evaluator.
    sigma_a : SigmaA object
        Sigma(a) surface density profile.
    r_j : float
        Evaluation point in radius.

    Returns
    -------
    float
        Integral value of ASD.
    """
    # Compute the midpoint and half-width of the integration interval
    mid = 0.5 * (a_hi + a_lo)
    half = 0.5 * (a_hi - a_lo)

    # Compute the integration points and weights
    a = half * nodes + mid

    # Evaluate the surface density and kernel
    sig_a = sigma_a.get_values(a)
    raw = kernel.get_values(a, r_j)

    # Handle 2D kernel output (e.g. for general eccentricity case)
    if raw.ndim == 2:
        phi = raw[:, 0]
    else:
        phi = raw

    # Compute the integrand
    integrand = (sig_a / a) * phi

    # Perform the Gauss-Legendre quadrature
    return half * np.dot(weights, integrand)

def adapt(
    a_lo: float,
    a_hi: float,
    level: int,
    x1: npt.NDArray[np.float64],
    w1: npt.NDArray[np.float64],
    x2: npt.NDArray[np.float64],
    w2: npt.NDArray[np.float64],
    tol_rel: float,
    tol_abs: float,
    max_level: int,
    kernel,
    sigma_a,
    r_j: float
    ) -> float:
    """
    Perform Gauss-Legendre quadrature at two levels (lower and higher order)
    and adaptively refine the result until the desired tolerance is achieved
    or the maximum recursion depth is reached.

    Parameters
    ----------
    a_lo : float
        Lower limit of integration.
    a_hi : float
        Upper limit of integration.
    level : int
        Current recursion depth.
    x1, w1 : (N,) arrays
        Lower-order Gauss-Legendre nodes and weights.
    x2, w2 : (N,) arrays
        Higher-order Gauss-Legendre nodes and weights.
    tol_rel : float
        Relative tolerance.
    tol_abs : float
        Absolute tolerance.
    max_level : int
        Maximum recursion depth.
    kernel : Kernel
        Kernel object.
    sigma_a : SigmaA
        Surface density profile.
    r_j : float
        Radius value at which the integral is being computed.

    Returns
    -------
    float
        Integral value of ASD.
    """
    # Compute the integral at the lower and higher order
    I1 = gl_quad(a_lo, a_hi, x1, w1, kernel, sigma_a, r_j)
    I2 = gl_quad(a_lo, a_hi, x2, w2, kernel, sigma_a, r_j)
    err = abs(I2 - I1)

    # Check if the integral is within the desired tolerance
    if (err <= tol_rel * abs(I2)) or (err <= tol_abs) or (level >= max_level):
        return I2

    # Recursively refine the integral until the desired tolerance is achieved or the maximum recursion depth is reached
    mid = 0.5 * (a_lo + a_hi)
    return (
        adapt(a_lo, mid, level + 1, x1, w1, x2, w2,
              tol_rel, tol_abs, max_level, kernel, sigma_a, r_j)
        +
        adapt(mid, a_hi, level + 1, x1, w1, x2, w2,
              tol_rel, tol_abs, max_level, kernel, sigma_a, r_j)
    )

def _sigma_at_r_gl(
    j: int,
    r_j: float,
    a_min: Optional[float] = None,
    a_max: Optional[float] = None,
    kernel=None,
    sigma_a=None,
    n_points: int = 64,
    tol_rel: float = 1e-8,
    tol_abs: float = 1e-8,
    max_level: int = 15
    ) -> Optional[Tuple[int, float]]:
    """
    Compute ASD at a specific radius using adaptive Gauss-Legendre quadrature.
    This function calls on the 'adapt' function to perform the adaptive integration.
    This function is used in the 'compute_sigma_r' method, and is not intended to be called directly.

    Parameters
    ----------
    j : int
        Index of the radius value.
    r_j : float
        Radius value at which to compute ASD.
    a_min : float, optional
        Lower bound of integration.
    a_max : float, optional
        Upper bound of integration.
    kernel : Kernel
        Kernel object used to compute Phi(r, a).
    sigma_a : SigmaA
        Surface density profile object.
    n_points : int, optional
        Number of quadrature points (default is 64).
    tol_rel : float, optional
        Relative tolerance for adaptive integration.
    tol_abs : float, optional
        Absolute tolerance for adaptive integration.
    max_level : int, optional
        Maximum recursion depth for adaptive refinement.

    Returns
    -------
    Optional[Tuple[int, float]]
        A tuple containing the index and the computed ASD,
        or None if integration fails.
    """
    try:
        # Compute the Gauss-Legendre nodes and weights for the lower and higher order
        x1, w1 = leggauss(n_points)
        x2, w2 = leggauss(2 * n_points)

        # Set the integration bounds
        if a_min is None:
            a_min = kernel.a_min
        if a_max is None:
            a_max = kernel.a_max

        # Perform the adaptive integration
        val = adapt(a_min, a_max, 0, x1, w1, x2, w2,
                    tol_rel, tol_abs, max_level, kernel, sigma_a, r_j)
        
        return j, val / np.pi

    except Exception as e:
        # If an error occurs, print the error message and return None
        print(f"Worker error at r[{j}]={r_j}: {e!r}")
        return None

# Main class for computing azimuthally averaged surface density Σ̄(r)
# ------------------------------------------------------------

class ASD:
    """
    Computes the azimuthally averaged surface density profile, ASD, from a given
    semi-major axis surface density Sigma(a) and kernel Phi(r, a).

    This class supports all kernel types. 

    For most use simple cases, the `compute_quadvec` method is the most convenient, 
    and does not require any additional tuning of parameters, or knowledge of the integration limits.

    For more complex cases, the `compute_gl` method is more flexible, this uses adaptive Gauss-Legendre quadrature,
    the user can specify the number of integration points, and the relative and absolute tolerances.
    Adaptive limits are supported and recommended for cases with very sharp features in the eccentricity profile.

    Adaptive limits are not supported for calculations involving eccentricity distributions (only for unique eccentricity profiles).
    
    Parameters
    ----------
    kernel : Kernel
        A kernel object used to compute Phi(r, a).
    sigma_a : SigmaA
        A surface density profile object defining Sigma(a).
    """

    def __init__(self, kernel, sigma_a) -> None:
        """
        Initialise the ASD object with a kernel and surface density profile.

        Attributes
        ----------
        kernel : Kernel
            The Phi(r, a) kernel object used in integration.
        sigma_a : SigmaA
            The Sigma(a) surface density profile.
        _sigma_r_vals : Optional[npt.NDArray[np.float64]]
            Cached array of ASD values, if previously computed.
        _r_vals : Optional[npt.NDArray[np.float64]]
            Cached array of r values used to compute ASD.
        _kernel_func : Callable
            Reference to kernel.get_values, used for evaluating Phi(r, a).
        _sigma_r_conv : Optional[npt.NDArray[np.float64]]
            Smoothed ASD values after convolution, if applied.
        _conv_width : Optional[float]
            Gaussian smoothing width (in physical units) used for convolution.
        _conv_M : Optional[int]
            Number of points used for Gaussian smoothing.
        """
        self.kernel = kernel   
        self.sigma_a = sigma_a 
        self._sigma_r_vals = None
        self._r_vals = None
        self._kernel_func = kernel.get_values
        self._sigma_r_conv = None
        self._conv_width = None
        self._conv_M = None

    def integrand(
            self, 
            a: Union[float, npt.NDArray[np.float64]], 
            r_vals: Union[float, npt.NDArray[np.float64]]
            ) -> npt.NDArray[np.float64]:
        """
        Compute the 2D integrand over a grid of a and r values.

        Parameters
        ----------
        a : float or array
            Semi-major axis value(s) at which to evaluate the integrand.
        r_vals : float or array
            Radial location(s) at which the azimuthally averaged profile is to be computed.

        Returns
        -------
        (A, R) array
            The evaluated integrand values over all combinations of `a` and `r_vals`.
        """
        a_arr = np.atleast_1d(a).astype(float)       # Shape (A,)
        r_arr = np.atleast_1d(r_vals).astype(float)  # Shape (R,)

        sig_a = self.sigma_a.get_values(a_arr)       # Shape (A,)
        factor = (sig_a / a_arr)[:, None]            # Shape (A, 1), broadcastable

        phi = self._kernel_func(a_arr, r_arr)        # Shape (A, R)
        return factor * phi                          # Shape (A, R)

    def plot_integrand(
            self,
            a_vals: Optional[npt.NDArray[np.float64]] = None,
            r_vals: Optional[npt.NDArray[np.float64]] = None,
            *, # Everything after this must be provided as keyword arguments
            save: bool = False,
            filename: Optional[str] = None,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            log: bool = True,
            cmap: str = "viridis",
            colorbar: bool = True,
            r_lim: Optional[Tuple[float, float]] = None,
            a_lim: Optional[Tuple[float, float]] = None,
            **imshow_kwargs
            ) -> None:
        """
        Plot the integrand as a 2D colourmap over (r, a).
        Useful for diagnosing sharp features or integration difficulties.

        Parameters
        ----------
        a_vals : array-like, optional
            Semi-major axis values to evaluate the integrand over. If None, uses a default linspace.
        r_vals : array-like, optional
            Radius values to evaluate the integrand over. If None, uses a default linspace.
        save : bool, optional
            If True, the plot is saved to disk. Default is False.
        filename : str, optional
            File path to save the plot. Required if `save` is True.
        vmin : float, optional
            Minimum value for the colour scale. If None, uses the minimum non-zero value.
        vmax : float, optional
            Maximum value for the colour scale. If None, uses the maximum value.
        log : bool, optional
            If True, applies logarithmic colour scaling (ignoring non-positive values). Default is True.
        cmap : str, optional
            Colormap to use for the image. Default is "viridis".
        colorbar : bool, optional
            If True, adds a colourbar to the plot. Default is True.
        r_lim : tuple of float, optional
            Limits for the x-axis (r-axis).
        a_lim : tuple of float, optional
            Limits for the y-axis (a-axis).
        **imshow_kwargs : dict
            Additional keyword arguments passed to `imshow()`.

        Raises
        ------
        ValueError
            If `save=True` but `filename` is not provided.
        """

        # Set default values if not provided
        if a_vals is None:
            a_vals = np.linspace(self.kernel.a_min, self.kernel.a_max, 1000)
        if r_vals is None:
            r_vals = np.linspace(self.kernel.r_min, self.kernel.r_max, 1000)

        # Compute the integrand
        Z = self.integrand(a_vals, r_vals)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        if log:
            Z_plot = np.ma.masked_where(Z <= 0.0, Z)

            if vmax is None: 
                vmax = Z_plot.max()
            if vmin is None:
                vmin = Z_plot.min()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmax is None:
                vmax = Z.max()
            if vmin is None:
                vmin = Z.min()
            Z_plot, norm = Z, Normalize(vmin=vmin, vmax=vmax)

        # Create the image
        im = ax.imshow(Z_plot,
                    origin="lower",
                    aspect="auto",
                    extent=[r_vals[0], r_vals[-1], a_vals[0], a_vals[-1]],
                    cmap=cmap,
                    norm=norm,
                    **imshow_kwargs)

        # Set the labels and tick parameters
        ax.set_xlabel("Radius,  $r$", fontsize=14)
        ax.set_ylabel("Semi-Major Axis,  $a$", fontsize=14)
        ax.tick_params(labelsize=12)

        # Add the colorbar if requested
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label("Integrand Value", fontsize=14)
            cbar.ax.tick_params(labelsize=12)

        # Set the limits if provided
        if r_lim is not None:
            ax.set_xlim(r_lim)
        if a_lim is not None:
            ax.set_ylim(a_lim)
        
        ax.grid(True, alpha=0.3)

        # Save the plot if requested
        if save:
            if filename is None:
                raise ValueError("filename must be provided if save is True")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    def compute_quadvec(
            self,
            r_vals: Union[float, npt.NDArray[np.float64]],
            tol_rel: float = 1e-8,
            tol_abs: float = 1e-8,
            ) -> None:
        """
        Compute and cache the azimuthally averaged surface density
        using vectorised adaptive quadrature via `scipy.integrate.quad_vec`.

        This method uses non-adaptive integration limits.

        Parameters
        ----------
        r_vals : float or array-like
            Radius or array of radii at which to compute ASD.
            The result is cached internally and accessible via `self._sigma_r_vals`.
        tol_rel : float, optional
            Relative tolerance for the integration.
        tol_abs : float, optional
            Absolute tolerance for the integration.
        """

        # Ensure r_vals is a 1D array
        r_vals = np.atleast_1d(r_vals)

        # Define integration range from kernel
        a_min = self.kernel.a_min
        a_max = self.kernel.a_max

        # Integrate using quad_vec over a_vals for all r_vals simultaneously
        result, _ = quad_vec(lambda a: self.integrand(a, r_vals),
                            a_min, a_max,
                            epsabs=tol_abs,
                            epsrel=tol_rel)

        # Store the result, dividing by π as per the definition of ASD
        self._sigma_r_vals = result.squeeze() / np.pi
        self._r_vals = r_vals

    def _get_integration_bounds(
            self, 
            r: float, 
            a_grid: npt.NDArray[np.float64], 
            ecc_vals: npt.NDArray[np.float64], pad: float
            ) -> Tuple[float, float, bool]:
        """
        Determine integration bounds in semi-major axis `a` for a given radius `r`.

        Parameters
        ----------
        r : float
            Radius at which Σ̄(r) is being computed.
        a_grid : ndarray
            Precomputed array of semi-major axis values.
        ecc_vals : ndarray
            Array of eccentricity values e(a).
        pad : float
            Padding added to the integration bounds.

        Returns
        -------
        a_lo : float
            Lower integration limit (max with kernel.a_min to avoid underflow).
        a_hi : float
            Upper integration limit (min with kernel.a_max to avoid overflow).
        has_bounds : bool
            True if bounds were tightened based on the condition; False if full range is used.
        """
        kappa = np.abs(1 - r / a_grid)
        mask = ecc_vals > kappa
        if np.any(mask):
            a_lo = max(a_grid[mask][0] - pad, self.kernel.a_min)
            a_hi = min(a_grid[mask][-1] + pad, self.kernel.a_max)
            return a_lo, a_hi, True
        return self.kernel.a_min, self.kernel.a_max, False

    def _find_internal_zeros(self, arr: npt.NDArray[np.float64]) -> List[int]:
        """
        Helper function used to identify indices of internal zero entries in a 1D array.

        This function returns the indices of all elements that are exactly zero and lie strictly
        between the first and last non-zero elements. This is used to catch suspicious zeros where 
        the integration may have failed.

        Parameters
        ----------
        arr : ndarray
            1D array of values, typically from numerical integration.

        Returns
        -------
        List[int]
            Indices of internal zeros (i.e., excluding leading or trailing zeros).
        """
        zeros = np.where(arr == 0.0)[0]
        nz = np.where(arr != 0.0)[0]
        if nz.size == 0:
            return []
        first_nz, last_nz = nz[0], nz[-1]
        return [i for i in zeros if i > first_nz and i < last_nz]

    def _run_pass_gl(
            self,
            kernel,
            sigma_a,
            task_list: List[Tuple[int, float, float, float]],
            pts: int,
            n_jobs: int,
            show_progress: bool,
            batch_size: int,
            tol_rel: float,
            tol_abs: float,
            max_level: int
            ) -> List[Optional[Tuple[int, float]]]:
        """
        Perform a single pass of adaptive Gauss-Legendre quadrature for ASD, serially or in parallel.

        This function evaluates the integral for each task in the list of (r, a_lo, a_hi) tasks.

        Parameters
        ----------
        kernel : Kernel
            The Phi(r, a) kernel used for computing the integrand.
        sigma_a : SigmaA
            The Sigma(a) surface density profile object.
        task_list : list of tuples
            Each tuple is (j, r, a_lo, a_hi) specifying the index `j`, radius `r`, and integration limits.
        pts : int
            Number of Gauss-Legendre points used for each integration (per subinterval).
        n_jobs : int
            Number of parallel workers to use. If 1, runs serially.
        show_progress : bool
            Whether to display a progress bar via `tqdm`.
        batch_size : int
            Number of tasks to group into each batch (only used when `n_jobs > 1`).
        tol_rel : float
            Relative tolerance for the adaptive integration.
        tol_abs : float
            Absolute tolerance for the adaptive integration.
        max_level : int
            Maximum recursion depth for adaptive integration.

        Returns
        -------
        results : list of (j, ASD) or None
            A list of results for each input task, where `j` is the index and the value is the 
            corresponding ASD. Returns `None` if a task fails.
        """
        # Serial execution
        if n_jobs == 1:
            # Define the worker function
            worker_fn = partial(_sigma_at_r_gl, kernel=kernel, sigma_a=sigma_a,
                                n_points=pts, tol_rel=tol_rel,
                                tol_abs=tol_abs, max_level=max_level)

            # Define the iterator
            iterator = tqdm(task_list, desc=f"GL @ {pts} pts") if show_progress else task_list

            # Compute the results
            results = []
            for j, r, a_lo, a_hi in iterator:
                result = worker_fn(j, r, a_lo, a_hi)
                results.append(result)
            return results

        # Parallel execution with batching
        else:
            # Batch the tasks
            batched_tasks = list(chunked(task_list, batch_size))

            # Define the parallel jobs
            jobs = (
                delayed(_sigma_batch)(
                    batch,
                    kernel,
                    sigma_a,
                    pts,
                    tol_rel,
                    tol_abs,
                    max_level
                )
                for batch in batched_tasks
            )

            # Define the iterator
            iterator = tqdm(jobs, total=len(batched_tasks), desc=f"GL @ {pts} pts") if show_progress else jobs

            # Run the parallel jobs
            batched_results = Parallel(
                n_jobs=n_jobs,
                backend='loky',
                max_nbytes=None,
            )(iterator)

            # Flatten batched results
            return [res for batch in batched_results for res in batch]

    def compute_gl(
            self,
            r_vals=None,
            n_points: int = 64,
            tol_rel: float = 1e-8,
            tol_abs: float = 1e-8,
            max_level: int = 15,
            n_jobs: int = -1,
            show_progress: bool = True,
            pad: float = 0.05,
            rf: float = 5.0,
            adaptive_limits: bool = False,
            batch_size: int = 10,
            verbose: bool = True
            ):
        """
        Compute the azimuthally averaged surface density profile, ASD,
        using adaptive Gauss-Legendre quadrature.

        Integration is performed using recursive adaptive Gauss-Legendre quadrature 
        with optional adaptive integration bounds for efficiency (recommended for unique eccentricity profiles).

        Parameters
        ----------
        r_vals : array-like or None
            Array of radius values at which to compute the ASD. If None, defaults to 
            500 evenly spaced points between `kernel.r_min` and `kernel.r_max`.
        n_points : int, optional
            Number of Gauss-Legendre points per subinterval (default: 64).
        tol_rel : float, optional
            Relative tolerance for adaptive integration (default: 1e-8).
        tol_abs : float, optional
            Absolute tolerance for adaptive integration (default: 1e-8).
        max_level : int, optional
            Maximum recursion depth for adaptive integration (default: 15).
        n_jobs : int, optional
            Number of parallel jobs to use (-1 for all available CPUs).
        show_progress : bool, optional
            Whether to show a progress bar using `tqdm` (default: True).
        pad : float, optional
            Padding added to adaptive integration bounds as a fraction of `a` (default: 0.05).
        rf : float, optional
            Rescue factor: multiplier for `n_points` in rescue passes for suspicious results (default: 5.0).
        adaptive_limits : bool, optional
            Whether to use adaptive integration bounds based on the eccentricity profile (default: False).
            Only supported for `UniqueEccentricity` kernels.
        batch_size : int, optional
            Batch size for parallel integration jobs (default: 10).
        verbose : bool, optional
            Whether to print verbose output (default: True).

        Raises
        ------
        ValueError
            If adaptive limits are enabled but the kernel does not use a unique eccentricity profile.

        Prints
        ------
        - Progress bar (optional) and integration diagnostics.
        - Warnings about any suspicious or unresolved zero results.

        Returns
        -------
        None
            Results are stored in `self._r_vals` and `self._sigma_r_vals`.
        """
        # Get the kernel and surface density profile
        kernel = self.kernel
        sigma_a   = self.sigma_a

        # Prepare r grid
        if r_vals is None:
            r_vals = np.linspace(kernel.r_min, kernel.r_max, 500)
        R = len(r_vals)

        # Create input list of valid (idx, r, a_min, a_max) for all r values
        all_tasks = []
        bounds_dict = {}  # Store bounds for each index for rescue passes
        unbounded_indices = set()

        # If adaptive limits are requested, check that the kernel uses a unique eccentricity profile
        if adaptive_limits:
            if not isinstance(kernel.ecc_profile, UniqueEccentricity):
                raise ValueError("Adaptive limits are only available for unique eccentricity profiles.")
            else:
                # Precompute a grid and eccentricity once
                a_grid = np.linspace(kernel.a_min, kernel.a_max, 500_000)
                ecc_vals = kernel.ecc_profile.eccentricity(a_grid)

                # Create the task list and bounds dictionary
                for j, r in enumerate(r_vals):
                    a_lo, a_hi, has_bounds = self._get_integration_bounds(r, a_grid, ecc_vals, pad=pad)
                    all_tasks.append((j, r, a_lo, a_hi))
                    bounds_dict[j] = (a_lo, a_hi)

                    # Add indices with no bounds to the unbounded set
                    if not has_bounds:
                        unbounded_indices.add(j)
                
                if verbose:
                    print("Computing ASD with Gauss-Legendre (with adaptive limits)...")
        else:
            # Create the task list and bounds dictionary
            for j, r in enumerate(r_vals):
                all_tasks.append((j, r, kernel.a_min, kernel.a_max))
                bounds_dict[j] = (kernel.a_min, kernel.a_max)

            if verbose:
                print("Computing ASD with Gauss-Legendre (with fixed limits)...")
    
        # --- FIRST PASS ---
        results1 = self._run_pass_gl(kernel= kernel, sigma_a = sigma_a, 
                                     task_list = all_tasks, pts = n_points, 
                                     n_jobs = n_jobs, show_progress = show_progress, 
                                     batch_size = batch_size, tol_rel = tol_rel, 
                                     tol_abs = tol_abs, max_level = max_level)

        # Create an array to store the results
        sigma_r = np.full(R, np.nan, dtype=float)

        # Fill from first pass results
        for (j, _, _, _), res in zip(all_tasks, results1):
            if res is not None:
                sigma_r[j] = res[1]
            else:
                sigma_r[j] = 0.0 

        # Check for suspicious zeros (i.e. zeros that are not at either end of the grid)
        susp1 = self._find_internal_zeros(sigma_r)

        if susp1:
            if verbose:
                print(f"Initial Pass: Found {len(susp1)} suspicious zero(s). Continuing with rescue pass...\n")

        # Define the rescue routine
        def rescue(
                indices: List[int], 
                factor_label: str, 
                factor: float
                ) -> List[int]:
            """
            Perform a rescue pass on suspicious zero(s) in the ASD.

            This function performs a rescue pass on any suspicious zero(s) in the ASD,
            using a higher-resolution Gauss-Legendre quadrature.

            The higher-resolution is defined by rescue factor, 
            which is a multiplier for the number of Gauss-Legendre points.
            """
            
            # Define the number of Gauss-Legendre points
            pts = int(n_points * factor)
            
            # Create task list for suspicious points only
            rescue_tasks = []
            for idx in indices:
                if idx in bounds_dict:
                    r = r_vals[idx]
                    a_lo, a_hi = bounds_dict[idx]
                    rescue_tasks.append((idx, r, a_lo, a_hi))
            
            if not rescue_tasks:
                return []
            
            # Run the rescue pass
            res = self._run_pass_gl(kernel= kernel, sigma_a = sigma_a, task_list = rescue_tasks, 
                                    pts = pts, n_jobs = n_jobs, show_progress = show_progress, 
                                    batch_size = batch_size, tol_rel = tol_rel, tol_abs = tol_abs, 
                                    max_level = max_level)
            
            # Create a list to store the indices of still zero(s)
            still = []

            # Iterate over the rescue tasks and results
            for (idx, _, _, _), out in zip(rescue_tasks, res):
                if out is None or out[1] == 0.0:
                    still.append(idx)
                else:
                    j, val = out
                    sigma_r[j] = val
            
            # If there are still zero(s), print the number of zero(s)
            if still:
                if verbose:
                    print(f"{factor_label}: Still {len(still)} zero(s).\n")
            else:
                if verbose:
                    print(f"{factor_label}: All points recovered!")
            return still

        # --- RESCUE PASS ---
        # If there are still zero(s), perform a rescue pass
        if susp1:
            # Perform the rescue pass
            still1 = rescue(susp1, "Rescue Pass", rf)

            # If there are still zero(s), perform a final check
            if still1:

                # Define the indices to drop based on they have a valid integration bound
                # If there is no valid region of support, the integration will fail
                # We can drop these indices as they are guaranteed to be zero, this is not a numerical error
                drop_indices = [idx for idx in still1 if idx in unbounded_indices]
                final_bad = [idx for idx in still1 if idx not in unbounded_indices]

                # If there are still zero(s), print the number of zero(s)
                if final_bad:
                    if verbose:
                        print(f"{len(final_bad)} point(s) may still be problematic after rescue -- if so, try increasing rescue factor.")
                    final_rs = [r_vals[i] for i in final_bad]
                    if verbose:
                        print(f"Problematic r-values: {final_rs}")

                if drop_indices:
                    mask = np.ones(len(r_vals), dtype=bool)
                    mask[drop_indices] = False
                    r_vals = r_vals[mask]
                    sigma_r = sigma_r[mask]
                    if verbose:
                        print(f"Removed {len(drop_indices)} points from final results (unbounded)\n")

        # Store results
        self._r_vals = r_vals
        self._sigma_r_vals = sigma_r
        
        if verbose:
            print(f"Done. Final result has {np.sum(~np.isnan(sigma_r))} valid points.\n")

    def refine(
            self,
            curvature_factor: float = 1.0,
            max_rounds: int = 3,
            subdiv: int = 2,
            n_jobs: int = -1,
            show_progress: bool = False,
            n_points: int = 64,
            tol_rel: float = 1e-8,
            tol_abs: float = 1e-8,
            max_level: int = 25,
            adaptive_limits: bool = False,
            rf: float = 10.0, 
            pad: float = 0.05,
            batch_size: int = 10):
        """
        Refine the existing ASD grid by adding intermediate points based on curvature.

        This method uses a curvature-driven refinement strategy: it identifies regions where
        the ASD changes rapidly and adds more sampling points in those regions.
        Optionally, it applies adaptive integration bounds based on the eccentricity profile
        and performs a rescue pass on suspicious zeros.

        This uses the same adaptive Gauss-Legendre quadrature as the `compute_sigma_r` method.
        Currently, this is only available for unique eccentricity profiles.

        Parameters
        ----------
        curvature_factor : float, optional
            Threshold multiplier for triggering refinement based on curvature.
        max_rounds : int, optional
            Maximum number of refinement rounds to perform. Each round may add new points.
        subdiv : int, optional
            Number of subdivisions to insert between points with high curvature.
            For example, `subdiv=2` inserts one point between each flagged pair.
        n_jobs : int, optional
            Number of parallel jobs to use for integration (-1 = all available CPUs).
        show_progress : bool, optional
            Whether to show a progress bar during integration.
        n_points : int, optional
            Number of Gauss-Legendre nodes to use in initial integration.
        tol_rel : float, optional
            Relative tolerance for the adaptive integrator.
        tol_abs : float, optional
            Absolute tolerance for the adaptive integrator.
        max_level : int, optional
            Maximum recursion depth for the adaptive Gauss-Legendre integrator.
        adaptive_limits : bool, optional
            If True, use adaptive (localised) integration bounds based on the eccentricity profile.
            Requires that the kernel's eccentricity profile is `UniqueEccentricity`.
        rf : float, optional
            Rescue factor: multiply `n_points` by this value during the rescue pass to resolve
            suspicious zeros.
        pad : float, optional
            Padding to apply to adaptive integration bounds (in units of semi-major axis).
        batch_size : int, optional
            Batch size for parallel job distribution.

        Raises
        ------
        ValueError
            If `adaptive_limits=True` but the kernel does not have a `UniqueEccentricity` profile.

        Returns
        -------
        None
            Updates the internal `self._r_vals` and `self._sigma_r_vals` arrays with the refined grid.
        """
        # Get the kernel and surface density profile
        kernel = self.kernel
        sigma_a = self.sigma_a

        # Create a dictionary to store the ASD values
        r_to_sigma = {r: s for r, s in zip(self._r_vals, self._sigma_r_vals)}

        # Create a grid of semi-major axis values
        a_grid = np.linspace(kernel.a_min, kernel.a_max, 500_000)

        # Get the eccentricity values
        ecc_vals = kernel.ecc_profile.eccentricity(a_grid)

        # Iterate over the refinement rounds
        for rnd in range(max_rounds):
            print(f"---- Refining Round {rnd+1}/{max_rounds} ----")

            # Get the sorted r-values and ASD values
            rs = np.array(sorted(r_to_sigma))
            ss = np.array([r_to_sigma[r] for r in rs])

            # Compute the finite difference of the ASD values
            diffs = np.abs(np.diff(ss))
            E_tol = curvature_factor * np.median(diffs)

            # Compute the curvature of the ASD values
            # This is a measure of the local curvature of the ASD values
            # If the curvature is greater than the tolerance, the ASD is flagged for refinement
            # The curvature is computed using the second finite difference of the ASD values

            curv = np.abs(ss[2:] - 2 * ss[1:-1] + ss[:-2])
            flagged = np.where(curv > E_tol)[0] + 1
            if not flagged.size:
                break

            # Compute the new points to add
            left, center, right = rs[flagged-1], rs[flagged], rs[flagged+1]
            fracs = np.arange(1, subdiv) / subdiv
            new_pts = np.unique(np.concatenate([
                left[:, None] + (center - left)[:, None] * fracs[None, :],
                center[:, None] + (right - center)[:, None] * fracs[None, :]
            ]).ravel())

            # Compute the new points to add
            new_pts = np.setdiff1d(new_pts, rs, assume_unique=True)
            if new_pts.size == 0:
                break

            # If adaptive limits are requested, compute the integration bounds
            if adaptive_limits:
                task_list = [(i, r, a_lo, a_hi)
                            for i, r in enumerate(new_pts)
                            for (a_lo, a_hi, has_bounds) in [self._get_integration_bounds(r, a_grid, ecc_vals, pad)]]
            else:
                task_list = [(i, r, kernel.a_min, kernel.a_max)
                            for i, r in enumerate(new_pts)]
            
            # Run the integration
            results = self._run_pass_gl(kernel=kernel, sigma_a=sigma_a, task_list=task_list, 
                                        pts=n_points, n_jobs=n_jobs, show_progress=show_progress, 
                                        batch_size=batch_size, tol_rel=tol_rel, tol_abs=tol_abs, 
                                        max_level=max_level)

            # Create a list to store the indices of zero(s)
            zeros = []

            # Iterate over the task list and results
            for (j, r, _, _), res in zip(task_list, results):
                if res is not None:
                    r_to_sigma[r] = res[1]
                    if res[1] == 0.0:
                        zeros.append(r)

            # Construct sorted arrays
            rs_all = np.array(sorted(r_to_sigma))
            sigma_all = np.array([r_to_sigma[r] for r in rs_all])

            # Use existing helper to find suspicious zeros
            suspicious = [rs_all[i] for i in self._find_internal_zeros(sigma_all)]

            print(f"Added {len(new_pts)} points.")

            # If there are still zero(s), perform a rescue pass
            if suspicious:
                print(f"{len(suspicious)} suspicious points, running rescue pass...")

                # Define the task list
                task_list = [
                    (i, r, a_lo, a_hi)
                    for i, r in enumerate(suspicious)
                    for a_lo, a_hi, has_bounds in [self._get_integration_bounds(r, a_grid, ecc_vals, pad)]
                ]

                # Run the rescue pass
                results = self._run_pass_gl(kernel=kernel, sigma_a=sigma_a, task_list=task_list, 
                                            pts=int(n_points*rf), n_jobs=n_jobs, show_progress=show_progress, 
                                            batch_size=batch_size, tol_rel=tol_rel, tol_abs=tol_abs, 
                                            max_level=max_level)
                
                # Create a list to store the indices of zero(s)
                still = []

                # Iterate over the task list and results
                for (j, r, _, _), res in zip(task_list, results):
                    if res is None or res[1] == 0.0:
                        a_lo, a_hi, has_bounds = self._get_integration_bounds(r, a_grid, ecc_vals, pad)
                        if not has_bounds:
                            print(f"Discarding r={r:.6f} (no integration bounds found)")
                            r_to_sigma.pop(r, None)
                        else:
                            still.append(r)
                    else:
                        r_to_sigma[r] = res[1]

                # If there are still zero(s), print the number of zero(s)
                if still:
                    print(f"Rescue pass: still zeros at r-values: {still}")
                else:
                    print("Rescue pass: all recovered or ignored.")

        self._r_vals = np.array(sorted(r_to_sigma))
        self._sigma_r_vals = np.array([r_to_sigma[r] for r in self._r_vals])

    def get_values(self):
        """
        Return the cached ASD values.

        Returns
        -------
        ndarray
            The cached azimuthally averaged surface density values.

        Raises
        ------
        RuntimeError if compute_sigma_r() has not been called yet.
        """
        if self._sigma_r_vals is None:
            raise RuntimeError("You must calculate the ASD before accessing results.")
        return self._r_vals, self._sigma_r_vals
    
    def convolve(self, width: float, M: int = 2048) -> np.ndarray:
        """
        Apply a Gaussian convolution to the computed ASD profile.

        Parameters
        ----------
        width : float
            The Gaussian kernel width (in the same units as r) to smooth ASD.
        M : int, optional
            Number of uniform samples used if r-grid is non-uniform. Default is 2048.

        Returns
        -------
        np.ndarray
            The convolved ASD values at the original r grid.
        """
        r_vals, sigma_r = self.get_values()
        self._conv_width = width
        self._conv_M = M

        # Compute the difference between the r-values
        drs = np.diff(r_vals)

        # If the r-values are uniformly spaced, can convolve directly
        if np.allclose(drs, drs[0]):
            sigma_pix = width / drs[0]
            self._sigma_r_conv = gaussian_filter1d(sigma_r, sigma_pix, mode='nearest')
        else:
            # Non-uniform spacing — interpolate to a uniform grid
            r_min, r_max = r_vals[0], r_vals[-1]
            r_unif = np.linspace(r_min, r_max, M)
            sigma_unif = np.interp(r_unif, r_vals, sigma_r)

            dr = r_unif[1] - r_unif[0]
            sigma_pix = width / dr

            # Convolve the ASD
            sigma_unif_blur = gaussian_filter1d(sigma_unif, sigma_pix, mode='nearest')

            # Interpolate back to the original r-grid
            self._sigma_r_conv = np.interp(r_vals, r_unif, sigma_unif_blur)

        return self._sigma_r_conv

    def plot(
            self, 
            a_vals: np.ndarray, 
            plot_a: bool = True, 
            x_lim: Optional[Tuple[float, float]] = None, 
            y_lim: Optional[Tuple[float, float]] = None, 
            save: bool = False, 
            filename: Optional[str] = None, 
            title: Optional[str] = None, 
            grid_hist: bool = False
            ) -> None:
        """
        Plot Sigma(a) and ASD on the same figure, optionally including a histogram 
        of the r-grid point distribution.

        Parameters
        ----------
        a_vals : array-like
            Array of semi-major axis values at which to evaluate Σ(a).
        plot_a : bool, default=True
            Whether to plot Sigma(a) alongside ASD.
        x_lim : tuple, optional
            Limits for the x-axis.
        y_lim : tuple, optional
            Limits for the y-axis.
        save : bool, default=False
            If True, saves the figure to file instead of displaying it.
        filename : str, optional
            Name of the output file (without extension) if saving the figure.
        title : str, optional
            Title for the plot.
        grid_hist : bool, default=False
            If True, adds a histogram of r grid points above the main plot.
        """

        if plot_a:
            if a_vals is None:
                raise ValueError("a_vals must be provided if plot_a is True")
            sigma_a_vals = self.sigma_a.get_values(a_vals)
        
        if self._sigma_r_vals is None:
            raise RuntimeError("You must calculate the ASD before plotting!")

        r_vals_cached, sigma_r = self.get_values()

        if grid_hist:
            fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.1)

            # Histogram subplot
            ax_top = fig.add_subplot(gs[0])
            ax_top.hist(r_vals_cached, bins=100, color='darkgrey')
            ax_top.grid(True, alpha=0.3)
            ax_top.tick_params(axis='x', labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=13)
            ax_top.set_ylabel(r"$N$", fontsize=16)

            if x_lim is not None:
                ax_top.set_xlim(x_lim)

            # Main plot below
            ax = fig.add_subplot(gs[1], sharex=ax_top)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        if plot_a:
            ax.plot(a_vals, sigma_a_vals, label=r'$\Sigma(a)$', lw=1)
        
        ax.plot(r_vals_cached, sigma_r, label=r'$\bar{\Sigma}(r)$', lw=1)

        # Set the labels and grid
        ax.set_xlabel(r'$a,\ r$', fontsize=16)
        ax.set_ylabel(r"$\Sigma_a(a),\ \bar{\Sigma}(r)$", fontsize=16)

        # Set the limits
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if title is not None:
            ax.set_title(title, fontsize=16)

        # Set the tick parameters
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3)

        if save:
            if filename is None:
                raise ValueError("Filename must be provided if saving the plot.")
            plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_convolution(
            self,
            width: float = None,
            M: int = None,
            overlay_sigma_r: bool = True,
            overlay_sigma_a: bool = False,
            x_lim: Optional[Tuple[float, float]] = None,
            y_lim: Optional[Tuple[float, float]] = None,
            save: bool = False,
            filename: Optional[str] = None,
            title: Optional[str] = None,
            ) -> None:
        """
        Plot the original and convolved ASD profiles.

        Parameters
        ----------
        width : float, optional
            The Gaussian kernel width (in the same units as r) to smooth ASD.
            Required if convolve() has not been called yet.
        M : int, optional
            Number of uniform samples used if r-grid is non-uniform.
            Required if convolve() has not been called yet.
        overlay_sigma_r : bool, optional
            If True, overlay the original ASD profile on the convolved profile.
        overlay_sigma_a : bool, optional
            If True, overlay the Sigma_a profile on the convolved profile.

        Returns
        -------
        None
            Displays the plot.
        """
        r_vals, sigma_r = self.get_values()
        a_vals = np.linspace(min(r_vals), max(r_vals), 1000)

        if width is None:
            if self._conv_width is None:
                raise ValueError("Width must either be provided or computed first via convolve().")
            width = self._conv_width
        
        if M is None:
            if self._conv_M is None:
                raise ValueError("M must either be provided or computed first via convolve().")
            M = self._conv_M

        sigma_conv = self.convolve(width=width, M=M)

        fig, ax = plt.subplots(figsize=(7,5))

        if overlay_sigma_r:
            ax.plot(r_vals, sigma_r,    label=r'$\bar{\Sigma}(r)$',   lw=1.5)
            ax.plot(r_vals, sigma_conv, label=f'Blurred ($\sigma$={width})', lw=1.5)
        else:
            ax.plot(r_vals, sigma_conv, label=f'Blurred ($\sigma$={width})', lw=1.5)
        
        if overlay_sigma_a:
            ax.plot(a_vals, self.sigma_a.get_values(a_vals), label=r'$\Sigma_a(a)$', lw=1, zorder = 0)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if title is not None:
            ax.set_title(title)

        # Set the labels and legend
        ax.set_xlabel(r'$a, r$', fontsize=16)
        ax.set_ylabel(r'$\Sigma_a(a), \bar{\Sigma}(r)$', fontsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)

        if save:
            if filename is None:
                raise ValueError("Filename must be provided if saving the plot.")
            plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()