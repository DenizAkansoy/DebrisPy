# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
from numpy.polynomial.legendre import leggauss

from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.special import erf, gamma
from scipy.integrate import quad

from functools import singledispatchmethod, partial
from typing import Optional, Callable, Tuple, Union

from tqdm import tqdm
from joblib import Parallel, delayed
import adaptive 

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import warnings
from .eccentricity import (
    UniqueEccentricity,
    RayleighEccentricity,
    TopHatEccentricity,
    TriangularEccentricity,
    PowerLawEccentricity,
    TruncGaussEccentricity,
    EccentricityDistribution,
    EccentricityProfile,
)
# ------------------------------------------------------------------------------------------------ #


class Kernel:
    """
    Kernel class for easy conversion of a defined eccentrcity profile into the 
    eccentricity kernel required for calculating the ASD.
    """
    def __init__(
            self,
            eccentricity_profile: EccentricityProfile,
            r_min: float,
            r_max: float,
            num_a_points: int = 500,
            num_r_points: int = 500,
    ) -> None:
        """
        Initialize the Kernel Object

        Parameters
        ----------
        eccentricity_profile : EccentricityProfile
            An instance of an EccentricityProfile subclass. 
            This will determine the eccentricity distribution used in calculating the kernel.
        r_min, r_max : float
            Radius range for evaluation
        num_a_points, num_r_points : int
            Resolution of the (a, r) grid
        """
        self.ecc_profile: EccentricityProfile = eccentricity_profile
        self.r_min: float = r_min
        self.r_max: float = r_max
        self.num_a_points: int = num_a_points
        self.num_r_points: int = num_r_points
        self.a_min: float = eccentricity_profile.a_min
        self.a_max: float = eccentricity_profile.a_max
        self.rayleigh_approx: bool = None

        # Define grids
        self.a_grid: np.ndarray = np.linspace(self.ecc_profile.a_min, self.ecc_profile.a_max, self.num_a_points)
        self.r_grid: np.ndarray = np.linspace(self.r_min, self.r_max, self.num_r_points)

        # Result grid and interpolator
        self.Phi_grid = None
        self.Phi_samples = None
        self._interpolator = None

    # Single dispatch method to compute the kernel - when the user calls the compute method, 
    # the appropriate _compute_kernel method is automatically called based on the eccentricity profile type.

    @singledispatchmethod
    def _compute_kernel(self, ecc_profile: EccentricityProfile, a: np.ndarray, r: np.ndarray, **kwargs: any) -> np.ndarray:
        raise TypeError(f"Unsupported eccentricity profile type: {type(ecc_profile).__name__}")
    
    @_compute_kernel.register(UniqueEccentricity)
    def _(self, ecc_profile: UniqueEccentricity, a: np.ndarray, r: np.ndarray, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for UniqueEccentricity:
        Phi(r, a) = 1 / sqrt(e^2 - kappa(r, a)^2) if e^2 - kappa(r, a)^2 > 0, else 0
        """
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)
        # Define the grids for a, r, kappa and e
        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa: np.ndarray = np.abs(1 - r / a)    # shape (A, R)

        e: np.ndarray = self.ecc_profile.eccentricity(a_vec)[:, None]
        # e: np.ndarray = np.atleast_1d(e)[:, None]   # shape (A, 1)

        # Broadcast e to match shape of kappa
        e_broadcasted: np.ndarray = np.broadcast_to(e, kappa.shape)  # shape (A, R)

        # Compute the square root argument safely
        sqrt_arg: np.ndarray = e_broadcasted**2 - kappa**2

        # Mask where sqrt_arg is valid
        valid_mask: np.ndarray = sqrt_arg > 0

        # Initialize Phi
        Phi: np.ndarray = np.zeros_like(kappa)

        # Only assign where sqrt_arg is valid
        Phi[valid_mask] = 1.0 / np.sqrt(sqrt_arg[valid_mask])

        return Phi
    
    @_compute_kernel.register(RayleighEccentricity)
    def _(self, ecc_profile: RayleighEccentricity, a: np.ndarray, r: np.ndarray, approx: bool = False, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for RayleighEccentricity

        Phi(r, a) = 1 / sigma) * exp(-kappa(r, a)^2 / (2 * sigma^2)) if approx = True (low eccentricity limit), else:
        Phi(r, a) = sqrt(pi / 2) * exp(-kappa(r, a)^2 / (2 * sigma^2)) / (sigma * (1 - exp(-1 / (2 * sigma^2)))) 
                    * erf(sqrt((1 - kappa(r, a)^2) / (2 * sigma^2))) (general case)
        """
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)
        # Define the grids for e, r, kappa
        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa: np.ndarray = np.abs(1 - r / a)

        # Define the sigma profile
        sigma0: float = self.ecc_profile.sigma0
        alpha: float = self.ecc_profile.power
        sigma: np.ndarray = sigma0 * (self.a_min / a_vec)**alpha
        sigma: np.ndarray = sigma[:, None]

        if approx:
            # Kernel approximation for the small eccentricity limit
            phi: np.ndarray = np.sqrt(np.pi / 2) * (1 / sigma) * np.exp(-kappa**2 / (2 * sigma**2))
        else:
            # Kernel for the general case
            norm: np.ndarray = sigma * (1 - np.exp(-1 / (2 * sigma**2)))
            phi: np.ndarray = np.sqrt(np.pi / 2) * np.exp(-kappa**2 / (2 * sigma**2)) / norm
            arg: np.ndarray = np.maximum((1 - kappa**2) / (2 * sigma**2), 0)
            phi *= erf(np.sqrt(arg))

        return phi
    
    @_compute_kernel.register(TopHatEccentricity)
    def _(self, ecc_profile: TopHatEccentricity, a: np.ndarray, r: np.ndarray, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for TopHatEccentricity

        Phi(r, a) = (pi / 2 * lambda(a)) if kappa(r, a) <= lambda(a), else 0
        """
        # Define the grids for a, r, kappa
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)
        
        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa: np.ndarray = np.abs(1 - r / a)  # shape (A, R)

        # Define the lambda profile
        lambda_vals: np.ndarray = self.ecc_profile.lambda_func(a_vec)  # shape (A,)
        lam: np.ndarray = lambda_vals[:, None]  # shape (A, 1), needs to be broadcasted

        # Broadcast lambda to match shape of kappa
        lam_full: np.ndarray = np.broadcast_to(lam, kappa.shape)  # shape (A, R)

        # Initialize Phi
        phi: np.ndarray = np.zeros_like(kappa)

        # Only assign where kappa <= lambda
        mask: np.ndarray = kappa <= lam_full
        phi[mask] = (np.pi / 2) / lam_full[mask]

        return phi
    
    @_compute_kernel.register(TriangularEccentricity)
    def _(self, ecc_profile: TriangularEccentricity, a: np.ndarray, r: np.ndarray, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for TriangularEccentricity

        Phi(kappa, a) = pi / lambda(a)^2 · (lambda(a) - kappa) for kappa <= lambda(a), 0 otherwise
        """
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)
        # Define the grids for a, r, kappa
        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa: np.ndarray = np.abs(1 - r / a)  # shape (A, R)

        # Define the lambda profile
        lambda_vals: np.ndarray = self.ecc_profile.lambda_func(a_vec)  # shape (A,)
        lam: np.ndarray = lambda_vals[:, None]  # shape (A, 1) → broadcast to (A, R)

        # Broadcast lambda to match shape of kappa
        lam_full: np.ndarray = np.broadcast_to(lam, kappa.shape)  # shape (A, R)

        # Initialize Phi
        phi: np.ndarray = np.zeros_like(kappa)

        # Only assign where kappa <= lambda
        mask: np.ndarray = kappa <= lam_full
        phi[mask] = np.pi / lam_full[mask]**2 * (lam_full[mask] - kappa[mask])

        return phi

    @_compute_kernel.register(PowerLawEccentricity)
    def _(self, ecc_profile: PowerLawEccentricity, a: np.ndarray, r:np.ndarray, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for PowerLawEccentricity

        Phi(kappa, a) = sqrt(pi) * lambda(a)^-(2*zeta + 1) * Gamma(zeta + 3/2) / Gamma(zeta + 1) * (lambda(a)^2 - kappa^2)^zeta for kappa <= lambda(a)
        """
        # Define the grids for a, r, kappa
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)

        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa: np.ndarray = np.abs(1 - r / a)  # shape (A, R)

        # Get parameters from the profile
        zeta: float = self.ecc_profile.zeta
        lam: np.ndarray = self.ecc_profile.lambda_func(a_vec)[:, None]  # shape (A, 1)

        # Broadcast lambda to match shape of kappa
        lam_full: np.ndarray = np.broadcast_to(lam, kappa.shape)  # shape (A, R)

        # Compute the square of kappa and lambda
        kappa_sq: np.ndarray = kappa**2
        lam_sq: np.ndarray = lam_full**2

        # Constant front factor
        coeff: np.ndarray = (
            np.sqrt(np.pi) *
            lam_full**(-(2 * zeta + 1)) *
            (gamma(zeta + 1.5) / gamma(zeta + 1))
        )

        # Safe computation mask
        mask: np.ndarray = kappa <= lam_full
        phi: np.ndarray = np.zeros_like(kappa)
        phi[mask] = coeff[mask] * (lam_sq[mask] - kappa_sq[mask])**zeta

        return phi

    @_compute_kernel.register(TruncGaussEccentricity)
    def _(self, ecc_profile: TruncGaussEccentricity, a: np.ndarray, r: np.ndarray, **kwargs: any) -> np.ndarray:
        """
        Compute Phi(r, a) for TruncGaussEccentricity

        Phi(kappa, a) = C(a) * exp(-kappa^2 / (2 * sigma_kappa(a)^2)) for kappa <= lambda(a), else 0
        """
        # Define the grids for a, r, kappa
        a_vec = np.atleast_1d(a)
        r_vec = np.atleast_1d(r)

        a: np.ndarray = a_vec[:, None]     # shape (A, 1)
        r: np.ndarray = r_vec[None, :]     # shape (1, R)
        kappa = np.abs(1 - r / a)  # shape (A, R)

        # Define the lambda and sigma profiles
        lam: np.ndarray = self.ecc_profile.lambda_func(a_vec)[:, None]   # shape (A, 1)
        sig: np.ndarray = self.ecc_profile.sigma_func(a_vec)[:, None]    # shape (A, 1)

        # Define normalization constant C(a)
        with np.errstate(divide='ignore', invalid='ignore'):
            C: np.ndarray = np.sqrt(np.pi / 2) * (1 / sig) * (1 / erf(lam / (np.sqrt(2) * sig)))

        # Broadcast lambda, sigma and C to match shape of kappa
        lam_full: np.ndarray = np.broadcast_to(lam, kappa.shape)
        sig_full: np.ndarray = np.broadcast_to(sig, kappa.shape)
        C_full: np.ndarray = np.broadcast_to(C, kappa.shape)

        # Initialize Phi
        phi: np.ndarray = np.zeros_like(kappa)

        # Only assign where kappa <= lambda
        mask: np.ndarray = kappa <= lam_full
        phi[mask] = C_full[mask] * np.exp(-kappa[mask]**2 / (2 * sig_full[mask]**2))

        return phi

    @_compute_kernel.register(EccentricityDistribution)
    def _(self, 
          ecc_profile: EccentricityDistribution, 
          a: np.ndarray, 
          r: np.ndarray,
          method: str = 'gauss', 
          eps: float = 1e-8,
          adaptive_grid: bool = False, 
          tol: float = 1e-10, 
          upper_limit: Optional[Union[float, Callable]] = None,
          adaptive_integration: bool = False, 
          split_points: Optional[list] = None,
          n_points: int = 64, 
          max_level: int = 25, 
          n_jobs: int = 4,
          **kwargs: any
        ) -> np.ndarray:
        """
        Compute Phi(r, a) for EccentricityDistribution (general case)

        Parameters
        ----------
        a : np.ndarray
            Semi-major axis values (unused).
        r : np.ndarray
            Radius values (unused).
        method : str
            Integration method to use.
            Options: 'gauss' (Gauss-Legendre Quadrature), 'trapz' (NumPy Trapezium), 'quad' (SciPy Quad Library)
        eps : float
            Epsilon precision parameter for the integration.
        adaptive_grid : bool
            For Gauss-Legendre Quadrature: Whether to use an adaptive grid for the integration.
        tol : float
            Tolerance parameter for the integration.
        upper_limit (optional): float or int or callable
            Upper limit for the integration.
        adaptive_integration (optional): bool
            For Gauss-Legendre Quadrature: Whether to use an adaptive integration for the integration.
        split_points (optional): list
            For Gauss-Legendre Quadrature: List of split points for the integration.
        n_points (optional): int
            For Gauss-Legendre Quadrature: Number of points for the integration.
        max_level (optional): int
            For Gauss-Legendre Quadrature: Maximum level for the integration.
        n_jobs (optional): int
            Number of jobs to run in parallel.
        **kwargs (optional): any
            Additional keyword arguments.
        
        Raises
        ------
        ValueError: If the integration method is not valid.
        """

        # If upper_limit is a float, convert it to a lambda function
        if isinstance(upper_limit, (int,float)):
            val = float(upper_limit)
            upper_limit = lambda a_val: val
        # This is only computed for the grid - therefore do not need to pass a and r into each method.

        if method == 'trapz' and adaptive_grid:
            raise ValueError("Trapzium rule does not support adaptive gridding!")

        if method == 'trapz':
            return self._compute_trapz(
                            eps=eps,
                            n_jobs=n_jobs
                            )
        elif method == 'gauss':
            return self._compute_gauss(
                            eps=eps, 
                            adaptive_grid=adaptive_grid, 
                            tol=tol, 
                            upper_limit=upper_limit,
                            adaptive_integration=adaptive_integration, 
                            split_points=split_points,
                            n_points=n_points, 
                            max_level=max_level,
                            n_jobs=n_jobs
                        )
        elif method == 'quad':
            return self._compute_quad(
                            eps=eps, 
                            upper_limit=upper_limit,
                            n_jobs=n_jobs
                        )
        else:
            raise ValueError(f"Invalid integration method: {method}. Choose from 'gauss' (recommended), 'trapz', or 'quad'.")
    
    def compute(
            self, 
            eps: float = 1e-8, 
            tol: float = 1e-10, 
            method: str = 'gauss',
            rayleigh_approx: bool = False, 
            adaptive_grid: bool = False,
            upper_limit: Optional[Union[float, Callable]] = None, 
            interpolation_method: str = 'linear',
            adaptive_integration: bool = False, 
            split_points: Optional[list] = None,
            n_points: int = 64, 
            max_level: int = 25,
            n_jobs: int = 4
        ) -> None:
        """
        Compute the kernel for the eccentricity distribution

        Parameters
        ----------
        eps : float
            Epsilon precision parameter for the integration.
        tol : float
            Tolerance parameter for the integration.
        method : str
            Integration method to use.
            Options: 'gauss' (Gauss-Legendre Quadrature), 'trapz' (NumPy Trapezium), 'quad' (SciPy Quad Library)
        rayleigh_approx : bool
            Whether to use the Rayleigh approximation for the integration. (Only used for RayleighEccentricity)
        adaptive_grid : bool
            For Gauss-Legendre Quadrature: Whether to use an adaptive grid for the integration.
        upper_limit : float or callable
            Upper limit for the integration.
        interpolation_method : str
            Interpolation method to use.
            Options: 'linear' (linear interpolation), 'cubic' (cubic interpolation), 'nearest' (nearest neighbor interpolation)
        adaptive_integration : bool
            For Gauss-Legendre Quadrature: Whether to use an adaptive integration for the integration.
        split_points : list
            For Gauss-Legendre Quadrature: List of split points for the integration.
        n_points : int
            For Gauss-Legendre Quadrature: Number of points for the integration.
        max_level : int
            For Gauss-Legendre Quadrature: Maximum level for the integration.
        n_jobs : int
            Number of jobs to run in parallel.
        
        Raises
        ------
        ValueError: If the integration method is not valid.
        """

        Phi = self._compute_kernel(
            self.ecc_profile,
            a = self.a_grid,
            r = self.r_grid,
            approx=rayleigh_approx,
            method=method,
            eps=eps,
            adaptive_grid=adaptive_grid,
            tol=tol,
            upper_limit=upper_limit,
            adaptive_integration=adaptive_integration,
            split_points=split_points,
            n_points=n_points,
            max_level=max_level,
            n_jobs=n_jobs
        )

        self.rayleigh_approx = rayleigh_approx

        if adaptive_grid:
            self.Phi_samples = Phi
        else:
            self.Phi_grid = Phi

        self._build_interpolator(method=interpolation_method)

    def _build_interpolator(
            self, 
            method: str = 'linear'
        ) -> None:
        """
        Build a 2-D interpolator for Phi after sampling.
        Interpolator used for the calculation of Phi(r, a) for the general case (no analytic solution available)
        Works for either a regular grid (Phi_grid) or scattered samples (for adaptive gridding).

        Parameters
        ----------
        method : str
            Interpolation method to use.
            Options: 'linear' (linear interpolation), 'nearest' (nearest neighbor interpolation), 'cubic' (cubic interpolation)
        """
        # Phi_grid is used for structured data (i.e. regular grid)
        if self.Phi_grid is not None:
            self._interpolator = RegularGridInterpolator(
                (self.a_grid, self.r_grid),
                self.Phi_grid,
                method=method,
                bounds_error=False,
                fill_value=0.0
            )
            return

        # Phi_samples used for unstructured data (i.e. adapative gridding of the Kernel)
        if self.Phi_samples is not None:
            pts   = self.Phi_samples[:, :2]   # (a, r) pairs
            vals  = self.Phi_samples[:,  2]   # Phi values
            fill = 0.0

            self._interpolator = partial(
                griddata,
                pts, vals,
                method=method,
                fill_value=fill
                )
            return

        raise RuntimeError("No Phi data present - run compute() first.")

    def _compute_trapz(
            self, 
            show_progress: bool = True, 
            n_jobs: int = 4, 
            eps: float = 1e-5
        ) -> np.ndarray:
        """
        Compute Phi(r,a) for a general eccentricity distribution Psi(e,a)

        Using the trapezium rule (NumPy), parallelised for multiple CPU cores using joblib.
        This method is recommended for short and quick calculations/verification.

        This computes the Phi grid one row at a time, using the trapezium rule. - Parallelization is done over the rows.

        IMPORTANT
        ---------
        This method is only available for uniform Kernel grids.
        This method is only available when the eccentricity a_grid is uniform and the same size as the Kernel a_grid.

        Parameters
        ----------
        show_progress : bool
            Whether to show a progress bar.
        n_jobs : int
            Number of CPU cores to use (Default is 4, choose -1 for all available cores)
        eps : float
            Epsilon precision parameter for the integration. 
            (Default is 1e-5, lower is more accurate but can lead to aliasing/numerical instability when using the trapezium rule)
        """
        # Check if the eccentricity profile is an EccentricityDistribution
        if not isinstance(self.ecc_profile, EccentricityDistribution): 
            raise TypeError("This method requires an EccentricityDistribution object.")

        # Get the sampled distribution from the eccentricity profile
        e_grid, a_grid_psi, psi_grid = self.ecc_profile.get_sampled_distribution()

        # Check if the eccentricity grid is uniform and the same size as the Kernel a_grid - only works for uniform grids
        if not np.allclose(a_grid_psi, self.a_grid):
            raise ValueError("Mismatch between eccentricity a_grid and kernel a_grid. If you are using a warped or adaptive eccentricity grid, don't use this method.")

        # Compute Phi(r,a) for each row of the grid (parallelised over the rows)
        A: int = len(self.a_grid)
        tasks: list[tuple[int, np.ndarray]] = (
            delayed(compute_phi_row_trapz)(i, self.a_grid[i], self.r_grid, e_grid, psi_grid[:, i], eps)
            for i in range(A)
        )

        results: list[tuple[int, np.ndarray]] = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, total=A, desc="Computing Φ(r,a) using Trapezium Rule") if show_progress else tasks
        )

        # Initialize the Phi grid   
        Phi_grid: np.ndarray = np.zeros((A, len(self.r_grid)))

        # Fill the Phi grid with the computed values
        for i, row in results:
            Phi_grid[i] = row
        
        return Phi_grid

    def _compute_gauss(
            self, 
            n_points: int = 64, 
            eps: float = 1e-8,
            tol: float = 1e-10,
            show_progress: bool = True, 
            n_jobs: int = 4, 
            adaptive_grid: bool = False, 
            upper_limit: Optional[Union[float, Callable]] = None, 
            adaptive_integration: bool = False, 
            split_points: Optional[list[Union[float, Callable]]] = None, 
            max_level: int = 25
        ) -> np.ndarray:
        """
        Compute Phi(r,a) for a general eccentricity distribution Psi(e,a) using Gauss-Legendre quadrature.

        Can work with both adaptive and fixed-order Gauss-Legendre quadrature.
        Separately, can also work with both adaptive gridding and fixed (uniform) gridding of the Kernel.
        Saves results internally in Phi_grid or Phi_samples.

        Parameters
        ----------
        n_points : int
            Number of points to use for the Gauss-Legendre quadrature.
        eps : float
            Epsilon precision parameter for the integration (to avoid numerical instabilities).
        tol : float
            Tolerance parameter for the integration (used by both adaptive integration, and adaptive gridding).
        show_progress : bool
            Whether to show a progress bar.
        n_jobs : int
            Number of CPU cores to use (Default is 4, choose -1 for all available cores).
        adaptive_grid : bool
            Whether to use adaptive gridding of the Kernel.
        upper_limit : float or callable
            Upper limit of the integration (used by both adaptive integration, and adaptive gridding).
        adaptive_integration : bool
            Whether to use adaptive integration.
        split_points : list[float | Callable]
            List of points to split the integration at (used by both adaptive integration, and adaptive gridding).
        max_level : int
            Maximum level of the adaptive integration.
        """

        # Check if the eccentricity profile is an EccentricityDistribution
        if not isinstance(self.ecc_profile, EccentricityDistribution):
            raise TypeError("This method requires an EccentricityDistribution object.")

        # If adaptive grid is used, define the scalar function for Learner2D
        if adaptive_grid:
            psi_func = self.ecc_profile.distribution     # function to be integrated over

            if adaptive_integration:
                phi_callable = partial(
                    compute_phi_single_gauss_adaptive, 
                    n_points=n_points,
                    eps=eps,
                    psi_func=psi_func,
                    upper_limit=upper_limit
                )
            else:
                phi_callable = partial(
                    compute_phi_single_gauss, 
                    n_points=n_points,
                    eps=eps,
                    psi_func=psi_func,
                    upper_limit=upper_limit,
                    split_points=split_points
                )

            bounds = [(float(self.a_min), float(self.a_max)),
                    (float(self.r_min), float(self.r_max))]

            learner = adaptive.Learner2D(phi_callable, bounds=bounds)
            self._adaptive_learner = learner

            adaptive.runner.BlockingRunner(
                learner,                    # same learner
                loss_goal=tol,              # same tolerance
                executor=None               # use default executor
            )
            
            # Collect the samples
            Phi_samples = learner.to_numpy()
            return Phi_samples

        elif adaptive_integration:
            A = len(self.a_grid)
            tasks = (
                delayed(compute_phi_row_gauss_adaptive)(i, self.a_grid[i], self.r_grid, self.ecc_profile.distribution,
                                            n_points=n_points, eps=eps, upper_limit=upper_limit, tol=tol, max_level=max_level)
                for i in range(A)
            )

            results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
                tqdm(tasks, total=A, desc="Computing Φ(r,a) [Adaptive Gauss]") if show_progress else tasks
            )

            Phi_grid = np.zeros((A, len(self.r_grid)))
            for i, row in results:
                Phi_grid[i] = row

            return Phi_grid

        else:
            # compute the Phi grid using fixed-order Gauss-Legendre quadrature
            A: int = len(self.a_grid)

            # Define the tasks to be computed in parallel (parallelised over the rows)
            tasks = (
                delayed(compute_phi_row_gauss)(i, self.a_grid[i], self.r_grid, self.ecc_profile.distribution,
                                            n_points=n_points, eps=eps, upper_limit=upper_limit, split_points = split_points)
                for i in range(A)
            )

            results: list[tuple[int, np.ndarray]] = Parallel(n_jobs=n_jobs, max_nbytes=None)(
                tqdm(tasks, total=A, desc="Computing Φ(r,a) [Gauss]") if show_progress else tasks
            )

            # Initialize the Phi grid
            Phi_grid: np.ndarray = np.zeros((A, len(self.r_grid)))

            # Fill the Phi grid with the computed values
            for i, row in results:
                Phi_grid[i] = row

            return Phi_grid

    def _compute_quad(
            self, 
            eps: float = 1e-8,
            tol: float = 1e-10,
            show_progress: bool = True, 
            n_jobs: int = 4, 
            adaptive_grid: bool = False, 
            upper_limit: Optional[Union[float, Callable]] = None
        ) -> np.ndarray:
        """
        Compute Phi(r,a) for a general eccentricity distribution Psi(e,a) using scipy.integrate.quad (adaptive)

        Can be used for both adaptive and uniform gridding of the Kernel.
        
        Parameters
        ----------
        eps : float
            Epsilon precision parameter for the integration (to avoid numerical instabilities).
        tol : float
            Tolerance parameter for the integration.
        show_progress : bool
            Whether to show a progress bar.
        n_jobs : int
            Number of CPU cores to use (Default is 4, choose -1 for all available cores).
        adaptive_grid : bool
            Whether to use adaptive gridding of the Kernel.
        upper_limit : float or callable
            Upper limit of the integration.
        """

        # Check if the eccentricity profile is an EccentricityDistribution
        if not isinstance(self.ecc_profile, EccentricityDistribution):
            raise TypeError("This method requires an EccentricityDistribution object.")

        # If adaptive grid is used, define the scalar function for Learner2D
        if adaptive_grid:
            psi_func = self.ecc_profile.distribution     # function to be integrated over

            phi_callable = partial(
                compute_phi_single_quad, 
                eps=eps,
                psi_func=psi_func,
                upper_limit=upper_limit
            )

            bounds = [(float(self.a_min), float(self.a_max)),
                    (float(self.r_min), float(self.r_max))]

            learner = adaptive.Learner2D(phi_callable, bounds=bounds)
            self._adaptive_learner = learner

            adaptive.runner.BlockingRunner(
                learner,                    # same learner
                loss_goal=tol,              # same tolerance
                executor=None               # use default executor
            )
            
            # Collect the samples
            Phi_samples = learner.to_numpy()
            return Phi_samples

        else:
            A = len(self.a_grid)
            tasks = (
                delayed(compute_phi_row_quad)(i, self.a_grid[i], self.r_grid, self.ecc_profile.distribution,
                                                        eps=eps, upper_limit=upper_limit)
                for i in range(A)
            )

            results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
                tqdm(tasks, total=A, desc="Computing Φ(r,a) [Adaptive Gauss]") if show_progress else tasks
            )

            Phi_grid = np.zeros((A, len(self.r_grid)))
            for i, row in results:
                Phi_grid[i] = row

            return Phi_grid
        
    def get_values(
            self, 
            a_vals, 
            r_vals
        ) -> np.ndarray:
        a_vals = np.atleast_1d(a_vals)
        r_vals = np.atleast_1d(r_vals)

        if type(self.ecc_profile) is EccentricityDistribution:
            A, R = np.meshgrid(a_vals, r_vals, indexing='ij')  # shape (len(a_vals), len(r_vals))
            points = np.column_stack((A.ravel(), R.ravel()))
            result = self._interpolator(points)
            return result.reshape(A.shape) if result.size > 1 else result[0]
        else:
            return self._compute_kernel(self.ecc_profile, a = a_vals, r = r_vals, approx=self.rayleigh_approx)
    
    def phi_grid(self) -> np.ndarray:
        """
        Returns phi_grid if it has been computed.
        """
        if self.Phi_grid is None:
            raise RuntimeError("Phi_grid is not computed. Run compute() first with adaptive_grid=False.")
        return self.Phi_grid
    
    def phi_samples(self) -> np.ndarray:
        """
        Returns phi_samples if it has been computed.
        """
        if self.Phi_samples is None:
            raise RuntimeError("Phi_samples is not computed. Run compute() first with adaptive_grid=True.")
        return self.Phi_samples

    def compute_grad(self) -> None:
        """
        Compute the gradient of Phi(r,a) for a unique eccentricity distribution e = e(a), for the initialised grid.
        Uses the chain rule to compute the gradient.
        """
        if isinstance(self.ecc_profile, UniqueEccentricity):
            # Initialize the Phi grid
            a = self.a_grid[:, None]          # shape (A,1)
            r = self.r_grid[None, :]          # shape (1,R)

            kappa = np.abs(1.0 - r / a)       # κ(a,r)
            sign_s = np.sign(1.0 - r / a)     # s = ±1, same shape

            # Get the eccentricity and its derivative
            e      = self.ecc_profile.eccentricity(self.a_grid)[:, None]     # (A,1)
            eprime = self.ecc_profile.derivative(self.a_grid)[:, None]       # (A,1)

            # Broadcast the eccentricity and its derivative to the shape of kappa
            e      = np.broadcast_to(e,      kappa.shape)
            eprime = np.broadcast_to(eprime, kappa.shape)

            # Compute the argument of the square root
            D = e**2 - kappa**2                     # argument of the square root
            valid = D > 0                           # inside the ellipse → Φ ≠ 0
            D_safe = D + (~valid) * 1e-15           # add eps where D==0 to avoid /0

            # Initialize the Phi grid
            Phi = np.zeros_like(D)
            Phi[valid] = 1.0 / np.sqrt(D_safe[valid])

            # dD/dr = 2 * (1 - r/a) / a  
            dD_dr = 2.0 * (1.0 - r/a) / a
            dPhi_dr = np.zeros_like(D)
            dPhi_dr[valid] = -0.5 * dD_dr[valid] / (D_safe[valid]**1.5)

            # dD/da = 2 e e′ - 2 (1 - r/a) * (r / a^2)
            dD_da = 2.0 * e * eprime - 2.0 * (1.0 - r/a) * (r / a**2)
            dPhi_da = np.zeros_like(D)
            dPhi_da[valid] = -0.5 * dD_da[valid] / (D_safe[valid]**1.5)

            # Compute the gradient of Phi
            grad_norm = np.sqrt(dPhi_da**2 + dPhi_dr**2)

            # Store the results
            self.dPhi_da_grid   = dPhi_da
            self.dPhi_dr_grid   = dPhi_dr
            self.grad_Phi_norm  = grad_norm

        else:
            raise TypeError("Analytic gradient calculation only implemented for UniqueEccentricity.")

    def num_adaptive_points(self) -> int:
        """
        Return the number of points in the adaptive Phi mesh.
        """
        if self.Phi_samples is None:
            raise RuntimeError("No adaptive samples present.")
        return self.Phi_samples.shape[0]
        
    def plot(
            self, 
            cmap: str = 'viridis', 
            vmin: float = None, 
            vmax: float = None,
            a_lim: tuple = None, 
            r_lim: tuple = None, 
            save: bool = False, 
            filename: str = None,
            shading: str = 'auto', 
            show_edges: bool = True, 
            edgecolor: str = 'k', 
            linewidth: float = 0.2,
            points: bool = False, 
            point_size: float = 10
        ) -> None:
        """
        Main plotting function for the Kernel.

        This function plots the Phi(r,a) grid, or the Phi(r,a) samples, based on the chosen grid values.

        Parameters
        ----------
        cmap : str
            The colormap to use.
        vmin : float
            The minimum value of the colormap.
        vmax : float
            The maximum value of the colormap.
        a_lim : tuple
            The limits of the semi-major axis.
        r_lim : tuple
            The limits of the radius.
        save : bool
            Whether to save the plot.
        filename : str
            The filename to save the plot to.
        shading : str
            The shading to use.
        show_edges : bool
            Whether to show the edges when using triangulation (unstructured grid).
        edgecolor : str
            The color of the edges (unstructured grid)
        linewidth : float
            The width of the edge (unstructured grid)
        points : bool
            Whether to plot a scatter plot instead of a color mesh.
        point_size : float
            The size of the points in the scatter plot.
        """
        plt.figure(figsize=(10, 8))

        if points:
            if self.Phi_grid is not None:
                plt.scatter(self.r_grid, self.a_grid, c=self.Phi_grid, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size)
                cbar = plt.colorbar(label=r'$\Phi(r,a)$')
                plt.grid(True, which='both', linestyle='-', alpha=0.3)
            elif self.Phi_samples is not None:
                a, r, phi = self.Phi_samples.T
                plt.scatter(r, a, c=phi, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size)
                cbar = plt.colorbar(label=r'$\Phi(r,a)$')
                plt.grid(True, which='both', linestyle='-', alpha=0.3)
        else:
            if self.Phi_grid is not None:
                # Regular grid plotting
                mesh = plt.pcolormesh(self.r_grid, self.a_grid, self.Phi_grid,
                                    shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(mesh, label=r'$\Phi(r,a)$')

            elif self.Phi_samples is not None:
                a, r, phi = self.Phi_samples.T
                tri = mtri.Triangulation(r, a)

                tpc = plt.tripcolor(tri, phi, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(tpc, label=r'$\Phi(r,a)$')

                if show_edges:
                    plt.triplot(tri, color=edgecolor, linewidth=linewidth, alpha=0.5)

            else:
                raise RuntimeError("No data available to plot. Run compute() first.")

        if a_lim is None:
            a_lim = (self.a_min, self.a_max)

        if r_lim is None:
            r_lim = (self.r_min, self.r_max)
        
        plt.xlim(r_lim)
        plt.ylim(a_lim)
        plt.xlabel(r'Radius, $r$', fontsize=14)
        plt.ylabel(r'Semi‑Major Axis, $a$', fontsize=14)

        plt.tight_layout()

        if save:
            if filename is None:
                raise ValueError("Provide filename when save=True.")
            plt.savefig(filename, dpi=300)

        plt.show()
    
    def plot_slice(
            self, 
            fix_a: float = None, 
            fix_r: float = None, 
            log_y: bool = False, 
            log_x: bool = False, 
            save: bool = False, 
            filename: str = None,
            x_lim: tuple = None,
            y_lim: tuple = None
        ) -> None:
        """
        Plot a 1D marginal slice of Phi(r, a) at fixed a or fixed r.

        Parameters
        ----------
        fix_a : float
            Value of a at which to fix and vary r.
        fix_r : float
            Value of r at which to fix and vary a.
        log_y : bool
            Whether to plot the y-axis on a logarithmic scale.
        log_x : bool
            Whether to plot the x-axis on a logarithmic scale.
        save : bool
            Whether to save the plot.
        filename : str
            Filename to save to (required if save=True).
        x_lim : tuple
            Limits of the x-axis.
        y_lim : tuple
            Limits of the y-axis.
        """
        if self.Phi_grid is None:
            raise RuntimeError("Phi_grid is not computed. This plot requires a regular grid.")

        if (fix_a is None and fix_r is None) or (fix_a is not None and fix_r is not None):
            raise ValueError("Specify exactly one of fix_a or fix_r.")

        r_vals = self.r_grid  # 1D array of r values
        a_vals = self.a_grid  # 1D array of a values

        if fix_a is not None:
            i = np.argmin(np.abs(a_vals - fix_a))
            x = r_vals
            y = self.Phi_grid[i, :]
            xlabel = r'Radius, $r$'
            title = f'$a = {fix_a}$'
        else:
            j = np.argmin(np.abs(r_vals - fix_r))
            x = a_vals
            y = self.Phi_grid[:, j]
            xlabel = r'Semi‑Major Axis, $a$'
            title = f'$r = {fix_r}$'

        plt.figure(figsize=(7, 5))
        plt.plot(x, y, lw=2, label = title)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(r'$\Phi(r,a)$', fontsize=14)
        plt.legend(fontsize=14)

        if log_y:
            plt.yscale('log')
        if log_x:
            plt.xscale('log')
        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)

        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save:
            if filename is None:
                raise ValueError("Provide filename when save=True.")
            plt.savefig(filename, dpi=300)

        plt.show()


    def plot_phi_kappa(
            self, 
            cmap: str = 'viridis', 
            save: bool = False, 
            filename: str = None, 
            a_slice: float = None
        ) -> None:
            """
            Helper function to plot Phi(kappa, a) using sorted kappa and a 2D color plot,
            plus an optional secondary 1D plot of Phi(kappa) at a given a_slice (default: middle of a_grid).

            Parameters
            ----------
            cmap : str
                The colormap to use.
            save : bool
                Whether to save the plot.
            filename : str
                Filename to save to (required if save=True).
            a_slice : float
                The a-value to slice at (default: middle of a_grid).
            """            
            # Set up the a and r grids
            a = self.a_grid
            r = self.r_grid
            A, R = len(a), len(r)

            # Set up the kappa and phi grids
            a_mesh = a[:, None]  # shape (A, 1)
            r_mesh = r[None, :]  # shape (1, R)
            kappa = np.abs(1 - r_mesh / a_mesh)  # shape (A, R)
            phi = self.phi_grid()  # shape (A, R)

            # Sort kappa and Phi along the r-axis for each row
            kappa_sorted = np.zeros_like(kappa)
            phi_sorted = np.zeros_like(phi)

            # Sort the kappa and phi grids
            for i in range(A):
                sort_idx = np.argsort(kappa[i])
                kappa_sorted[i] = kappa[i, sort_idx]
                phi_sorted[i] = phi[i, sort_idx]

            # Compute kappa edges per row (along r-axis)
            kappa_edges = 0.5 * (kappa_sorted[:, :-1] + kappa_sorted[:, 1:])
            kappa_edges = np.pad(kappa_edges, ((0, 0), (1, 0)), mode='edge')  # shape (A, R)

            # Compute a edges (1D)
            a_edges = 0.5 * (a[:-1] + a[1:])
            a_edges = np.pad(a_edges, (1, 0), mode='edge')  # shape (A+1,)

            # Set up figure with two rows: main image + 1D slice 
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 7), height_ratios=[3, 1], sharex=True, constrained_layout=True
            )

            # Top Plot: 2D colour-mesh
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                mesh = ax1.pcolormesh(
                    kappa_edges, a_edges[:, None], phi_sorted,
                    shading='auto', cmap=cmap
                )
            ax1.set_ylabel(r'$a$', fontsize=14)
            cbar = fig.colorbar(mesh, ax=ax1, label=r'$\Phi(\kappa, a)$')
            cbar.ax.tick_params(labelsize=10)
            ax1.set_xlim(0, 0.99)

            # Bottom Plot: 1D slice at fixed a
            if a_slice is None:
                i_slice = A // 2
                a_slice_val = a[i_slice]
            else:
                i_slice = np.argmin(np.abs(a - a_slice))
                a_slice_val = a[i_slice]

            ax2.plot(
                kappa_sorted[i_slice], phi_sorted[i_slice],
                lw=2, color='darkorange', label = f'$a = {a_slice_val:.1f}$'
            )
            ax2.set_xlabel(r'$\kappa = |1 - r/a|$', fontsize=14)
            ax2.set_ylabel(r'$\Phi(\kappa)$', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=12)
            ax2.set_xlim(0, 0.99)

            if save:
                if filename is None:
                    raise ValueError("Must provide filename if save=True.")
                fig.savefig(filename, bbox_inches='tight')

            plt.show()

    def plot_grad(
            self, 
            type: str = 'norm', 
            vmin: float = None, 
            vmax: float = None, 
            a_lim: tuple = None, 
            r_lim: tuple = None, 
            cmap: str = 'viridis', 
            save: bool = False, 
            filename: str = None
        ) -> None:
        """
        Plot the gradient of the kernel.

        Parameters
        ----------
        type : str
            The type of gradient to plot ('norm', 'da', 'dr', or 'all').
        vmin : float
            The minimum value of the gradient to plot.
        vmax : float
            The maximum value of the gradient to plot.
        a_lim : tuple
            The limits of the a-axis.
        r_lim : tuple
            The limits of the r-axis.
        cmap : str
            The colormap to use.
        save : bool
            Whether to save the plot.
        filename : str
            The filename to save the plot to.
        """
        # Check if gradients have been computed
        if self.dPhi_da_grid is None or self.dPhi_dr_grid is None:
            raise RuntimeError("You must run compute_grad() first.")

        # Define a helper function for single plot
        def single_plot(ax, data, title, label):
            im = ax.pcolormesh(self.r_grid, self.a_grid, data, shading='auto',
                            cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(r'$r$', fontsize=12)
            ax.set_ylabel(r'$a$', fontsize=12)
            ax.set_xlim(r_lim)
            ax.set_ylim(a_lim)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label(label, fontsize=12)

        # Multiple plot case
        if type == 'all':
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

            single_plot(axs[0], self.grad_Phi_norm, r'$|\nabla \Phi|$', r'$|\nabla \Phi(r,a)|$')
            single_plot(axs[1], self.dPhi_da_grid, r'$\partial \Phi / \partial a$', r'$\partial \Phi / \partial a$')
            single_plot(axs[2], self.dPhi_dr_grid, r'$\partial \Phi / \partial r$', r'$\partial \Phi / \partial r$')

            if save:
                if filename is None:
                    raise ValueError("Filename must be provided if save=True.")
                plt.savefig(filename, dpi=300)
            plt.show()
            return

        # Single plot case
        if type == 'norm':
            vals = self.grad_Phi_norm
            label = r'$|\nabla \Phi(r,a)|$'
        elif type == 'da':
            vals = self.dPhi_da_grid
            label = r'$\partial \Phi / \partial a$'
        elif type == 'dr':
            vals = self.dPhi_dr_grid
            label = r'$\partial \Phi / \partial r$'
        else:
            raise ValueError("Invalid type. Use 'norm', 'da', 'dr', or 'all'.")

        plt.figure(figsize=(10, 8))
        im = plt.pcolormesh(self.r_grid, self.a_grid, vals, shading='auto',
                            cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=label)
        plt.xlabel(r'Radius, $r$', fontsize=14)
        plt.ylabel(r'Semi-major Axis, $a$', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.xlim(r_lim)
        plt.ylim(a_lim)

        if save:
            if filename is None:
                raise ValueError("Filename must be provided if save=True.")
            plt.savefig(filename, dpi=300)
    
        plt.show()

def compute_phi_row_trapz(
        i: int, 
        a_val: float, 
        r_grid: np.ndarray, 
        e_grid: np.ndarray, 
        psi_col: np.ndarray, 
        eps: float = 1e-5
    ) -> Tuple[int, np.ndarray]:
    """
    Compute Phi(r,a) for a general eccentricity distribution Psi(e,a)
    This method is used to compute a single row of Phi(r,a) for a given a_val using the trapezium rule.

    Parameters:
    -----------
    i : int
        The index of the a_val in the a_grid.
    a_val : float
        The value of a to compute Phi(r,a) for (row)
    r_grid : np.ndarray
        The r_grid to compute Phi(r,a) for.
    e_grid : np.ndarray
        The eccentricity grid to compute Phi(r,a) from (integrate over).
    psi_col : np.ndarray
        The eccentricity distribution Psi(e,a) to compute Phi(r,a) from.
    eps : float
        Epsilon precision parameter for the integration. 
        (Default is 1e-5, lower is more accurate but can lead to aliasing/numerical instability when using the trapezium rule)
    """
    # Initialize the Phi row
    phi_row: np.ndarray = np.zeros(len(r_grid))

    # Compute the kappa row
    kappa_row: np.ndarray = abs(1.0 - r_grid / a_val)

    # Compute Phi(r,a) for each r_val in the row
    for j, kappa in enumerate(kappa_row):
        mask: np.ndarray = e_grid > kappa
        if not np.any(mask):
            continue

        # Get the eccentricity values and distribution values for the valid mask
        e_vals: np.ndarray = e_grid[mask]
        psi_vals: np.ndarray = psi_col[mask]

        # Compute the square of the eccentricity and the delta term
        delta: np.ndarray = e_vals**2 - kappa**2
        valid_mask: np.ndarray = delta >= eps
        if not np.any(valid_mask):
            continue

        # Compute the square root term and the integrand
        sqrt_term: np.ndarray = np.sqrt(delta[valid_mask])
        integrand: np.ndarray = psi_vals[valid_mask] / sqrt_term

        # Compute the square root term and the integrand
        phi_row[j] = np.trapz(integrand, e_vals[valid_mask])
    return i, phi_row

def compute_phi_row_gauss(
        i: int, 
        a_val: float, 
        r_grid: np.ndarray, 
        psi_func: Callable,
        n_points: int = 64, 
        eps: float = 1e-8,
        upper_limit: Optional[Union[float, Callable]] = None, 
        split_points: Optional[list[Union[float, Callable]]] = None
    ):
    """
    Compute Phi(r,a) for a general eccentricity distribution Psi(e,a)
    Piecewise, fixed-order Gauss-Legendre integration for a single row of the Phi(r,a) kernel.

    Parameters
    ----------
    i : int
        Row index corresponding to a_val.
    a_val : float
        Semi-major axis value at this row.
    r_grid : np.ndarray
        Grid of radius values r.
    psi_func : Callable
        Function ψ(e, a) returning the eccentricity distribution.
    n_points : int
        Number of Gauss-Legendre quadrature points.
    eps : float
        Small epsilon to avoid sqrt singularities.
    upper_limit : float or callable (optional)
        Function of a (i.e., lambda a: … ) returning the upper limit of integration.
        If None, uses fixed upper limit of 1.0.
    split_points : list of float or callables, optional
        Breakpoints for piecewise integration.  Each element may be a number
        or a function of a_val; only those strictly between e_min and e_max
        are used, in ascending order.

    Returns
    -------
    i : int
        The same row index passed in.
    phi_row : ndarray
        The computed Φ values for this a_val across r_grid.
    """
    # Initialize the Phi row
    phi_row: np.ndarray = np.zeros_like(r_grid)

    # precompute nodes and weights on [-1, 1]
    x_nodes, weights = leggauss(n_points)
    kappa_row: np.ndarray = abs(1.0 - r_grid / a_val)
    # loop over radii
    for j, kappa in enumerate(kappa_row):

        # unphysical or degenerate - skip
        if kappa >= 1.0 - eps:
            continue

        # get the minimum eccentricity (eps to avoid singularities)
        e_min: float = kappa + eps

        # figure out e_max
        if upper_limit is None:
            e_max: float = 1.0
        elif callable(upper_limit):
            e_max: float = upper_limit(a_val)
        else:
            raise ValueError("upper_limit must be None or a callable function of a.")

        # if there is nothing to integrate over, skip
        if e_max <= e_min + eps:
            continue 

        # check for split points and build breakpoints list
        breaks: list[float] = [e_min]
        if split_points:
            # evaluate & filter split points
            splits: list[float] = []
            for sp in split_points:
                # evaluate the split point
                sp_val: float = sp(a_val) if callable(sp) else float(sp)
                if (e_min + eps) < sp_val < (e_max - eps):
                    splits.append(sp_val)
            # sort the split points
            for sp_val in sorted(splits):
                breaks.append(sp_val)
        # add the maximum eccentricity
        breaks.append(e_max)

        # initialize the total integral
        total_integral: float = 0.0

        # integrate piecewise (fixed-order Gauss-Legendre)
        for e_lo, e_hi in zip(breaks[:-1], breaks[1:]):
            # linear map from x∈[-1,1] to e∈[e_lo,e_hi]
            mid: float = 0.5 * (e_hi + e_lo)
            half: float = 0.5 * (e_hi - e_lo)
            e_vals: np.ndarray = half * x_nodes + mid

            # evaluate the eccentricity distribution
            try:
                psi_vals = psi_func(e_vals, a_val)
            except Exception:
                # skip this segment if psi() fails
                continue

            # avoid sqrt of negative value
            delta: np.ndarray = e_vals**2 - kappa**2
            valid: np.ndarray = delta > eps
            if not np.any(valid):
                continue

            # build integrand = ψ / sqrt(e² − κ²)
            sqrt_term: np.ndarray = np.zeros_like(delta)
            sqrt_term[valid] = np.sqrt(delta[valid])
            integrand: np.ndarray = np.zeros_like(delta)
            integrand[valid] = psi_vals[valid] / sqrt_term[valid]

            # Gauss–Legendre on this subinterval
            integral_seg: float = np.sum(weights * integrand) * half
            total_integral += integral_seg

        # fill result, converting NaN→0
        phi_row[j] = np.nan_to_num(total_integral)

    return i, phi_row


def compute_phi_row_gauss_adaptive(
        i,
        a_val,
        r_grid,
        psi_func,
        n_points=64,
        eps=1e-8,
        upper_limit=None,
        tol=1e-10,
        max_level=25):
    """
    Adaptive Gauss-Legendre integration for one row of Phi(r,a).

    This is particularly useful when there are discontinuities in the eccentricity distribution,
    but the user does not know where they exactly are (i.e. cannot provide split points).

    Recursively splits [e_min,e_max] until the relative error tolerance is met or the maximum recursion depth is reached.

    Parameters
    ----------
    i : int
        row index
    a_val : float
        semi-major axis
    r_grid : array_like
        radii to evaluate
    psi_func : callable
        psi(e,a) → eccentricity distribution
    n_points : int
        base number of Gauss-Legendre quadrature points
    eps : float
        small offset from kappa to avoid singularity
    upper_limit: None or callable
        if callable, upper_limit(a_val) → e_max; else e_max=1
    tol : float
        relative error tolerance (Default is 1e-10)
    max_level : int
        recursion depth limit (Default is 25)

    Returns
    -------
    i : int
        row index
    phi_row : np.ndarray
        computed Phi(r,a) values for this row
    """
    # Precompute nodes & weights
    x1, w1 = leggauss(n_points)
    x2, w2 = leggauss(2*n_points)
    small_value = 1e-14  # Small value to avoid division by zero in tolerance check

    phi_row: np.ndarray = np.zeros_like(r_grid)

    def gl_quad(
            e_lo: float,
            e_hi: float,
            x_nodes: np.ndarray,
            w_nodes: np.ndarray,
            kappa: float
        ) -> float:
        """
        Single Gauss-Legendre pass on [e_lo, e_hi].

        Parameters
        ----------
        e_lo : float
            lower limit of integration
        e_hi : float
            upper limit of integration
        x_nodes : np.ndarray
            Gauss-Legendre quadrature nodes
        w_nodes : np.ndarray
            Gauss-Legendre quadrature weights
        kappa : float
            kappa value
        """
        # Compute the mid and half points
        mid: float = 0.5*(e_hi + e_lo)
        half: float = 0.5*(e_hi - e_lo)
        e: np.ndarray = half*x_nodes + mid

        # Compute the delta term - mask out small values
        delta: np.ndarray = e*e - kappa*kappa
        valid: np.ndarray = delta > eps
        if not np.any(valid):
            return 0.0

        # Compute the eccentricity distribution
        try:
            psi: np.ndarray = psi_func(e, a_val)
        except Exception:
            return 0.0
        integrand: np.ndarray = np.zeros_like(e)
        integrand[valid] = psi[valid] / np.sqrt(delta[valid])

        # Compute the integral
        return half * np.dot(w_nodes, integrand)

    def adapt(
            e_lo: float,
            e_hi: float,
            kappa: float,
            level: int
        ) -> float:
        """
        Recursive adaptive quadrature.
        If the relative error tolerance is met or the maximum recursion depth is reached, return the fine integral.
        Otherwise, split the interval and recurse.

        Parameters
        ----------
        e_lo : float
            lower limit of integration
        e_hi : float
            upper limit of integration
        kappa : float
            kappa value
        level : int
            recursion depth
        """
        # Compute the coarse and fine integrals
        I_coarse: float = gl_quad(e_lo, e_hi, x1, w1, kappa)
        I_fine: float = gl_quad(e_lo, e_hi, x2, w2, kappa)

        # Compute the error estimate
        err: float = abs(I_fine - I_coarse)

        # If the relative error tolerance is met or the maximum recursion depth is reached, return the fine integral.
        if level >= max_level or err <= tol * max(abs(I_fine), abs(I_coarse), small_value) or abs(e_hi - e_lo) < 1e-12:
            return I_fine

        # If the relative error tolerance is not met, split the interval and recurse.
        mid: float = 0.5*(e_lo + e_hi)
        return (adapt(e_lo, mid,   kappa, level+1) +
                adapt(mid,  e_hi,   kappa, level+1))

    # Loop over all radii, compute the Phi(r,a) values adaptively
    kappa_row: np.ndarray = abs(1.0 - r_grid / a_val)

    for j, kappa in enumerate(kappa_row):
        if kappa >= 1.0 - eps:
            continue
        e_min = kappa + eps
        e_max = (upper_limit(a_val) if callable(upper_limit)
                 else 1.0)
        if e_max <= e_min + eps:
            continue

        phi_row[j] = adapt(e_min, e_max, kappa, level=0)

    return i, phi_row
        

def compute_phi_single_gauss(
        pt: tuple, 
        n_points: int, 
        eps: float, 
        psi_func: Callable, 
        upper_limit: Optional[Callable], 
        split_points: Optional[list[Union[float, Callable]]]
    ):
    """
    Compute Phi(r,a) for a single point using fixed-order Gauss-Legendre quadrature with piecewise integration.
    This is used for fixed-order Gauss-Legendre integration as part of the adaptive grid method.

    Parameters
    ----------
    pt : tuple
        (a, r)
    n_points : int
        Number of Gauss-Legendre quadrature points per subinterval.
    eps : float
        Small epsilon to avoid sqrt singularities and overlapping bounds.
    psi_func : callable
        psi(e,a) → eccentricity distribution
    upper_limit : None or callable
        if callable, upper_limit(a) → e_max; else e_max=1
    split_points : list of float or callables, optional
        Breakpoints for piecewise integration.  Each element may be a number
        or a function of a_val; only those strictly between e_min and e_max
        are used, in ascending order.
    """
    # Adaptive gridding requires a point (a, r)
    # Unpack the point
    a, r = pt

    # Compute the kappa value
    kappa: float = abs(1.0 - r / a)

    # Unphysical case
    if kappa >= 1.0 - eps:
        return 0.0

    # Compute the minimum and maximum eccentricity values
    e_min: float = kappa + eps
    if upper_limit is None:
        e_max: float = 1.0
    elif callable(upper_limit):
        e_max: float = upper_limit(a)
    else:
        raise ValueError("upper_limit must be None or a callable")

    # Degenerate case where e_max <= e_min + eps
    if e_max <= e_min + eps:
        return 0.0

    # Build breakpoints list
    breaks: list[float] = [e_min]
    if split_points:
        # evaluate & filter split points
        splits: list[float] = []
        for sp in split_points:
            # evaluate the split point
            sp_val: float = sp(a) if callable(sp) else float(sp)
            if (e_min + eps) < sp_val < (e_max - eps):
                splits.append(sp_val)
        # sort the split points
        for sp_val in sorted(splits):
            breaks.append(sp_val)
    # add the maximum eccentricity
    breaks.append(e_max)

    # initialize the total integral
    total_integral: float = 0.0

    # precompute nodes and weights on [-1, 1]
    x_nodes, weights = leggauss(n_points)

    # integrate piecewise (fixed-order Gauss-Legendre)
    for e_lo, e_hi in zip(breaks[:-1], breaks[1:]):
        # linear map from x∈[-1,1] to e∈[e_lo,e_hi]
        mid: float = 0.5 * (e_hi + e_lo)
        half: float = 0.5 * (e_hi - e_lo)
        e_vals: np.ndarray = half * x_nodes + mid

        # evaluate the eccentricity distribution
        try:
            psi_vals = psi_func(e_vals, a)
        except Exception:
            # skip this segment if psi() fails
            continue

        # avoid sqrt of negative value
        delta: np.ndarray = e_vals**2 - kappa**2
        valid: np.ndarray = delta > eps
        if not np.any(valid):
            continue

        # build integrand = ψ / sqrt(e² − κ²)
        sqrt_term: np.ndarray = np.zeros_like(delta)
        sqrt_term[valid] = np.sqrt(delta[valid])
        integrand: np.ndarray = np.zeros_like(delta)
        integrand[valid] = psi_vals[valid] / sqrt_term[valid]

        # Gauss–Legendre on this subinterval
        integral_seg: float = np.sum(weights * integrand) * half
        total_integral += integral_seg

    return np.nan_to_num(total_integral)

def compute_phi_single_gauss_adaptive(
        pt: tuple, 
        n_points: int, 
        eps: float, 
        psi_func: Callable, 
        upper_limit: Optional[Union[float, Callable]], 
        tol: float = 1e-10, 
        max_level: int = 25
    ):
    """
    Compute Phi(r,a) for a single point using adaptive Gauss-Legendre quadrature.
    This is used for adaptive integration when using the adaptive grid method.

    Parameters
    ----------
    pt : tuple
        (a, r)
    n_points : int
        Base number of Gauss-Legendre quadrature points.
    eps : float
        Small epsilon to avoid sqrt singularities.
    psi_func : callable
        psi(e,a) → eccentricity distribution
    upper_limit : None or callable
        if callable, upper_limit(a) → e_max; else e_max=1
    tol : float
        Relative error tolerance (Default is 1e-10).
    max_level : int
        Recursion depth limit (Default is 25).
    """
    # Unpack the point
    a, r = pt

    # Compute the kappa value
    kappa: float = abs(1.0 - r / a)

    # Unphysical case
    if kappa >= 1.0 - eps:
        return 0.0

    # Compute the minimum and maximum eccentricity values
    e_min: float = kappa + eps
    if upper_limit is None:
        e_max: float = 1.0
    elif callable(upper_limit):
        e_max: float = upper_limit(a)
    else:
        raise ValueError("upper_limit must be None or a callable")

    # Degenerate case where e_max <= e_min + eps
    if e_max <= e_min + eps:
        return 0.0

    # Precompute nodes & weights for the two rules
    x1, w1 = leggauss(n_points)
    x2, w2 = leggauss(2 * n_points)
    small_value: float = 1e-14  # Small value to avoid division by zero in tolerance check

    def gl_quad(
            e_lo: float,
            e_hi: float,
            x_nodes: np.ndarray,
            w_nodes: np.ndarray,
            kappa_val: float
        ) -> float:
        mid: float = 0.5 * (e_hi + e_lo)
        half: float = 0.5 * (e_hi - e_lo)
        e_vals: np.ndarray = half * x_nodes + mid
        delta: np.ndarray = e_vals**2 - kappa_val**2
        valid: np.ndarray = delta > eps
        if not np.any(valid):
            return 0.0
        try:
            psi_vals = psi_func(e_vals, a)
        except Exception:
            return 0.0
        integrand: np.ndarray = np.zeros_like(e_vals)
        integrand[valid] = psi_vals[valid] / np.sqrt(delta[valid])
        return half * np.dot(w_nodes, integrand)

    def adapt(
            e_lo: float,
            e_hi: float,
            kappa_val: float,
            level: int
        ) -> float:
        I_coarse: float = gl_quad(e_lo, e_hi, x1, w1, kappa_val)
        I_fine: float = gl_quad(e_lo, e_hi, x2, w2, kappa_val)
        err: float = abs(I_fine - I_coarse)

        if level >= max_level or err <= tol * max(abs(I_fine), abs(I_coarse), small_value) or abs(e_hi - e_lo) < 1e-12:
            return I_fine

        mid: float = 0.5 * (e_lo + e_hi)
        I_left: float = adapt(e_lo, mid, kappa_val, level + 1)
        I_right: float = adapt(mid, e_hi, kappa_val, level + 1)
        return I_left + I_right

    return adapt(e_min, e_max, kappa, level=0)


def compute_phi_single_quad(
        pt: Tuple[float, float],
        eps: float,
        psi_func: Callable,
        upper_limit: Optional[Callable],
    ) -> float:
    """
    Compute Phi(r,a) for a single point using scipy.integrate.quad.

    Parameters
    ----------
    pt : tuple
        (a, r)
    eps : float
        Small epsilon to avoid sqrt singularities.
    psi_func : callable
        psi(e, a) → eccentricity distribution. The first argument is eccentricity (float),
        the second is semi-major axis (float).
    upper_limit : callable or None
        Function of a (i.e., lambda a: ...) returning the upper limit of integration.
        If None, uses fixed upper limit of 1.0.

    Returns
    -------
    phi_val : float
        The computed Phi(r, a) value.
    """
    # Unpack the point
    a, r = pt

    # Compute the kappa value
    kappa: float = abs(1.0 - r / a)

    # Unphysical case
    if kappa >= 1.0 - eps:
        return 0.0

    # Compute the minimum and maximum eccentricity values
    e_min: float = kappa + eps
    if upper_limit is None:
        e_max: float = 1.0
    elif callable(upper_limit):
        e_max: float = upper_limit(a)
    else:
        raise ValueError("upper_limit must be None or a callable")

    # Define the integrand function
    def integrand(e: float) -> float:
        delta: float = e**2 - kappa**2
        if delta <= eps:
            return 0.0
        try:
            psi_e: float = psi_func(e, a)
        except Exception:
            return 0.0
        return psi_e / np.sqrt(delta)

    # Perform the integration using quad
    phi_val, abserr = quad(integrand, e_min, e_max)
    return phi_val


def compute_phi_row_quad(
        i: int, 
        a_val: float, 
        r_grid: np.ndarray, 
        psi_func: Callable,
        eps: float = 1e-8,
        upper_limit: Optional[Callable] = None, 
    ):
    """
    Compute Phi(r,a) for an entire row using scipy.integrate.quad.

    Parameters
    ----------
    i : int
        The index of the a_val in the a_grid.
    a_val : float
        The value of a to compute Phi(r,a) for (row)
    r_grid : np.ndarray
        The r_grid to compute Phi(r,a) for.
    psi_func : callable
        psi(e, a) → eccentricity distribution. The first argument is eccentricity (float),
        the second is semi-major axis (float).
    eps : float
        Small epsilon to avoid sqrt singularities.
    upper_limit : callable or None
        Function of a (i.e., lambda a: ...) returning the upper limit of integration.
        If None, uses fixed upper limit of 1.0.

    Returns
    -------
    phi_row : np.ndarray
        The computed Phi(r, a) values for the entire row.
    """
    phi_row: np.ndarray = np.zeros(len(r_grid))
    kappa_row: np.ndarray = abs(1.0 - r_grid / a_val)

    for j, kappa in enumerate(kappa_row):

        # unphysical or degenerate - skip
        if kappa >= 1.0 - eps:
            continue

        # get the minimum eccentricity (eps to avoid singularities)
        e_min: float = kappa + eps

        # figure out e_max
        if upper_limit is None:
            e_max: float = 1.0
        elif callable(upper_limit):
            e_max: float = upper_limit(a_val)
        else:
            raise ValueError("upper_limit must be None or a callable function of a.")

        # if there is nothing to integrate over, skip
        if e_max <= e_min + eps:
            continue 

        # Define the integrand function
        def integrand(e: float) -> float:
            delta: float = e**2 - kappa**2
            if delta <= eps:
                return 0.0
            try:
                psi_e: float = psi_func(e, a_val)
            except Exception:
                return 0.0
            return psi_e / np.sqrt(delta)

        # Perform the integration using quad
        phi_row[j] = quad(integrand, e_min, e_max)[0]

    return i, phi_row