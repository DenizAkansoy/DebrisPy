# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
from tqdm import tqdm

from .asd import ASD
from .kernel import Kernel 
from .eccentricity import UniqueEccentricity

from matplotlib import pyplot as plt
import fast_histogram
from matplotlib.colors import LogNorm
# ------------------------------------------------------------------------------------------------ #


def kepler_solver(
        M: float, 
        e: float, 
        tol: float = 1e-10, 
        max_iter: int = 100
    ) -> np.ndarray:
    """
    Solve Kepler's equation for the eccentric anomaly E given the mean anomaly M and eccentricity e.
    This function uses Newton-Raphson method to solve the equation.

    Parameters
    ----------
    M : float or array-like
        Mean anomaly.
    e : float or array-like
        Eccentricity.
    tol : float, optional
        Tolerance for the solution.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    E : float or array-like
        Eccentric anomaly.
    """
    # Ensure M and e are arrays
    M = np.atleast_1d(M)
    E = np.copy(M)

    # Iterate until the solution converges
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if np.max(np.abs(delta_E)) < tol:
            break
    return E

class MonteCarlo:
    """
    Monte Carlo sampler for generating particle positions in a debris disc.

    This class generates random samples of semi-major axis `a`, eccentricity `e`,
    and true anomaly `f`, then computes the corresponding radial positions `r`
    using orbital mechanics. The sampling is based on a given surface density 
    profile with respect to semi-major axis (sigma_a) and an eccentricity profile 
    (either unique or a function of semi-major axis)

    Attributes
    ----------
    sigma_a : SigmaA
        The surface density profile used to sample semi-major axis values.
    ecc_profile : EccentricityProfile
        The eccentricity profile used to sample eccentricities (can be unique or a function of semi-major axis)
    n_samples : int
        The total number of Monte Carlo particles to generate.
    a_samples : np.ndarray or None
        Cached array of sampled semi-major axis values after sampling.
    r_samples : np.ndarray or None
        Cached array of radial positions computed from a, e, f.
    e_samples : np.ndarray or None
        Cached array of eccentricity values if manually supplied or reused.
    f_samples : np.ndarray or None
        Cached array of true anomalies used in sampling.
    """
    def __init__(
            self, 
            sigma_a, 
            ecc_profile, 
            n_samples: int = 10_000_000
        ):
        """
        Initialise the Monte Carlo sampler.

        Parameters
        ----------
        sigma_a : SigmaA
            The semi-major axis surface density profile object.
        ecc_profile : EccentricityProfile
            An object representing the eccentricity profile (can be unique or a function of semi-major axis).
        n_samples : int, optional
            The number of samples to generate (default is 10 million).
        """
        self.sigma_a = sigma_a
        self.ecc_profile = ecc_profile
        self.n_samples = n_samples
        self.a_samples = None
        self.r_samples = None
        self.e_samples = None
        self.f_samples = None
        self._use_jacobian = None

    def sample_a(
            self, 
            use_jacobian: bool = True
        ) -> np.ndarray:
        """
        Sample semi-major axis values from the surface density profile.
        This function uses batched and vectorised rejection sampling.

        Parameters
        ----------
        use_jacobian : bool, optional
            Whether to use the Jacobian of the surface density profile in the sampling process.
            If True, the sampling is weighted by the product of the surface density and semi-major axis: Sigma(a)*a
            If False, the sampling is uniform in the surface density: Sigma(a)
        
        Returns
        -------
        a_samples : np.ndarray
            Array of sampled semi-major axis values.
        """
        # Initialise variables
        max_iterations = 100
        accepted = np.empty(self.n_samples)
        filled = 0
        self._use_jacobian = use_jacobian
        # Test values of a (used to determine the maximum of the PDF)
        a_test = np.linspace(self.sigma_a.a_min, self.sigma_a.a_max, 1000)

        # Determine the maximum of the PDF
        if use_jacobian:
            target = self.sigma_a.get_values(a_test) * a_test
        else:
            target = self.sigma_a.get_values(a_test)
        M = 1.1*np.max(target)

        # Determine the batch size
        batch_size = int(self.n_samples)

        # Rejection sampling loop
        for _ in range(max_iterations):
            # Sample a batch of a values
            a_prop = np.random.uniform(self.sigma_a.a_min, self.sigma_a.a_max, batch_size)
            u = np.random.uniform(0, M, batch_size)
    
            # Compute the PDF
            if use_jacobian:
                p = self.sigma_a.get_values(a_prop) * a_prop
            else:
                p = self.sigma_a.get_values(a_prop)

            # Determine the number of accepted samples
            mask = u < p
            num_accepted = np.sum(mask)

            # Determine the number of samples to take
            remaining = self.n_samples - filled
            to_take = min(num_accepted, remaining)

            # Take the samples
            if to_take > 0:
                accepted[filled:filled + to_take] = a_prop[mask][:to_take]
                filled += to_take

            # Check if we have enough samples
            if filled >= self.n_samples:
                break

        # Check if we have enough samples
        if filled < self.n_samples:
            raise RuntimeError(f"Only accepted {filled} samples out of {self.n_samples} requested.")
        
        # Store the samples
        self.a_samples = accepted

        return accepted


    def sample_eccentricities(self, a_samples: np.ndarray) -> np.ndarray:
        """
        Sample eccentricities using proper rejection sampling 
        conditioned on each input semi-major axis a_i.

        Parameters
        ----------
        a_samples : np.ndarray
            Semi-major axis values. Each e_i will be drawn from Psi(e | a_i).

        Returns
        -------
        e_samples : np.ndarray
            Eccentricity values corresponding to each a_i.
        """
        if isinstance(self.ecc_profile, UniqueEccentricity):
            return self.ecc_profile.eccentricity(a_samples)

        profile = self.ecc_profile
        N = len(a_samples)
        e_samples = np.empty(N)
        e_grid = np.linspace(0, 1, 1000)

        for i, a in enumerate(a_samples):
            # Estimate maximum of Psi(e | a)
            psi_vals = profile.distribution_func(e_grid, np.full_like(e_grid, a))
            M = np.max(psi_vals) * 1.1  # safety margin

            # Rejection sampling loop for this a_i
            while True:
                e_prop = np.random.uniform(0, 1)
                u = np.random.uniform(0, M)
                psi_val = profile.distribution_func(np.array([e_prop]), np.array([a]))[0]
                if u < psi_val:
                    e_samples[i] = e_prop
                    break

        self.e_samples = e_samples
        return e_samples



    def sampler(
            self, 
            use_jacobian: bool = True,
            verbose: bool = True,
            return_samples: bool = True
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the Monte Carlo sampling of semi-major axis, eccentricities, 
        and true anomalies, and then compute the corresponding radial positions.

        This method orchestrates the entire sampling process, including:
        - Sampling semi-major axis values
        - Sampling (or calculating) eccentricities
        - Solving Kepler's equation for the eccentric anomaly
        - Computing radial positions

        Parameters
        ----------
        use_jacobian : bool, optional
            Whether to use the Jacobian of the surface density profile in the sampling process.
            If True, the sampling is weighted by the product of the surface density and semi-major axis: Sigma(a)*a
            If False, the sampling is uniform in the surface density: Sigma(a)
        verbose : bool, optional
            Whether to print progress messages.
        return_samples : bool, optional
            Whether to return the samples. If False, the samples are cached internally but not returned directly.
        
        Returns
        -------
        a_samples : np.ndarray
            Array of sampled semi-major axis values.
        r_samples : np.ndarray
            Array of radial positions computed from a, e, f.
        e_samples : np.ndarray
            Array of eccentricities corresponding to the given semi-major axes.
        f_samples : np.ndarray
            Array of true anomalies corresponding to the given semi-major axes and eccentricities.
        """
        self._use_jacobian = use_jacobian

        if verbose:
            print("Sampling semi-major axis...")

        # Sample the semi-major axis values
        a_samples = self.sample_a(use_jacobian=use_jacobian)
        
        if verbose:
            print("Sampling eccentricities...")

        # Sample the eccentricities
        mean_anomalies = np.random.uniform(0, 2*np.pi, size=self.n_samples)
        e_samples = self.sample_eccentricities(a_samples)

        if verbose:
            print("Solving Kepler's equation...")

        # Solve Kepler's equation for the eccentric anomaly 
        E_sol = kepler_solver(mean_anomalies, e_samples)
        cosf = (np.cos(E_sol) - e_samples) / (1 - e_samples * np.cos(E_sol))
        sinf = np.sqrt(1 - e_samples**2) * np.sin(E_sol) / (1 - e_samples * np.cos(E_sol))

        # Compute the true anomaly
        f_samples = np.arctan2(sinf, cosf)

        # Compute the radial positions
        r_samples = a_samples * (1.0 - e_samples**2) / (1.0 + e_samples * cosf)

        if verbose:
            print("Done.")

        # Store the samples
        self.f_samples = f_samples
        self.r_samples = r_samples

        if return_samples:
            return a_samples, r_samples, e_samples, f_samples

    def get_1d_histogram(
            self, 
            bins: int = 500, 
            scale: bool = True,
            verbose: bool = True,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 1D histogram of semi-major axis and radial positions.

        This method computes the 1D histogram of semi-major axis and radial positions,
        optionally scaling the histogram to match the true area under the surface density profile.

        Parameters  
        ----------
        bins : int, optional
            Number of bins for the histogram.
        scale : bool, optional
            Whether to scale the histogram to match the true area under the surface density profile.
        verbose : bool, optional
            Whether to print progress messages.
        
        Returns
        -------
        bin_centers_a : np.ndarray
            Bin centers for the semi-major axis histogram.
        hist_a : np.ndarray
            Histogram of semi-major axis values.
        bin_centers_r : np.ndarray
            Bin centers for the radial position histogram.
        hist_r : np.ndarray
            Histogram of radial positions.
        """
        # If the radial positions are not already cached with Jacobian = True, sample them
        if self._use_jacobian is True:
            if self.r_samples is None:
                a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=True, verbose=verbose)
            else:
                a_samples = self.a_samples
                r_samples = self.r_samples
        else:
            a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=True, verbose=verbose)
        
        # Compute the histogram of semi-major axis values   
        hist_a, bins_a = np.histogram(a_samples, bins=bins, density=True)  
        bin_centers_a = 0.5 * (bins_a[1:] + bins_a[:-1])

        # Compute the histogram of radial positions
        hist_r, bins_r = np.histogram(r_samples, bins=bins, density=True)
        bin_centers_r = 0.5 * (bins_r[1:] + bins_r[:-1])      

        # If scaling is requested, compute the scaling factor
        if scale:
            bin_widths = np.diff(bins_a)
            sigma_a_est = hist_a / bin_centers_a
            area_est = np.sum(sigma_a_est * bin_widths)
            true_area = self.sigma_a.compute_area()
            scaling = true_area / area_est
        else:
            scaling = 1.0
        
        # Return the histograms
        return bin_centers_a, hist_a * scaling / bin_centers_a, bin_centers_r, hist_r * scaling / bin_centers_r
            
    def plot_1d(
            self, 
            bins: int = 500, 
            save: bool = False, 
            filepath: str = None, 
            overlay: bool = False, 
            scale: bool = True, 
            asd = None, 
            x_lim: tuple[float, float] = None, 
            y_lim: tuple[float, float] = None):
        """
        Plot the 1D histogram of semi-major axis and radial positions.

        Parameters
        ----------
        bins : int, optional
            Number of bins for the histogram.
        save : bool, optional
            Whether to save the figure.
        filepath : str, optional
            Path to save the figure.
        overlay : bool, optional
            Whether to overlay the histogram with the analytic ASD.
        scale : bool, optional
            Whether to scale the histogram to match the true area under the surface density profile.
        asd : ASD, optional
            ASD object to use for the overlay.
        x_lim : tuple[float, float], optional
            Limits for the x-axis.
        y_lim : tuple[float, float], optional
            Limits for the y-axis.
        """
        bin_centers_a, hist_a, bin_centers_r, hist_r = self.get_1d_histogram(bins=bins, scale=scale, verbose=False)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the histograms
        ax.plot(bin_centers_a, hist_a, label="MC - $\\Sigma_a(a)$", color="red", linestyle="-")
        ax.plot(bin_centers_r, hist_r, label="MC - $\\Sigma_r(r)$", color="green", linestyle="-")

        # If overlay is requested, compute the analytic ASD
        if overlay:
            if asd is None:
                raise ValueError("`asd` must be provided if overlay=True.")
            else:
                r_vals, sigma_r_vals = asd.get_values()
                
            ax.plot(r_vals, sigma_r_vals, label="$\\bar{\\Sigma}(r)$", color="darkorange", linestyle="--")

            a_vals = np.linspace(min(r_vals), max(r_vals), 1000)
            sigma_a_analytic = self.sigma_a.get_values(a_vals)
            ax.plot(a_vals, sigma_a_analytic, label="$\\Sigma_a(a)$", color="blue", linestyle="--")

        # Finalize main plot
        ax.set_ylabel(r"$\Sigma_a(a), \bar{\Sigma}(r)$", fontsize=15)
        ax.set_xlabel(r"$a, r$", fontsize=15)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=15)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.grid()

        if save:
            if filepath is None:
                raise ValueError("`filepath` must be specified if save=True.")
            plt.savefig(filepath, dpi=300)
        else:
            plt.show()


    def get_cart_histogram(self, bins=500, varpi_func=None, verbose: bool = True):
        """
        Return a 2D histogram and edges in Cartesian (x, y) coordinates.

        Parameters
        ----------
        bins : int
            Number of bins along each axis for the 2D histogram.
        varpi_func : callable, optional
            Function of a_samples returning varpi(a).
        verbose : bool, optional
            Whether to print progress messages.

        Returns
        -------
        hist : ndarray
            2D histogram (bins x bins).
        x_edges : ndarray
            Bin edges along x.
        y_edges : ndarray
            Bin edges along y.
        """
        if self.r_samples is None:
            a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=False, verbose=verbose)
        else:
            a_samples = self.a_samples
            r_samples = self.r_samples
            e_samples = self.e_samples
            f_samples = self.f_samples

        varpi_samples = np.zeros_like(a_samples) if varpi_func is None else varpi_func(a_samples)
        theta_samples = f_samples + varpi_samples

        x_samples = r_samples * np.cos(theta_samples)
        y_samples = r_samples * np.sin(theta_samples)

        x_min, x_max = x_samples.min(), x_samples.max()
        y_min, y_max = y_samples.min(), y_samples.max()

        hist = fast_histogram.histogram2d(
            x_samples, y_samples,
            bins=[bins, bins],
            range=[[x_min, x_max], [y_min, y_max]]
        )
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)
        return hist, x_edges, y_edges

    def get_polar_histogram(self, bins=500, varpi_func=None, verbose: bool = True):
        """
        Return a 2D histogram on a Cartesian (r, φ) grid.

        Returns
        -------
        hist : ndarray
            2D histogram (shape: [phi_bins, r_bins]).
        r_edges : ndarray
            Bin edges in radius.
        phi_edges : ndarray
            Bin edges in azimuth (radians, from 0 to 2pi).
        verbose : bool, optional
            Whether to print progress messages.
        """
        if self.r_samples is None:
            a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=False, verbose=verbose)
        else:
            a_samples = self.a_samples
            r_samples = self.r_samples
            e_samples = self.e_samples
            f_samples = self.f_samples

        varpi_samples = np.zeros_like(a_samples) if varpi_func is None else varpi_func(a_samples)
        phi_samples = (f_samples + varpi_samples) % (2 * np.pi)

        r_min, r_max = r_samples.min(), r_samples.max()
        phi_min, phi_max = 0, 2 * np.pi

        hist = fast_histogram.histogram2d(
            r_samples, phi_samples,
            bins=[bins, bins],
            range=[[r_min, r_max], [phi_min, phi_max]]
        )

        r_edges = np.linspace(r_min, r_max, bins + 1)
        phi_edges = np.linspace(phi_min, phi_max, bins + 1)
        return hist.T, r_edges, phi_edges

    def plot_2d(self, varpi_func=None, bins=500, log=True, mode='cartesian', save=False, filepath=None):
        """
        Plot a 2D spatial histogram of particle positions in Cartesian (x, y) or polar-grid (r, φ) view.

        Parameters
        ----------
        varpi_func : callable, optional
            Function of a_samples returning varpi(a).
        bins : int 
            Number of bins.
        log : bool
            Whether to use logarithmic colormap normalization.
        mode : str
            Either 'cartesian' or 'polar'.
        save : bool
            Whether to save the figure.
        filepath : str or Path, optional
            Save path if save=True.
        """
        print(f"Generating 2D histogram in {mode} coordinates...")

        if mode == 'cartesian':
            hist, x_edges, y_edges = self.get_cart_histogram(bins=bins, varpi_func=varpi_func, verbose=False)

            plt.figure(figsize=(8, 8))
            norm = LogNorm() if log else None
            plt.pcolormesh(x_edges, y_edges, hist.T, cmap='magma', norm=norm)
            plt.colorbar(label='Counts per pixel')
            plt.xlabel("x [AU]")
            plt.ylabel("y [AU]")
            plt.axis("equal")

        elif mode == 'polar':
            hist, r_edges, phi_edges = self.get_polar_histogram(bins, varpi_func=varpi_func, verbose=False)

            plt.figure(figsize=(10, 5))
            norm = LogNorm() if log else None
            plt.pcolormesh(r_edges, phi_edges, hist, cmap='magma', norm=norm, shading='auto')
            plt.xlabel("r [AU]")
            plt.ylabel(r"$\phi$ [rad]")
            plt.colorbar(label='Counts per pixel')
            plt.ylim(0, 2 * np.pi)

        else:
            raise ValueError("mode must be 'cartesian' or 'polargrid'")

        if save:
            if filepath is None:
                raise ValueError("`filepath` must be specified if save=True.")
            plt.savefig(filepath, dpi=300)

        plt.show()