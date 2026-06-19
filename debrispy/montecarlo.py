# Import necessary modules
# ------------------------------------------------------------------------------------------------ #
import numpy as np
from numpy.fft import rfft2, irfft2
import scipy.signal as sig
import warnings

from .eccentricity import UniqueEccentricity

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import fast_histogram
from matplotlib.patches import Ellipse

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

# ------------------------------------------------------------------------------------------------ #


# Constant
TWO_SQRT2_LN2 = 2.0 * np.sqrt(2.0 * np.log(2.0))

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
    # Initial guess
    E = M + e * np.sin(M)

    # Iterate until the solution converges
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if np.max(np.abs(delta_E)) < tol:
            break
    return E

def _gaussian_2d_kernel(sx_bins: float, sy_bins: float, theta: float = 0.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel for Cartesian coordinates.
    """
    nx = int(np.ceil(4.0 * sx_bins))
    ny = int(np.ceil(4.0 * sy_bins))
    x = np.arange(-nx, nx + 1, dtype=float)
    y = np.arange(-ny, ny + 1, dtype=float)
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)

    c, s = np.cos(theta), np.sin(theta)
    Xp =  c * X + s * Y
    Yp = -s * X + c * Y

    K = np.exp(-0.5 * ((Xp / sx_bins)**2 + (Yp / sy_bins)**2))
    K /= K.sum()
    return K

def _mean_step(edges: np.ndarray) -> float:
    d = np.diff(edges)
    return float(np.mean(d))

@dataclass
class Histogram1D:
    """
    Container for a 1D histogram from MonteCarlo sampling.

    Attributes
    ----------
    edges : np.ndarray
        Bin edges, shape (N+1,).
    values : np.ndarray
        Histogram values after surface-density normalisation, shape (N,).
    kind : {'a', 'r'}
        'a' = semi-major axis Sigma_a(a), 'r' = radial (ASD).
    scaled : bool
        True if the histogram was scaled to match the true area under Sigma_a(a),
        False if no scaling applied.
    """
    edges: np.ndarray
    values: np.ndarray
    kind: Literal['a', 'r']
    scaled: bool

    @property
    def centers(self) -> np.ndarray:
        """Bin centers."""
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self) -> np.ndarray:
        """Bin widths."""
        return np.diff(self.edges)

    # def as_tuple_centers(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """(centers, values) for plotting."""
    #     return self.centers, self.values

    # def as_tuple_edges(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """(edges, values) for plotting or rebinning."""
    #     return self.edges, self.values  
    
    def get_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(edges, centers, centre values) for plotting."""
        return self.edges, self.centers, self.values


    def plot(self, ax=None, label=None, color=None, linestyle='-', **kwargs):
        """Plot the histogram.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        label : str, optional
            Label for the plot.
        color : str, optional
            Color for the plot.
        linestyle : str, optional
            Linestyle for the plot.
        **kwargs : dict, optional
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        edges, centers, values = self.get_values()
        ax.plot(centers, values, label=label, color=color, linestyle=linestyle, **kwargs)
        return ax

@dataclass
class Histogram2D:
    """
    Container for a 2D histogram.

    Attributes
    ----------
    x_edges : np.ndarray
        Bin edges along x (Cartesian) or r (polar), shape (Nx+1,).
    y_edges : np.ndarray
        Bin edges along y (Cartesian) or phi (polar), shape (Ny+1,).
    values : np.ndarray
        2D array of histogram values, shape (Ny, Nx), suitable for:
            plt.pcolormesh(x_edges, y_edges, values)
    mode : {'cartesian', 'polar'}
        Coordinate system.
    """
    x_edges: np.ndarray
    y_edges: np.ndarray
    values: np.ndarray
    mode: Literal['cartesian', 'polar']

    # --- Aliases/Helpers ---
    @property
    def r_edges(self) -> np.ndarray:
        """Alias for x_edges when mode='polar'."""
        if self.mode != 'polar':
            raise AttributeError("r_edges is only available when mode='polar'.")
        return self.x_edges

    @property
    def phi_edges(self) -> np.ndarray:
        """Alias for y_edges when mode='polar'."""
        if self.mode != 'polar':
            raise AttributeError("phi_edges is only available when mode='polar'.")
        return self.y_edges
    
    def get_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(edges, centers, centre values) for plotting."""
        return self.x_edges, self.y_edges, self.values

    def pad_to_limits(self,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    floor_value: float = 0.0) -> "Histogram2D":
        """
        Return a histogram whose edges cover (xlim, ylim). Outside the original
        extent we create new bins (same mean bin size) and fill them with
        `floor_value`. If limits lie inside, we crop.
        """
        if xlim is None: xlim = (self.x_edges[0], self.x_edges[-1])
        if ylim is None: ylim = (self.y_edges[0], self.y_edges[-1])

        # Ensure xlim, ylim are increasing
        x0, x1 = min(xlim), max(xlim)
        y0, y1 = min(ylim), max(ylim)

        # Mean bin sizes (works fine if bins are already uniform)
        dx = _mean_step(self.x_edges)
        dy = _mean_step(self.y_edges)

        # Compute how many bins to add on each side
        import math
        n_left  = max(0, math.ceil((self.x_edges[0] - x0) / dx))
        n_right = max(0, math.ceil((x1 - self.x_edges[-1]) / dx))
        n_bot   = max(0, math.ceil((self.y_edges[0] - y0) / dy))
        n_top   = max(0, math.ceil((y1 - self.y_edges[-1]) / dy))

        # Build new edges
        if n_left:
            x_extra_left = self.x_edges[0] - dx * np.arange(n_left, 0, -1)
            x_edges_new = np.concatenate([x_extra_left, self.x_edges])
        else:
            x_edges_new = self.x_edges.copy()

        if n_right:
            x_extra_right = self.x_edges[-1] + dx * np.arange(1, n_right + 1)
            x_edges_new = np.concatenate([x_edges_new, x_extra_right])

        if n_bot:
            y_extra_bot = self.y_edges[0] - dy * np.arange(n_bot, 0, -1)
            y_edges_new = np.concatenate([y_extra_bot, self.y_edges])
        else:
            y_edges_new = self.y_edges.copy()

        if n_top:
            y_extra_top = self.y_edges[-1] + dy * np.arange(1, n_top + 1)
            y_edges_new = np.concatenate([y_edges_new, y_extra_top])

        # Pad values with floor_value to match the new edges
        V = np.asarray(self.values, float)
        pad_y = (n_bot, n_top)
        pad_x = (n_left, n_right)
        if any(p > 0 for p in (*pad_y, *pad_x)):
            V = np.pad(V, (pad_y, pad_x), mode="constant", constant_values=floor_value)

        # Now crop to requested limits exactly (bin-aligned)
        # Figure out slice indices in new grid covering [x0,x1], [y0,y1]
        ix0 = max(0, int(np.floor((x0 - x_edges_new[0]) / dx)))
        ix1 = min(len(x_edges_new) - 1, int(np.ceil((x1 - x_edges_new[0]) / dx)))
        iy0 = max(0, int(np.floor((y0 - y_edges_new[0]) / dy)))
        iy1 = min(len(y_edges_new) - 1, int(np.ceil((y1 - y_edges_new[0]) / dy)))

        x_edges_out = x_edges_new[ix0:ix1+1]
        y_edges_out = y_edges_new[iy0:iy1+1]
        V_out = V[iy0:iy1, ix0:ix1]

        return Histogram2D(x_edges=x_edges_out, y_edges=y_edges_out, values=V_out, mode=self.mode)

    def convolve_gaussian(self, *,
                        fwhm_x: Optional[float] = None,
                        fwhm_y: Optional[float] = None,
                        sigma_x: Optional[float] = None,
                        sigma_y: Optional[float] = None,
                        theta: float = 0.0,
                        pad: float = 5.0) -> "Histogram2D":
        """
        Convolve with a rotated Gaussian PSF.

        Parameters
        ----------
        fwhm_x, fwhm_y : float, optional
            FWHM in *axis units*
            If only fwhm_x is given, use circular PSF (fwhm_y = fwhm_x).
        sigma_x, sigma_y : float, optional
            Sigma in *axis units*
            Mutually exclusive with FWHM. If only sigma_x is given, use circular PSF.
        theta : float
            Rotation angle (radians, CCW).
        pad : float
            Padding margin to retain PSF wings.

        Returns
        -------
        Histogram2D
            Convolved histogram.
        """
        if self.mode != "cartesian":
            raise ValueError("Gaussian PSF convolution only supported for cartesian histograms.")

        # --- Mutually exclusive & defaults ---
        has_fwhm = (fwhm_x is not None) or (fwhm_y is not None)
        has_sigma = (sigma_x is not None) or (sigma_y is not None)
        if has_fwhm and has_sigma:
            raise ValueError("Provide either FWHM or sigma, not both.")
        if not has_fwhm and not has_sigma:
            raise ValueError("Provide at least one of FWHM or sigma.")
        if fwhm_y is None and fwhm_x is not None:
            fwhm_y = fwhm_x
        if sigma_y is None and sigma_x is not None:
            sigma_y = sigma_x

        # --- Bin scales ---
        dx = float(np.mean(np.diff(self.x_edges)))
        dy = float(np.mean(np.diff(self.y_edges)))

        if has_fwhm:
            sx_bins = (fwhm_x / TWO_SQRT2_LN2) / dx
            sy_bins = (fwhm_y / TWO_SQRT2_LN2) / dy
        else:
            sx_bins = sigma_x / dx
            sy_bins = sigma_y / dy

        # Build Gaussian kernel in bin units
        K = _gaussian_2d_kernel(sx_bins, sy_bins, theta)

        V = np.ascontiguousarray(self.values, dtype=float)
        Vy, Vx = V.shape

        # How many pixels of padding to capture sides
        pad_x = int(np.ceil(pad * sx_bins))
        pad_y = int(np.ceil(pad * sy_bins))

        if pad_x or pad_y:
            Vpad = np.pad(V, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0.0)
            # Expand edges accordingly (preserve axis units)
            x_edges_out = np.r_[self.x_edges[0] - np.arange(pad_x, 0, -1)*dx,
                                self.x_edges,
                                self.x_edges[-1] + np.arange(1, pad_x+1)*dx]
            y_edges_out = np.r_[self.y_edges[0] - np.arange(pad_y, 0, -1)*dy,
                                self.y_edges,
                                self.y_edges[-1] + np.arange(1, pad_y+1)*dy]
        else:
            Vpad = V
            x_edges_out = self.x_edges
            y_edges_out = self.y_edges

        # Convolve via FFT
        try:
            Vconv = sig.fftconvolve(Vpad, K, mode="same")
        except Exception:
            Ky, Kx = K.shape
            Fy = int(2**np.ceil(np.log2(Vpad.shape[0] + Ky - 1)))
            Fx = int(2**np.ceil(np.log2(Vpad.shape[1] + Kx - 1)))
            A = np.zeros((Fy, Fx), dtype=float); A[:Vpad.shape[0], :Vpad.shape[1]] = Vpad
            B = np.zeros((Fy, Fx), dtype=float); B[:Ky, :Kx] = K
            Vfull = irfft2(rfft2(A) * rfft2(B), s=(Fy, Fx))
            Vconv = Vfull[:Vpad.shape[0], :Vpad.shape[1]]
        
        Vconv[np.abs(Vconv) < 1e-10] = 0.0

        H_out = Histogram2D(x_edges=x_edges_out, y_edges=y_edges_out, values=Vconv, mode='cartesian')

        # Normalise the histogram to match the true area under the curve
        H_out._psf_info = {
            "sigma_x": sx_bins * dx,
            "sigma_y": sy_bins * dy,
            "fwhm_x": sx_bins * dx * TWO_SQRT2_LN2,
            "fwhm_y": sy_bins * dy * TWO_SQRT2_LN2,
            "theta": theta,
        }
        return H_out


    def plot(self, *,
            log: bool = False,
            cmap: str = "magma",
            shading: str = "auto",
            vmin=None, vmax=None,
            floor_threshold=None, floor_value=None,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            ax=None, colorbar: bool = True, cbar_label: str = "Counts per pixel",
            show_psf: bool = False,
            psf_scale: float = 1.0,
            psf_loc: Tuple[float, float] = (0.12, 0.12),
            psf_facecolor: str = "white",
            psf_edgecolor: str = "black",
            psf_alpha: float = 0.9,
            save: bool = False, filepath: str = None):
        """
        Plot with optional xlim/ylim. Regions outside the histogram are
        binned with same bin size and filled with `floor_value` (default=0).

        Parameters
        ----------
        log : bool, optional
            If True, use a logarithmic scale.
        cmap : str, optional
            Colormap to use.
        shading : str, optional
            Shading to use.
        vmin : float, optional
            Minimum value to use for the colorbar.
        vmax : float, optional
            Maximum value to use for the colorbar.
        floor_threshold : float, optional
            Threshold value to use for flooring.
        floor_value : float, optional
            Value to use for flooring.
        xlim : tuple, optional
            x-axis limits.
        ylim : tuple, optional
            y-axis limits.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        colorbar : bool, optional
            If True, show the colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        show_psf: bool, optional
            If True, show the PSF.
        psf_scale: float, optional
            Scale factor for the PSF.
        psf_loc: tuple, optional
            Location of the PSF.
        psf_facecolor: str, optional
            Facecolor of the PSF.
        psf_edgecolor: str, optional
            Edgecolor of the PSF.
        psf_alpha: float, optional
            Alpha of the PSF.
        save : bool, optional
            If True, save the figure.
        filepath : str, optional
            Filepath to save the figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object with the plot.
        """
        # If limits extend beyond, pad out with a floor
        pad_floor = floor_value if floor_value is not None else (floor_threshold if floor_threshold is not None else 0.0)
        H = self if (xlim is None and ylim is None) else self.pad_to_limits(xlim, ylim, floor_value=pad_floor)

        if ax is None:
            fig_w = 12 if H.mode == 'cartesian' else 10
            fig_h = 8 if H.mode == 'cartesian' else 5
            _, ax = plt.subplots(figsize=(fig_w, fig_h))

        data = np.array(H.values, copy=True)

        # Apply flooring inside the plotted region
        if floor_threshold is not None:
            if floor_value is None:
                floor_value = floor_threshold
            data[data < floor_threshold] = floor_value

        # Norm / scaling
        if log:
            vmin_eff = vmin if (vmin is not None) else np.nanmax([np.nanmin(data[data > 0]), 1e-300])
            norm = LogNorm(vmin=vmin_eff, vmax=vmax if vmax is not None else np.nanmax(data))
            pcm = ax.pcolormesh(H.x_edges, H.y_edges, data, cmap=cmap, norm=norm, shading=shading)
        else:
            pcm = ax.pcolormesh(H.x_edges, H.y_edges, data, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)

        # Labels/aspect
        if H.mode == 'cartesian':
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
        else:
            ax.set_xlabel("r [AU]"); ax.set_ylabel(r"$\phi$ [rad]")
            ax.set_ylim(0, 2*np.pi)
        
        # Draw PSF marker if present
        if show_psf and H.mode == "cartesian" and hasattr(H, "_psf_info"):
            psf = H._psf_info

            # Use FWHM to draw the beam/PSF marker
            width = psf_scale * psf["fwhm_x"]
            height = psf_scale * psf["fwhm_y"]
            angle_deg = np.degrees(psf["theta"])

            # Position in axes-fraction coordinates, converted to data coordinates
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            xc = x0 + psf_loc[0] * (x1 - x0)
            yc = y0 + psf_loc[1] * (y1 - y0)

            beam = Ellipse(
                (xc, yc),
                width=width,
                height=height,
                angle=angle_deg,
                facecolor=psf_facecolor,
                edgecolor=psf_edgecolor,
                alpha=psf_alpha,
                lw=1.0,
                zorder=10
            )
            ax.add_patch(beam)

        if colorbar:
            plt.colorbar(pcm, ax=ax, label=cbar_label)

        if save:
            plt.savefig(filepath, dpi = 300, bbox_inches = "tight")

        return ax

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
            verbose: bool = True
        ):
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
        histA : Histogram1D
            1D Histogram object of semi-major axis values.
        histR : Histogram1D
            1D Histogram object of radial positions.
        """
        # Ensure we have samples with Jacobian=True (physical)
        if self._use_jacobian is True:
            if self.r_samples is None:
                a_samples, r_samples, _, _ = self.sampler(use_jacobian=True, verbose=verbose)
            else:
                a_samples = self.a_samples
                r_samples = self.r_samples
        else:
            if self._use_jacobian is not True:
                warnings.warn("For 1D surface-density histograms, forcing use_jacobian=True.", RuntimeWarning)
            a_samples, r_samples, _, _ = self.sampler(use_jacobian=True, verbose=verbose)

        # Histograms
        pdf_a, edges_a = np.histogram(a_samples, bins=bins, density=True)
        pdf_r, edges_r = np.histogram(r_samples, bins=bins, density=True)

        # Bin centers
        centers_a = 0.5 * (edges_a[1:] + edges_a[:-1])
        centers_r = 0.5 * (edges_r[1:] + edges_r[:-1])

        eps = 0.0
        if np.any(centers_a == 0) or np.any(centers_r == 0):
            eps = np.finfo(float).eps

        # Surface-density normalisation: divide by radius
        vals_a = pdf_a / (centers_a + eps)
        vals_r = pdf_r / (centers_r + eps)

        if scale:
            bin_widths_a = np.diff(edges_a)
            area_est = np.sum(vals_a * bin_widths_a)
            true_area = self.sigma_a.compute_area()
            if area_est > 0:
                scaling = true_area / area_est
                vals_a *= scaling
                vals_r *= scaling
            scaled_flag = True
        else:
            scaled_flag = False

        histA = Histogram1D(edges=edges_a, values=vals_a, kind='a', scaled=scaled_flag)
        histR = Histogram1D(edges=edges_r, values=vals_r, kind='r', scaled=scaled_flag)
        return histA, histR

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
        histA, histR = self.get_1d_histogram(bins=bins, scale=scale, verbose=False)

        # Plot the histograms
        fig, ax = plt.subplots(figsize=(8, 6))
        histA.plot(ax=ax, label=r"MC - $\Sigma_a(a)$", color="red")
        histR.plot(ax=ax, label=r"MC - $\Sigma_r(r)$", color="green")


        # If overlay is requested, compute the analytic ASD
        if overlay:
            if asd is None:
                raise ValueError("`asd` must be provided if overlay=True.")
            
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
        if x_lim: ax.set_xlim(x_lim)
        if y_lim: ax.set_ylim(y_lim)
        ax.grid()

        if save:
            if filepath is None:
                raise ValueError("`filepath` must be specified if save=True.")
            plt.savefig(filepath, dpi=300)
        else:
            plt.show()

    def get_cart_histogram(self, bins=500, varpi_func=None, verbose: bool = True, *, surface_density: bool = True):
        """
        Return a 2D histogram and edges in Cartesian (x, y) coordinates.

        Returns
        -------
        hist_cart : Histogram2D
            2D Histogram object in Cartesian coordinates (values shape: Ny x Nx).
        """
        if self.r_samples is None:
            a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=True, verbose=verbose)
        else:
            a_samples = self.a_samples
            r_samples = self.r_samples
            f_samples = self.f_samples

        varpi_samples = np.zeros_like(a_samples) if varpi_func is None else varpi_func(a_samples)
        theta_samples = f_samples + varpi_samples

        x_samples = r_samples * np.cos(theta_samples)
        y_samples = r_samples * np.sin(theta_samples)

        # Build edges first, then use their span as histogram range
        if isinstance(bins, int):
            x_min, x_max = x_samples.min(), x_samples.max()
            y_min, y_max = y_samples.min(), y_samples.max()
            x_edges = np.linspace(x_min, x_max, bins + 1)
            y_edges = np.linspace(y_min, y_max, bins + 1)
            H = fast_histogram.histogram2d(
                x_samples, y_samples,
                bins=[bins, bins],
                range=[[x_edges[0], x_edges[-1]], [y_edges[0], y_edges[-1]]]
            )
        else:
            raise ValueError("`bins` must be an integer")
        
        H2 = H.T
        if surface_density:
            dx = np.diff(x_edges); dy = np.diff(y_edges)
            if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
                raise ValueError("surface_density=True requires uniform Cartesian bins.")
            H2 = H2 / (dx[0] * dy[0])

        hist_cart = Histogram2D(x_edges=x_edges, y_edges=y_edges, values=H2, mode='cartesian')
        return hist_cart

    def get_polar_histogram(self, bins=500, varpi_func=None, verbose: bool = True, *, surface_density: bool = True):
        """
        Return a 2D histogram on a polar (r, phi) grid.

        Returns
        -------
        hist_polar : Histogram2D
            2D Histogram object in polar coordinates.
        """
        if self.r_samples is None:
            a_samples, r_samples, e_samples, f_samples = self.sampler(use_jacobian=True, verbose=verbose)
        else:
            a_samples = self.a_samples
            r_samples = self.r_samples
            f_samples = self.f_samples

        varpi_samples = np.zeros_like(a_samples) if varpi_func is None else varpi_func(a_samples)
        phi_samples = (f_samples + varpi_samples) % (2.0 * np.pi)

        if isinstance(bins, int):
            r_min, r_max = r_samples.min(), r_samples.max()
            r_edges = np.linspace(r_min, r_max, bins + 1)
            phi_edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
            H_rphi = fast_histogram.histogram2d(
                r_samples, phi_samples,
                bins=[bins, bins],
                range=[[r_edges[0], r_edges[-1]], [phi_edges[0], phi_edges[-1]]]
            )
        else:
         raise ValueError("`bins` must be an integer") 
        
        H2 = H_rphi.T

        if surface_density:
            r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])   # (Nr,)
            dr        = np.diff(r_edges)                     # (Nr,)
            dphi      = np.diff(phi_edges)                   # (Nphi,)
            area = (r_centers[None, :] * dr[None, :] * dphi[:, None])  # (Nphi, Nr)
            H2 = H2 / area

        hist_polar = Histogram2D(x_edges=r_edges, y_edges=phi_edges, values=H2, mode='polar')
        return hist_polar

    def plot_2d(self, varpi_func=None, bins=500, log=True, mode='cartesian',
                save=False, filepath=None, surface_density=True, **plot_kwargs):
        """
        Thin wrapper around Histogram2D.plot().
        Extra kwargs are forwarded to Histogram2D.plot (e.g., cmap, shading, vmin, vmax, colorbar=False).
        """
        print(f"Generating 2D histogram in {mode} coordinates...")

        if mode == 'cartesian':
            hist = self.get_cart_histogram(bins=bins, varpi_func=varpi_func, verbose=False, surface_density=surface_density)
        elif mode == 'polar':
            hist = self.get_polar_histogram(bins=bins, varpi_func=varpi_func, verbose=False, surface_density=surface_density)
        else:
            raise ValueError("mode must be 'cartesian' or 'polar'")

        if surface_density:
            plot_kwargs['cbar_label'] = "Surface Density (Unnormalised)"
        else:
            plot_kwargs['cbar_label'] = "Counts per Bin"

        ax = hist.plot(log=log, **plot_kwargs)

        if save:
            if filepath is None:
                raise ValueError("`filepath` must be specified if save=True.")
            plt.savefig(filepath, dpi=300)
        else:
            plt.show()