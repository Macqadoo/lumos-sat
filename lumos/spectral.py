import pandas as pd
import numpy as np


def load_filter_dat(path):
    """Load a filter profile exported from SVO.

    :param path: Path to the ``.dat`` file downloaded from SVO.
    :type path: str
    :returns: ``lam`` (wavelength grid in nm) and ``trans`` (normalized transmission 0–1).
    :rtype: tuple[np.ndarray, np.ndarray]

    note: Files may specify Angstrom or nm; the loader infers the unit.
    """
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    lam = df[0].values.astype(float)
    trans = df[1].values.astype(float)

    w = np.asarray(lam, float)

    lam = w / 10.0 if np.nanmedian(w) > 2000 else w

    return lam, trans


def interp_to(x_src, y_src, x_tgt, fill=0.0):
    # Linear interpolate y(x) from (x_src,y_src) onto x_tgt, fill outside range.
    return np.interp(x_tgt, x_src, y_src, left=fill, right=fill)


def bandpass_irradiance(
    lam_nm_filter, T_filter, lam_nm_sun, E_sun_Wm2_per_nm, return_norm=False
):
    """
    Docstring for bandpass_irradiance
    Compute in band solar irradiance for a surface with spectral reflectance rho(lambda)

    :param lam_nm_filter: array, wavelength grid of filter transmission curve nm
    :param T_filter: array, Filter transmission 0-1
    :param lam_nm_sun: array, wavelength grid of solar spectrum nm
    :param E_sun_Wm2_per_nm: array, Solar spectral irradiance at TOA on lam_nm_sun W/m^2/nm
    :param return_norm: If true, return integral T dlamb and effective wavelength
    :return: I_band: in W/m^-2
    :rtype: float

    """
    lam_nm_filter = np.asarray(lam_nm_filter, float)
    T_filter = np.asarray(T_filter, float)
    lam_nm_sun = np.asarray(lam_nm_sun, float)
    E_sun_Wm2_per_nm = np.asarray(E_sun_Wm2_per_nm, float)

    # ensure filter throughput is fraction
    if np.nanmax(T_filter) > 1.5:
        T_filter = T_filter / 100.0

    # interp solar spectrum onto filter grid
    E_on_f = interp_to(lam_nm_sun, E_sun_Wm2_per_nm, lam_nm_filter, fill=0.0)

    # BRDF already includes spectral reflectance → do NOT apply here
    rho_on_f = np.ones_like(lam_nm_filter)

    # In band irradiance (nm integration gives W m^-2)
    # I_band = integral of E_sun(lam) * T(lam) * rho(lam) dlam
    integrand = E_on_f * T_filter * rho_on_f
    I_band = float(
        np.trapz(integrand, lam_nm_filter)
    )  # lam_f in nm, E in W m^-2 nm^-1 → W m^-2

    if not return_norm:
        return I_band

    int_T = float(np.trapz(T_filter, lam_nm_filter))
    num_piv = float(np.trapz(T_filter * lam_nm_filter, lam_nm_filter))
    den_piv = float(
        np.trapz(T_filter / np.clip(lam_nm_filter, 1e-12, None), lam_nm_filter)
    )
    lambda_p = (
        np.sqrt(num_piv / den_piv) if den_piv > 0 else np.nan
    )  # pivot lambda for AB mag conv

    return I_band, {"int_T": int_T, "lambda_p": lambda_p}


# sampling & frames
def sample_cosine_hemisphere(n_samples, rng=None):
    """Samples points uniformly on a hemisphere with a cosine-weighted distribution.

    This function generates `n_samples` random points on the upper hemisphere (z >= 0)
    such that the probability density is proportional to the cosine of the angle with
    the surface normal (the +Z axis). This is commonly used in rendering and Monte Carlo
    integration for simulating diffuse reflection.

    :param n_samples: Number of samples to generate.
    :type n_samples: int
    :param rng: Optional NumPy random number generator instance. If None, a new default generator is created.
    :type rng: numpy.random.Generator, optional
    :return: Tuple of three NumPy arrays (x, y, z) containing the sampled coordinates.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng() if rng is None else rng
    u1 = rng.random(n_samples)
    u2 = rng.random(n_samples)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1.0 - u1)
    return x, y, z  # local (+Z = surface normal)


def _frame_from_normal(n):
    """makes local tangent/bitangent plane that matches n

    takes an arbitrary surface normal n and builds an orthonormal basis aligned with that normal
    so sampled directions can be rotated in the global coordinate system

    :param n: arbitrary surface normal
    :type n: float
    :return: t, b, n (tangent, bitangent, normal)
    """

    nx, ny, nz = n
    n = np.array(n, float)
    n /= np.linalg.norm(n) + 1e-16
    up = np.array([0, 0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0, 0])
    t = np.cross(up, n)
    t /= np.linalg.norm(t) + 1e-16
    b = np.cross(n, t)
    return t, b, n


def local_to_world(ox, oy, oz, n):
    """Convert local-frame direction cosines into the global frame.

    :param ox: Local-frame x components sampled around the surface normal
    :type ox: np.ndarray or float
    :param oy: Local-frame y components sampled around the surface normal.
    :type oy: np.ndarray or float
    :param oz: Local-frame z components sampled around the surface normal
    :type oz: np.ndarray or float
    :param n: Surface normal expressed in global coordinates
    :type n: tuple[float, float, float]
    :return: global-frame components (X, Y, Z) corresponding to the inputs
    :rtype: tuple[np.ndarry, np.ndarray, np.ndarray]
    """

    t, b, nn = _frame_from_normal(n)
    # stack and multiply (3x3) but keep component arrays for lumos-style BRDF
    X = t[0] * ox + b[0] * oy + nn[0] * oz
    Y = t[1] * ox + b[1] * oy + nn[1] * oz
    Z = t[2] * ox + b[2] * oy + nn[2] * oz
    return X, Y, Z


# hemispherical reflectance with lumos-style BRDF
def hemispherical_reflectance(
    brdf_fn, wi, n, lam=None, nk=None, n_samples=4096, seed=42
):
    # cosine-weighted sampling in local frame
    ox_l, oy_l, oz_l = sample_cosine_hemisphere(n_samples, np.random.default_rng(seed))

    # world frame (n is z-up -> this just passes through, but keeps things general)
    ox, oy, oz = local_to_world(ox_l, oy_l, oz_l, n)

    try:
        vals = brdf_fn(wi, n, (ox, oy, oz), lam=lam, nk=nk)  # keyword form
    except TypeError:
        try:
            vals = brdf_fn(wi, n, (ox, oy, oz), lam)  # positional form
        except TypeError:
            vals = brdf_fn(wi, n, (ox, oy, oz))  # no wavelength

    # Cosine-weighted estimator: R = π * mean(f)
    return float(np.pi * np.mean(vals))


def interp_R(lam_table_nm, R_table):
    lam_table_nm = np.asarray(lam_table_nm, float)
    R_table = np.asarray(R_table, float)

    def R_of_lambda(lambda_nm):
        return float(np.interp(lambda_nm, lam_table_nm, R_table))

    return R_of_lambda


def diffuse_spectral_energy_function(
    base_brdf,
    reflectance_df,
    normalize_incident_deg: float = 8.0,
    rho_samples: int = 30000,
    rho_seed=None,
):
    """Construct a BRDF wrapper that matches measured reflected energy per wavelength and returns a wavelength aware BRDF.

    This routine evaluates 'base_brdf' at each wavelength in reflectance_df, computes the
    hemispherical-directional reflectance via  Monte Carlo integration, and derives a
    wavelength dependent scale factor so the model reproduces the measured spectrum.
    The returned callable mirrors the lumos BRDF interface but
    applies the calibrated scale factor before emitting radiance values.

    note: this function assumes the measured spectrum already represents diffuse hemispherical reflectance,
    so it simply scales the base BRDF per wavelength without modelling specular lobes or coherence effects. It treats surface roughness as
    wavelength invariant (no micro-geometry adjustments), ignores any phase-dependent scattering or polarisation, and assumes a fixed incident
    zenith angle for calibration.

    :param base_brdf: Callable following the lumos BRDF interface, used as the uncalibrated spectral BRDF to be scaled
    :type base_brdf: Callable[[tuple, tuple, tuple], np.ndarray]
    :param reflectance_df: Measured hemispherical-directional reflectance spectrum.
    First column should be wavelength in nm, second colour is reflectance 0-1.
    :type reflectance_df: pandas.Dataframe
    :param normalize_incident_deg: incident zenith angle (degrees) used when computing the model's reference hemispherical reflectance, defaults to 8.0
    :type normalize_incident_deg: float, optional
    :param rho_samples: Number of Monte Carlo samples used to estimate the hemispherical reflectance of base BRDF at each wavelength, defaults to 30000
    :type rho_samples: int, optional
    :param rho_seed: Seed or generator passed to the cosine-hemisphere sampler, controls reproducibility of Monte Carlo integration, defaults to None
    :type rho_seed: int | numpy.random.Generator | None, optional
    :return: A callable BRDF identical to base_brdf but rescaled per wavelength to match measured spectrum
    :rtype: Callable
    """
    lam_tab = reflectance_df.iloc[:, 0].to_numpy(float)
    R_meas = reflectance_df.iloc[:, 1].to_numpy(float)

    # Per-λ model hemispherical reflectance
    th = np.deg2rad(normalize_incident_deg)
    wi = (np.sin(th), 0.0, np.cos(th))
    n = (0.0, 0.0, 1.0)

    R_model = np.array(
        [
            hemispherical_reflectance(
                base_brdf,
                wi,
                n,
                lam=L,
                n_samples=rho_samples,
                seed=rho_seed if rho_seed is not None else 42,
            )
            for L in lam_tab
        ],
        dtype=float,
    )

    eps = 1e-12
    C = R_meas / (R_model + eps)  # IMPORTANT: measured / model
    C_of = interp_R(lam_tab, C)  # smooth if you like before interp

    def BRDF_lambda(i_vec, n_vec, o_vec, *args, **kwargs):
        lam = kwargs.pop("lam", None)
        if lam is None and len(args) > 0:
            lam = args[0]
        lam_eff = lam_tab[0] if lam is None else float(lam)

        scale = float(C_of(lam_eff))

        try:
            vals = base_brdf(i_vec, n_vec, o_vec, lam=lam_eff, **kwargs)
        except TypeError:
            vals = base_brdf(i_vec, n_vec, o_vec, lam_eff, **kwargs)

        return scale * vals

    return BRDF_lambda
