from dataclasses import dataclass
import pandas as pd
import numpy as np
import lumos.constants


def load_filter_dat(path):
    """
    Load a filter transmission profile exported from SVO.

    :param path: Path to the ``.dat`` file downloaded from SVO.
    :type path: str
    :returns: ``lam`` (wavelength grid in nm) and ``trans`` (transmission values as
        stored in the file).
    :rtype: tuple[np.ndarray, np.ndarray]

    note: Files may specify wavelength in Angstrom or nm; this loader infers the
    unit from the wavelength scale. Transmission values are not normalized here.
    """

    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    lam = df[0].values.astype(float)
    trans = df[1].values.astype(float)

    w = np.asarray(lam, float)

    lam = w / 10.0 if np.nanmedian(w) > 2000 else w

    return lam, trans


def load_solar_xlxs(path, normalize_to=lumos.constants.SUN_INTENSITY):
    """
    Load a solar spectrum from an Excel file.

    The file is expected to contain wavelength in micrometers in the first column
    and spectral irradiance in ``W/m^2/um`` in the second column. The returned
    spectrum is converted to nm and ``W/m^2/nm`` and sorted by wavelength.

    :param path: Path to the Excel file containing the solar spectrum.
    :type path: str
    :param normalize_to: If not ``None``, rescale the spectrum so its integral
        equals this value in ``W/m^2``.
    :type normalize_to: float or None, optional
    :returns: ``lam`` (wavelength grid in nm) and ``E_sun`` (solar spectral
        irradiance at TOA in ``W/m^2/nm``).
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If normalization is requested and the spectrum integral is
        not positive.
    """

    df = pd.read_excel(path)
    lam_um = df.iloc[:, 0].values.astype(float)
    E_sun_um = df.iloc[:, 1].values.astype(float)

    sun_lam_nm = np.asarray(lam_um, float) * 1000.0  # convert from um to nm
    # clean any bad values
    E_sun_nm = np.asarray(E_sun_um, float) / 1000.0
    # sort by wavelength
    order = np.argsort(sun_lam_nm)
    sun_lam_nm = sun_lam_nm[order]
    E_wm2_nm = E_sun_nm[order]

    if normalize_to is not None:
        total = float(np.trapz(E_wm2_nm, sun_lam_nm))
        if total <= 0.0:
            raise ValueError("Solar spectrum integral must be positive")
        E_wm2_nm = E_wm2_nm * float(normalize_to) / total

    return sun_lam_nm, E_wm2_nm


def as_fractional_throughput(T):
    """
    Convert throughput values to fractional units if they appear to be percentages.

    :param T: Throughput values expressed either as fractions in ``[0, 1]`` or
        percentages in ``[0, 100]``.
    :type T: array_like
    :return: Throughput as fractional values.
    :rtype: np.ndarray
    """

    T = np.asarray(T, float)
    return T / 100.0 if np.nanmax(T) > 1.5 else T


@dataclass(frozen=True)
class SpectralBandpass:
    """
    Container for a spectral bandpass and its matched solar spectrum.

    :param lam_nm: Wavelength grid in nanometers.
    :type lam_nm: np.ndarray
    :param transmission: Fractional bandpass transmission sampled on ``lam_nm``.
    :type transmission: np.ndarray
    :param solar_irradiance_Wm2: Solar spectral irradiance sampled on ``lam_nm`` in
        ``W/m^2/nm``.
    :type solar_irradiance_Wm2: np.ndarray
    """

    lam_nm: np.ndarray
    transmission: np.ndarray
    solar_irradiance_Wm2: np.ndarray

    @property
    def solar_weight(self):
        return self.transmission * self.solar_irradiance_Wm2

    @property
    def in_band_solar_wm2(self):
        return float(np.trapz(self.solar_weight, self.lam_nm))


def make_bandpass(
    filter_path=None,
    solar_path=None,
    *,
    lam_filter=None,
    T_filter=None,
    lam_sun=None,
    E_sun_Wm2_per_nm=None,
):
    """
    Construct a :class:`SpectralBandpass` from filter and solar spectrum data.

    Inputs may be supplied either as file paths or as arrays. The solar spectrum is
    interpolated onto the filter wavelength grid, and the filter throughput is
    converted to fractional units if needed.

    :param filter_path: Path to a filter profile file readable by
        :func:`load_filter_dat`.
    :type filter_path: str, optional
    :param solar_path: Path to a solar spectrum file readable by
        :func:`load_solar_xlxs`.
    :type solar_path: str, optional
    :param lam_filter: Filter wavelength grid in nm.
    :type lam_filter: array_like, optional
    :param T_filter: Filter transmission values.
    :type T_filter: array_like, optional
    :param lam_sun: Solar spectrum wavelength grid in nm.
    :type lam_sun: array_like, optional
    :param E_sun_Wm2_per_nm: Solar spectral irradiance in ``W/m^2/nm``.
    :type E_sun_Wm2_per_nm: array_like, optional
    :return: Bandpass object on the filter wavelength grid.
    :rtype: SpectralBandpass
    :raises ValueError: If filter data or solar spectrum data is missing.
    """

    if filter_path is not None:
        lam_filter, T_filter = load_filter_dat(filter_path)
    if solar_path is not None:
        lam_sun, E_sun_Wm2_per_nm = load_solar_xlxs(solar_path)

    if lam_filter is None or T_filter is None:
        raise ValueError("Filter data must be provided either via path or array")
    if lam_sun is None or E_sun_Wm2_per_nm is None:
        raise ValueError(
            "Solar spectrum data must be provided either via path or array"
        )
    lam_filter = np.asarray(lam_filter, float)
    T_filter = as_fractional_throughput(T_filter)
    E_on_band = interp_to(lam_sun, E_sun_Wm2_per_nm, lam_filter, fill=0.0)

    return SpectralBandpass(
        lam_nm=lam_filter,
        transmission=T_filter,
        solar_irradiance_Wm2=np.asarray(E_on_band, float),
    )


def integrate_band_response(response_lam, bandpass):
    """
    Integrate a spectral response over a bandpass weighted by the solar spectrum.

    The last axis of ``response_lam`` must match the wavelength grid of
    ``bandpass.lam_nm``.

    :param response_lam: Spectral response sampled on the bandpass wavelength grid.
    :type response_lam: array_like
    :param bandpass: Bandpass definition and solar weighting data.
    :type bandpass: SpectralBandpass
    :return: Band-integrated response.
    :rtype: float or np.ndarray
    :raises ValueError: If the last axis of ``response_lam`` does not match the
        bandpass wavelength grid.
    """

    response_lam = np.asarray(response_lam, float)
    if response_lam.shape[-1] != bandpass.lam_nm.shape[-1]:
        raise ValueError(
            "Last axis of response_lam must match bandpass wavelength grid"
        )
    return np.trapz(response_lam * bandpass.solar_weight, bandpass.lam_nm, axis=-1)


def interp_to(x_src, y_src, x_tgt, fill=0.0):
    """
    Linearly interpolate ``y(x)`` from a source grid onto a target grid.

    Values outside the source range are filled with ``fill``.

    :param x_src: Source x coordinates.
    :type x_src: array_like
    :param y_src: Source y values.
    :type y_src: array_like
    :param x_tgt: Target x coordinates.
    :type x_tgt: array_like
    :param fill: Fill value used outside the source range.
    :type fill: float, optional
    :return: Interpolated values on ``x_tgt``.
    :rtype: np.ndarray
    """
    return np.interp(x_tgt, x_src, y_src, left=fill, right=fill)


def bandpass_irradiance(bandpass, return_norm=False):
    """
    Compute the solar irradiance transmitted through a bandpass.

    :param bandpass: Bandpass definition and matched solar spectrum.
    :type bandpass: SpectralBandpass
    :param return_norm: If ``True``, also return normalization metadata including
        the integrated transmission and pivot wavelength.
    :type return_norm: bool, optional
    :return: Either the in-band solar irradiance alone, or a tuple
        ``(I_band, {"int_T": ..., "lambda_p": ...})`` when ``return_norm=True``.
    :rtype: float or tuple[float, dict[str, float]]
    """

    lam = bandpass.lam_nm
    T = bandpass.transmission
    I_band = float(np.trapz(T * bandpass.solar_irradiance_Wm2, lam))

    if not return_norm:
        return I_band

    int_T = float(np.trapz(T, lam))
    num_piv = float(np.trapz(T * lam, lam))
    den_piv = float(np.trapz(T / np.clip(lam, 1e-12, None), lam))
    lambda_p = np.sqrt(num_piv / den_piv) if den_piv > 0 else np.nan

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
    """
    Estimate directional-hemispherical reflectance for a BRDF by Monte Carlo integration.

    A fixed incident direction ``wi`` and surface normal ``n`` are used, and the
    BRDF is integrated over the outgoing hemisphere using cosine-weighted sampling.

    :param brdf_fn: BRDF callable following the Lumos BRDF interface.
    :type brdf_fn: callable
    :param wi: Incident direction in world coordinates.
    :type wi: array_like
    :param n: Surface normal in world coordinates.
    :type n: array_like
    :param lam: Wavelength or wavelength sample passed through to the BRDF.
    :type lam: float or array_like, optional
    :param nk: Optional additional BRDF parameter passed through to the BRDF.
    :type nk: any, optional
    :param n_samples: Number of Monte Carlo samples.
    :type n_samples: int, optional
    :param seed: Seed used to initialize the random number generator.
    :type seed: int, optional
    :return: Estimated directional-hemispherical reflectance.
    :rtype: float
    """
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
    """
    Create a wavelength-interpolation function for reflectance data.

    :param lam_table_nm: Wavelength samples in nanometers.
    :type lam_table_nm: array_like
    :param R_table: Reflectance values at ``lam_table_nm``.
    :type R_table: array_like
    :return: Function mapping wavelength in nm to interpolated reflectance.
    :rtype: callable
    """

    lam_table_nm = np.asarray(lam_table_nm, float)
    R_table = np.asarray(R_table, float)

    def R_of_lambda(lambda_nm):
        return np.interp(lambda_nm, lam_table_nm, R_table)

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

    :param base_brdf: Callable following the Lumos BRDF interface, used as the
        uncalibrated BRDF model.
    :type base_brdf: callable
    :param reflectance_df: Measured reflectance spectrum. The first column must be
        wavelength in nm and the second column reflectance in ``[0, 1]``.
    :type reflectance_df: pandas.DataFrame
    :param normalize_incident_deg: Incident zenith angle, in degrees, used when
        computing the model reference reflectance.
    :type normalize_incident_deg: float, optional
    :param rho_samples: Number of Monte Carlo samples used in the hemispherical
        reflectance estimate.
    :type rho_samples: int, optional
    :param rho_seed: Seed or generator control for Monte Carlo reproducibility.
    :type rho_seed: int | numpy.random.Generator | None, optional
    :return: BRDF callable with the same interface as ``base_brdf`` but scaled as a
        function of wavelength.
    :rtype: callable
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

        if lam is None:
            lam_arr = np.asarray([lam_tab[0]], float)
        else:
            lam_arr = np.asarray(lam, float)

        scale = C_of(lam_arr)
        lam_list = np.atleast_1d(lam_arr)

        # try vectorised base_brdf first
        try:
            vals = base_brdf(i_vec, n_vec, o_vec, lam=lam_arr, **kwargs)
        except TypeError:
            vals_list = []
            for L in lam_list:
                try:
                    vals_list.append(
                        base_brdf(i_vec, n_vec, o_vec, lam=float(L), **kwargs)
                    )
                except TypeError:
                    try:
                        vals_list.append(
                            base_brdf(i_vec, n_vec, o_vec, lam=L, **kwargs)
                        )
                    except TypeError:
                        # base_brdf doesn't accept wavelength at all
                        vals_no_lam = base_brdf(i_vec, n_vec, o_vec, **kwargs)
                        vals_list = [vals_no_lam] * lam_list.size
                        break
            vals = np.asarray(vals_list, float)

        out = scale * vals
        return float(out) if np.ndim(out) == 0 else out

    return BRDF_lambda


def band_to_abmag(I_band, bandpass, zeropoint_jy=3631.0, offset_mag=0.0):
    """
    Convert band-integrated irradiance to AB magnitude using the bandpass pivot wavelength.

    :param I_band: Band-integrated irradiance in ``W/m^2``.
    :type I_band: float or array_like
    :param bandpass: Bandpass definition used to compute the pivot wavelength.
    :type bandpass: SpectralBandpass
    :param zeropoint_jy: AB magnitude zero-point flux density in Jy.
    :type zeropoint_jy: float, optional
    :param offset_mag: Additive magnitude offset applied after conversion.
    :type offset_mag: float, optional
    :return: AB magnitude values with the same shape as ``I_band``. Non-positive
        irradiance values are returned as ``NaN``.
    :rtype: np.ndarray
    """
    _, meta = bandpass_irradiance(bandpass, return_norm=True)
    lambda_p_m = meta["lambda_p"] * 1e-9

    I_band = np.asarray(I_band, float)
    fnu = I_band * lambda_p_m / lumos.constants.SPEED_OF_LIGHT

    mag = np.full_like(fnu, np.nan, dtype=float)
    good = fnu > 0
    mag[good] = -2.5 * np.log10(fnu[good] / (zeropoint_jy * 1e-26)) + offset_mag
    return mag
