"""
Local Vertical, Local Horizontal (LVLH) brightness calculator for Lumos - attempting to refactor coordinate system
"""

import numpy as np
import lumos.constants
import lumos.conversions
import lumos.geometry
import astropy.coordinates
import lumos.attitude
import lumos.spectral


def unit(v, eps=1e-12):
    """
    Return `v` normalized to unit length.

    :param v: Input vector.
    :type v: array_like
    :param eps: Minimum norm treated as non-zero. Vectors with norm less than or equal
        to this value are returned unchanged.
    :type eps: float, optional
    :return: Normalized vector, or the original vector if its norm is too small to
        normalize safely.
    :rtype: numpy.ndarray
    """

    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def build_body_basis_from_rv(r_km, v_km_s):
    """
    Constructs the body-fixed basis vectors in Earth-Centered Inertial (ECI) coordinates from position and velocity vectors.

    The body-fixed frame convention is:
        +Z axis: Nadir direction (points from spacecraft to Earth's center)
        +X axis: Along-track direction (velocity projected onto local horizontal plane)
        +Y axis: Completes right-handed system (cross product of Z and X)

    :param r_km: Position vector in ECI coordinates (kilometers).
    :type r_km: array_like
    :param v_km_s: Velocity vector in ECI coordinates (kilometers per second).
    :type v_km_s: array_like

    :returns:
    B : ndarray, shape (3, 3)
        Matrix whose columns are the body-fixed axes [Xb, Yb, Zb] expressed in ECI coordinates.
    """

    r = np.asarray(r_km, float)
    v = np.asarray(v_km_s, float)
    Rhat = unit(r)
    Zb = -Rhat
    v_perp = v - Zb * np.dot(v, Zb)
    Xb = unit(v_perp)
    Yb = unit(np.cross(Zb, Xb))  # completes RHS
    Xb = np.cross(Yb, Zb)
    B = np.column_stack([Xb, Yb, Zb])  # columns are body axes in ECI
    return B


def eci_to_body(B, vec_eci):
    """Project an ECI vector into body coords: v_body = B^T v_eci"""
    return B.T @ np.asarray(vec_eci, float).reshape(
        3,
    )


def get_body_vectors_from_state(
    r_sat_km, v_sat_km_s, r_obs_km, r_sun_km, *, earth_radius_km=6371.0
):
    """
    Core geometry helper from raw state vectors

    Inputs are ECI vectors in km / km/s
    Returns body-frame sun/observer unit vectors plus geometry scalars

    :param r_sat_km: Satellite position vector in ECI coordinates (kilometers).
    :type r_sat_km: array_like
    :param v_sat_km_s: Satellite velocity vector in ECI coordinates (kilometers per second).
    :type v_sat_km_s: array_like
    :param r_obs_km: Observer position vector in ECI coordinates (kilometers).
    :type r_obs_km: array_like
    :param r_sun_km: Sun position vector in ECI coordinates (kilometers).
    :type r_sun_km: array_like
    :param earth_radius_km: Earth radius in kilometers, defaults to 6371.0
    :type earth_radius_km: float, optional
    """
    r_sat_km = np.asarray(r_sat_km, float).reshape(3)
    v_sat_km_s = np.asarray(v_sat_km_s, float).reshape(3)
    r_obs_km = np.asarray(r_obs_km, float).reshape(3)
    r_sun_km = np.asarray(r_sun_km, float).reshape(3)

    sat_to_obs_km = r_obs_km - r_sat_km
    sat_to_sun_km = r_sun_km - r_sat_km

    dist_sat_to_obs = float(np.linalg.norm(sat_to_obs_km) * 1000.0)

    B = build_body_basis_from_rv(r_sat_km, v_sat_km_s)
    obs_vec_body = unit(eci_to_body(B, sat_to_obs_km))
    sun_vec_body = unit(eci_to_body(B, sat_to_sun_km))

    sat_height_m = float((np.linalg.norm(r_sat_km) - earth_radius_km) * 1000.0)

    cos_theta_zenith = np.clip(-sun_vec_body[2], -1.0, 1.0)
    angle_past_terminator = float(np.arccos(cos_theta_zenith))

    return (
        sun_vec_body,
        obs_vec_body,
        angle_past_terminator,
        dist_sat_to_obs,
        sat_height_m,
    )


def get_earthshine_panels(sat_z, angle_past_terminator, density):
    """
    Creates a mesh of pixels on Earth's surface which are visible to the satellite and illuminated
    by the sun.

    :param sat_z: The height of the satellite above the center of Earth (meters)
    :type sat_z: float
    :param angle_past_terminator: The angle of the satellite past the terminator (radians)
    :type angle_past_terminator: float
    :param density: The density of the pixels. Grid will have size density x density.
    :type density: int
    :returns:
        - (x, y, z) - Positions of pixels (meters)
        - (nx, ny, nz) - Normal vectors of pixels
        - areas - Areas of pixels (:math:`m^2`)
    """

    R = lumos.constants.EARTH_RADIUS

    max_angle = np.arccos(R / sat_z)

    angles_off_plane = np.linspace(-max_angle, max_angle, density)
    angles_on_plane = np.linspace(angle_past_terminator, max_angle, density)

    d_phi = abs(angles_off_plane[1] - angles_off_plane[0])
    d_theta = abs(angles_on_plane[1] - angles_on_plane[0])

    angles_on_plane, angles_off_plane = np.meshgrid(angles_on_plane, angles_off_plane)
    angles_on_plane, angles_off_plane = (
        angles_on_plane.flatten(),
        angles_off_plane.flatten(),
    )

    # Set up panel positions
    nz = 1 / np.sqrt(1 + np.tan(angles_on_plane) ** 2 + np.tan(angles_off_plane) ** 2)
    nx = np.tan(angles_off_plane) * nz
    ny = np.tan(angles_on_plane) * nz

    # Clip the panels which aren't visible to the satellite
    visible_to_sat = np.arccos(nz) < max_angle
    angles_off_plane, angles_on_plane = (
        angles_off_plane[visible_to_sat],
        angles_on_plane[visible_to_sat],
    )
    nx, ny, nz = nx[visible_to_sat], ny[visible_to_sat], nz[visible_to_sat]

    # Calculate Jacobian determinant to get panel areas
    x, y, z = nx * R, ny * R, nz * R

    phi = angles_off_plane
    theta = angles_on_plane

    dx_dr = nx / nz * z / R
    dx_dphi = z**3 / (R**2 * np.cos(phi) ** 2 * np.cos(theta) ** 2)
    dx_dtheta = -(ny / nz * nx / nz * z**3) / (R**2 * np.cos(theta) ** 2)

    dy_dr = np.tan(theta) * z / R
    dy_dphi = -(ny / nz * nx / nz * z**3) / (R**2 * np.cos(phi) ** 2)
    dy_dtheta = dx_dphi

    dz_dr = z / R
    dz_dphi = -(nx / nz * z**3) / (R**2 * np.cos(phi) ** 2)
    dz_dtheta = -(ny / nz * z**3) / (R**2 * np.cos(theta) ** 2)

    determinant = (
        dx_dr * (dy_dphi * dz_dtheta - dy_dtheta * dz_dphi)
        - dy_dr * (dx_dphi * dz_dtheta - dx_dtheta * dz_dphi)
        + dz_dr * (dx_dphi * dy_dtheta - dx_dtheta * dy_dphi)
    )

    areas = determinant * d_phi * d_theta

    return x, y, z, nx, ny, nz, areas


def get_intensity_satellite_frame(
    sat_surfaces,
    sat_height,
    sun_vec_body,
    obs_vec_body,
    dist_sat_2_obs,
    include_sun=True,
    include_earthshine=True,
    earth_panel_density=150,
    earth_brdf=None,
    wavelength_nm=None,
):
    """
    Calculates the flux scattered by a satellite and seen by an observer, in the satellite body frame.
    Does not currently work with earthshine, as the earthshine implementation is brightness-frame specific and needs to be reworked.

    :param sat_surfaces: List of surfaces on the satellite.
    :type sat_surfaces: list[lumos.geometry.Surface]
    :param sat_height: Height of the satellite above geodetic nadir (meters).
    :type sat_height: float
    :param sun_vec_body: Unit vector from satellite to sun in body frame.
    :type sun_vec_body: np.ndarray
    :param obs_vec_body: Unit vector from satellite to observer in body frame.
    :type obs_vec_body: np.ndarray
    :param dist_sat_2_obs: Distance from satellite to observer (meters).
    :type dist_sat_2_obs: float
    :param include_sun: Whether to include flux scattered by the satellite from the sun.
    :type include_sun: bool, optional
    :param include_earthshine: Whether to include flux scattered by the satellite from earthshine.
    :type include_earthshine: bool, optional
    :param earth_panel_density: Number of panels per axis in the earthshine mesh (total panels = density x density).
    :type earth_panel_density: int, optional
    :param earth_brdf: Function representing the BRDF of Earth's surface.
    :type earth_brdf: function, optional
    :param wavelength_nm: Wavelength(s) at which to evaluate BRDFs (scalar or array, in nm).
    :type wavelength_nm: float or np.ndarray, optional
    :return: Flux of light incident on the observer (W / m^2), scalar or array depending on wavelength_nm.
    :rtype: float or np.ndarray
    """

    # Detect vector wavelength
    lam = None
    vectorised = False
    if wavelength_nm is not None:
        lam = np.asarray(wavelength_nm, float)
        vectorised = lam.ndim > 0

    sat_height = max(float(sat_height), 0.0)
    earth_ratio = lumos.constants.EARTH_RADIUS / (
        lumos.constants.EARTH_RADIUS + sat_height
    )
    # Earth apparent angular radius from the satellite (cone around +Z nadir).
    horizon_angle = np.arcsin(np.clip(earth_ratio, -1.0, 1.0))

    # With +Z = nadir, nadir axis is +Z - (0,0,1)
    # angle from nadir to sun = acos(sun_z)
    angle_from_nadir = np.arccos(np.clip(sun_vec_body[2], -1.0, 1.0))
    if angle_from_nadir < horizon_angle:
        # sun is "behind earth"
        print("Not visible to observer")
        return 0.0 if not vectorised else np.zeros_like(lam)

    sun_vec_body = unit(sun_vec_body)
    obs_vec_body = unit(obs_vec_body)

    if include_earthshine:
        raise NotImplementedError(
            "Earthshine in the old implementation is brightness-frame specific. run with include_earthshine=False until reworked."
        )
        # Distances from earthshine panels to satellite

    intensity = np.zeros_like(lam) if vectorised else 0.0

    for surface in sat_surfaces:

        surface_normal = (
            surface.normal if not callable(surface.normal) else surface.normal(0.0)
        )
        surface_normal = np.asarray(surface_normal, float).reshape(
            3,
        )

        sun_contribution = 0.0

        mu_in = np.dot(surface_normal, sun_vec_body)
        mu_in = np.clip(mu_in, 0.0, None)

        mu_out = np.dot(surface_normal, obs_vec_body)
        mu_out = np.clip(mu_out, 0.0, None)

        if include_sun and (mu_in > 0) and (mu_out > 0):
            if vectorised:
                brdf_val = surface.brdf(
                    sun_vec_body, surface_normal, obs_vec_body, lam=lam
                )
                sun_contribution = brdf_val * mu_in * mu_out
            else:
                brdf_val = surface.brdf(
                    sun_vec_body, surface_normal, obs_vec_body, lam=wavelength_nm
                )
                sun_contribution = brdf_val * mu_in * mu_out

        intensity = intensity + surface.area * (sun_contribution) / dist_sat_2_obs**2
    return intensity


def _broadcast_geometry_inputs(
    sat_height,
    sun_vec_body,
    obs_vec_body,
    dist_sat_2_obs,
):
    """
    Broadcast geometry inputs to a common sample shape and flatten them for iteration.

    This helper accepts scalar or array-valued geometry inputs. `sun_vec_body` and
    `obs_vec_body` must have a final axis of length 3. The returned arrays are
    broadcast to a common shape and reshaped so the caller can evaluate one sample
    at a time.

    :param sat_height: Satellite height above Earth's surface in meters.
    :type sat_height: float or array_like
    :param sun_vec_body: Sun vector or vectors in the satellite body frame.
    :type sun_vec_body: array_like
    :param obs_vec_body: Observer vector or vectors in the satellite body frame.
    :type obs_vec_body: array_like
    :param dist_sat_2_obs: Distance or distances from satellite to observer in
        meters.
    :type dist_sat_2_obs: float or array_like
    :return: Tuple containing the broadcast sample shape, flattened satellite
        heights, flattened sun vectors, flattened observer vectors, and flattened
        observer distances.
    :rtype: tuple
    :raises ValueError: If `sun_vec_body` or `obs_vec_body` does not have a final
        axis of length 3.
    """

    sat_height = np.asarray(sat_height, float)
    sun_vec_body = np.asarray(sun_vec_body, float)
    obs_vec_body = np.asarray(obs_vec_body, float)
    dist_sat_2_obs = np.asarray(dist_sat_2_obs, float)

    if sun_vec_body.shape[-1] != 3:
        raise ValueError("sun_vec_body must have last axis size 3")
    if obs_vec_body.shape[-1] != 3:
        raise ValueError("obs_vec_body must have last axis size 3")

    sample_shape = np.broadcast_shapes(
        sat_height.shape,
        dist_sat_2_obs.shape,
        sun_vec_body.shape[:-1],
        obs_vec_body.shape[:-1],
    )

    sat_height_flat = np.broadcast_to(sat_height, sample_shape).reshape(-1)
    dist_flat = np.broadcast_to(dist_sat_2_obs, sample_shape).reshape(-1)
    sun_flat = np.broadcast_to(sun_vec_body, sample_shape + (3,)).reshape(-1, 3)
    obs_flat = np.broadcast_to(obs_vec_body, sample_shape + (3,)).reshape(-1, 3)

    return sample_shape, sat_height_flat, sun_flat, obs_flat, dist_flat


def get_intensity_observer_frame(
    sat_surfaces,
    sat_height,
    sun_vec_body,
    obs_vec_body,
    dist_sat_2_obs,
    include_sun=True,
    include_earthshine=True,
    earth_panel_density=150,
    earth_brdf=None,
    wavelength_nm=None,
):
    """
    Compute observer-frame irradiance from body-frame satellite geometry.

    Scalar or array-valued geometry inputs are broadcast to a common sample shape.
    If `wavelength_nm` is array-valued, the returned irradiance includes a final
    wavelength axis.

    :param sat_surfaces: List of surfaces on the satellite.
    :type sat_surfaces: list[lumos.geometry.Surface]
    :param sat_height: Satellite height above Earth's surface in meters.
    :type sat_height: float or array_like
    :param sun_vec_body: Unit sun vector or vectors in the satellite body frame.
    :type sun_vec_body: array_like
    :param obs_vec_body: Unit observer vector or vectors in the satellite body frame.
    :type obs_vec_body: array_like
    :param dist_sat_2_obs: Distance or distances from satellite to observer in
        meters.
    :type dist_sat_2_obs: float or array_like
    :param include_sun: Whether to include solar illumination scattered by the
        satellite.
    :type include_sun: bool, optional
    :param include_earthshine: Whether to include Earthshine contributions.
    :type include_earthshine: bool, optional
    :param earth_panel_density: Number of Earth panels per axis for Earthshine
        sampling.
    :type earth_panel_density: int, optional
    :param earth_brdf: BRDF model for Earth panels. Reserved for Earthshine
        calculations.
    :type earth_brdf: callable, optional
    :param wavelength_nm: Wavelength or wavelengths at which to evaluate the BRDFs,
        in nanometers.
    :type wavelength_nm: float or array_like, optional
    :return: Observer-frame irradiance in W / m^2. Returns a scalar for scalar
        inputs, an array with the broadcast sample shape for vectorized geometry, or
        an array with shape `sample_shape + (n_wavelength,)` for vectorized
        wavelengths.
    :rtype: float or numpy.ndarray
    :raises NotImplementedError: If `include_earthshine` is `True`, since
        Earthshine is not yet implemented in the LVLH path.
    """

    (
        sample_shape,
        sat_height_flat,
        sun_flat,
        obs_flat,
        dist_flat,
    ) = _broadcast_geometry_inputs(
        sat_height,
        sun_vec_body,
        obs_vec_body,
        dist_sat_2_obs,
    )

    lam = None if wavelength_nm is None else np.asarray(wavelength_nm, float)
    vectorized_lam = lam is not None and lam.ndim > 0
    if vectorized_lam:
        intensity = np.zeros((sat_height_flat.size, lam.size), float)
    else:
        intensity = np.zeros((sat_height_flat.size,), float)

    for i in range(sat_height_flat.size):
        intensity[i] = get_intensity_satellite_frame(
            sat_surfaces,
            sat_height=sat_height_flat[i],
            sun_vec_body=sun_flat[i],
            obs_vec_body=obs_flat[i],
            dist_sat_2_obs=dist_flat[i],
            include_sun=include_sun,
            include_earthshine=include_earthshine,
            earth_panel_density=earth_panel_density,
            earth_brdf=earth_brdf,
            wavelength_nm=wavelength_nm,
        )

    if sample_shape == ():
        return intensity[0]

    if vectorized_lam:
        return intensity.reshape(sample_shape + (lam.size,))
    return intensity.reshape(sample_shape)


_EARTH = None


def set_earth(body):
    """
    Set the default Earth ephemeris object used by time-based helper functions.

    The supplied object is stored in the module-level `_EARTH` variable and is used
    when an explicit `earth` argument is not provided.

    :param body: Skyfield Earth ephemeris object.
    :type body: skyfield.positionlib.Barycentric
    :return: None
    :rtype: None
    """

    global _EARTH
    _EARTH = body


def get_body_vectors_at_time(t, satellite, observer, sun, earth=_EARTH):
    """
    Compute the sun and observer vectors in the satellite body frame at a given time.
    :param t: Time of observation (Skyfield Time object)
    :type t: skyfield.timelib.Time
    :param satellite: Skyfield EarthSatellite object
    :type satellite: skyfield.sgp4lib.EarthSatellite
    :param observer: Skyfield Topos or position object for the observer
    :type observer: skyfield.toposlib.Topos or skyfield.positionlib.Geocentric
    :param sun: Skyfield ephemeris object for the sun
    :type sun: skyfield.positionlib.Barycentric
    :param earth: Skyfield ephemeris object for the earth (optional, defaults to global _EARTH)
    :type earth: skyfield.positionlib.Barycentric, optional

    :returns:
        - sun_vec_body: Unit vector from satellite to sun in satellite body frame
        - obs_vec_body: Unit vector from satellite to observer in satellite body frame
        - angle_past_terminator: Angle of satellite past terminator (radians)
        - dist_sat_to_obs: Distance from satellite to observer (meters)
        - sat_height: Height of satellite above Earth's surface (meters)
    :rtype: tuple
    """
    if earth is None:
        earth = _EARTH
    if earth is None:
        raise ValueError(
            "Earth ephemeris not set. Call lumos.calculatorLVLH.set_earth(eph['earth'])."
        )
    sat_state = satellite.at(t)
    r_sat_km = sat_state.position.km
    v_sat_km_s = sat_state.velocity.km_per_s

    r_obs_km = observer.at(t).position.km
    r_sun_km = sun.at(t).position.km

    r_earth_km = earth.at(t).position.km
    r_obs_km = r_obs_km - r_earth_km
    r_sun_km = r_sun_km - r_earth_km

    return get_body_vectors_from_state(
        r_sat_km,
        v_sat_km_s,
        r_obs_km,
        r_sun_km,
    )


def get_intensity_observer_frame_at_time(
    sat_surfaces,
    t,
    satellite,
    observer,
    sun,
    include_sun=True,
    include_earthshine=True,
    earth_panel_density=150,
    earth_brdf=None,
    wavelength_nm=None,
    earth=_EARTH,
):
    """

    This is a convenience wrapper around get_body_vectors_at_time and get_intensity_observer_frame to compute the intensity at a given time from skyfield objects.

    :param sat_surfaces: List of surfaces on satellite
    :type sat_surfaces: list[lumos.geometry.Surface]
    :param t: Time of observation (Skyfield Time object)
    :type t: skyfield.timelib.Time
    :param satellite: Skyfield EarthSatellite object
    :type satellite: skyfield.sgp4lib.EarthSatellite
    :param observer: Skyfield Topos or position object for the observer
    :type observer: skyfield.toposlib.Topos or skyfield.positionlib.Geocentric
    :param sun: Skyfield ephemeris object for the sun
    :type sun: skyfield.positionlib.Barycentric
    :param include_sun: Whether to include solar irradiance, defaults to True
    :type include_sun: bool, optional
    :param include_earthshine: Whether to include Earthshine, defaults to True
    :type include_earthshine: bool, optional
    :param earth_panel_density: Density of Earth panels, defaults to 150
    :type earth_panel_density: int, optional
    :param earth_brdf: Earth BRDF, defaults to None
    :type earth_brdf: skyfield.positionlib.Barycentric, optional
    :param wavelength_nm: Wavelength in nanometers, defaults to None
    :type wavelength_nm: float, optional
    :param earth: Skyfield ephemeris object for the earth, defaults to _EARTH
    :type earth: skyfield.positionlib.Barycentric, optional
    """
    if np.ndim(t) == 0:
        t_list = [t]
        sample_shape = ()
    else:
        t_list = list(t)
        sample_shape = (len(t_list),)

    lam = None if wavelength_nm is None else np.asarray(wavelength_nm, float)
    vectorized_lam = lam is not None and lam.ndim > 0

    if vectorized_lam:
        intensity = np.zeros((len(t_list), lam.size), float)
    else:
        intensity = np.zeros((len(t_list),), float)

    for i, t_i in enumerate(t_list):
        sun_vec_body, obs_vec_body, _, dist_m, sat_h = get_body_vectors_at_time(
            t_i, satellite, observer, sun, earth=earth
        )
        intensity[i] = get_intensity_observer_frame(
            sat_surfaces,
            sat_height=sat_h,
            sun_vec_body=sun_vec_body,
            obs_vec_body=obs_vec_body,
            dist_sat_2_obs=dist_m,
            include_sun=include_sun,
            include_earthshine=include_earthshine,
            earth_panel_density=earth_panel_density,
            earth_brdf=earth_brdf,
            wavelength_nm=wavelength_nm,
        )

    if sample_shape == ():
        return intensity[0]
    if vectorized_lam:
        return intensity.reshape(sample_shape + (lam.size,))
    return intensity.reshape(sample_shape)


def band_intensity_from_geometry(
    sat_surfaces,
    *,
    sat_height_m,
    sun_vec_body,
    obs_vec_body,
    dist_sat_obs_m,
    bandpass,
    panel_tracking=False,
    hinge_axis=np.array([0.0, 1.0, 0.0]),
    max_angle_deg=360.0,
):
    """

    Core one-sample band-integrated brightness from prepped geometry

    :param sat_surfaces: List of surfaces on satellite
    :type sat_surfaces: list[lumos.geometry.Surface]
    :param sat_height_m: Height of satellite above Earth's surface
    :type sat_height_m: float
    :param sun_vec_body: Sun vector in body frame
    :type sun_vec_body: numpy.ndarray
    :param obs_vec_body: Observer vector in body frame
    :type obs_vec_body: numpy.ndarray
    :param dist_sat_obs_m: Distance from satellite to observer in meters
    :type dist_sat_obs_m: float
    :param bandpass: Bandpass object containing wavelength array and response function
    :type bandpass: lumos.spectral.Bandpass
    :param panel_tracking: Whether to track the panels towards the sun, defaults to False
    :type panel_tracking: bool, optional
    :param hinge_axis: Axis around which the panels can rotate, defaults to np.array([0.0, 1.0, 0.0])
    :type hinge_axis: numpy.ndarray, optional
    :param max_angle_deg: Maximum angle in degrees that the panels can rotate, defaults to 360.0
    :type max_angle_deg: float, optional
    """
    if panel_tracking:
        lumos.attitude.apply_single_axis_tracking(
            sat_surfaces,
            np.asarray(sun_vec_body, float),
            hinge_axis=np.asarray(hinge_axis, float),
            max_angle_deg=max_angle_deg,
        )

    I_lam = get_intensity_observer_frame(
        sat_surfaces,
        sat_height=sat_height_m,
        sun_vec_body=sun_vec_body,
        obs_vec_body=obs_vec_body,
        dist_sat_2_obs=dist_sat_obs_m,
        include_sun=True,
        include_earthshine=False,
        wavelength_nm=bandpass.lam_nm,
    )
    return lumos.spectral.integrate_band_response(I_lam, bandpass)


def band_intensity_at_time(
    sat_surfaces,
    t,
    satellite,
    observer,
    sun,
    bandpass,
    panel_tracking=False,
    hinge_axis=np.array([0.0, 1.0, 0.0]),
    max_angle_deg=360.0,
    earth=_EARTH,
):
    """

    Skyfield convenience wrapper for one band integrated sample

    :param sat_surfaces: List of surfaces on satellite
    :type sat_surfaces: list[lumos.geometry.Surface]
    :param t: Time of observation (Skyfield Time object)
    :type t: skyfield.timelib.Time
    :param satellite: Skyfield EarthSatellite object
    :type satellite: skyfield.sgp4lib.EarthSatellite
    :param observer: Skyfield Topos object
    :type observer: skyfield.toposlib.Topos
    :param sun: Skyfield Body object
    :type sun: skyfield.bodies.Body
    :param bandpass: Bandpass object containing wavelength array and response function
    :type bandpass: lumos.spectral.Bandpass
    :param panel_tracking: Whether to track the panels towards the sun, defaults to False
    :type panel_tracking: bool, optional
    :param hinge_axis: Axis around which the panels can rotate, defaults to np.array([0.0, 1.0, 0.0])
    :type hinge_axis: numpy.ndarray, optional
    :param max_angle_deg: Maximum angle in degrees that the panels can rotate, defaults to 360.0
    :type max_angle_deg: float, optional
    :return: Band-integrated observer irradiance
    :rtype: float
    """

    sun_vec_body, obs_vec_body, _, dist_m, sat_h = get_body_vectors_at_time(
        t, satellite, observer, sun, earth=earth
    )

    return band_intensity_from_geometry(
        sat_surfaces,
        sat_height_m=sat_h,
        sun_vec_body=sun_vec_body,
        obs_vec_body=obs_vec_body,
        dist_sat_obs_m=dist_m,
        bandpass=bandpass,
        panel_tracking=panel_tracking,
        hinge_axis=hinge_axis,
        max_angle_deg=max_angle_deg,
    )


def _coerce_track(track_or_context):
    """
    Return a prepared orbit track from either an orbit context or a track object.

    If the input has a `track` attribute, that attribute is returned. Otherwise the
    input is returned unchanged.

    :param track_or_context: Orbit context or prepared orbit track.
    :type track_or_context: lumos.orbit.OrbitContext or lumos.orbit.PreparedOrbitTrack
    :return: Prepared orbit track.
    :rtype: lumos.orbit.PreparedOrbitTrack
    """

    return (
        track_or_context.track
        if hasattr(track_or_context, "track")
        else track_or_context
    )


def band_intensity_series(
    orbit_ctx_or_track,
    sat_surfaces_factory,
    bandpass,
    *,
    panel_tracking=False,
    hinge_axis=None,
    max_angle_deg=360.0,
    fill_value=np.nan,
):
    """
    Evaluate a bandpass over a prepared orbit time series.

    :param orbit_ctx_or_track: Prepared orbit context or prepared track containing
        body-frame geometry for each sample.
    :type orbit_ctx_or_track: lumos.orbit.OrbitContext or lumos.orbit.PreparedOrbitTrack
    :param sat_surfaces_factory: Callable returning the satellite surface list used
        for the series evaluation.
    :type sat_surfaces_factory: callable
    :param bandpass: Bandpass object containing wavelength samples and response.
    :type bandpass: lumos.spectral.Bandpass
    :param panel_tracking: Whether to apply single-axis panel tracking toward the
        sun at each sample.
    :type panel_tracking: bool, optional
    :param hinge_axis: Panel hinge axis in the satellite body frame. If `None`,
        defaults to `[0, 1, 0]`.
    :type hinge_axis: array_like, optional
    :param max_angle_deg: Maximum allowed panel rotation angle in degrees.
    :type max_angle_deg: float, optional
    :param fill_value: Value used for samples with invalid or missing geometry.
    :type fill_value: float, optional
    :return: Band-integrated observer irradiance for each time sample in the input
        track, aligned with `track.times_utc`.
    :rtype: numpy.ndarray
    """

    track = _coerce_track(orbit_ctx_or_track)

    if hinge_axis is None:
        hinge_axis = np.array([0.0, 1.0, 0.0], float)
    else:
        hinge_axis = np.asarray(hinge_axis, float)

    sat_surfaces = sat_surfaces_factory()
    out = np.full(len(track.times_utc), fill_value, dtype=float)

    for k in range(len(track.times_utc)):
        if not np.isfinite(track.sat_height_m[k]):
            continue
        if not np.isfinite(track.dist_sat_obs_m[k]):
            continue
        if not np.all(np.isfinite(track.sun_vec_body[k])):
            continue
        if not np.all(np.isfinite(track.obs_vec_body[k])):
            continue

        out[k] = band_intensity_from_geometry(
            sat_surfaces,
            sat_height_m=track.sat_height_m[k],
            sun_vec_body=track.sun_vec_body[k],
            obs_vec_body=track.obs_vec_body[k],
            dist_sat_obs_m=track.dist_sat_obs_m[k],
            bandpass=bandpass,
            panel_tracking=panel_tracking,
            hinge_axis=hinge_axis,
            max_angle_deg=max_angle_deg,
        )
    return out


def get_sun_alt_az(time, observer_location):
    """
    Convenience function for finding the altitude and azimuth of the sun

    :param time: Time of observation
    :type time: :class:`astropy.time.Time`
    :param observer_location: Location of observation
    :type observer_location: :class:`astropy.coordinates.EarthLocation`
    """

    aa_frame = astropy.coordinates.AltAz(obstime=time, location=observer_location)
    sun_altaz = astropy.coordinates.get_sun(time).transform_to(aa_frame)
    return sun_altaz.alt.degree, sun_altaz.az.degree
