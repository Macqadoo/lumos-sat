"""
Functions for satellite attitude calculations.
"""

import numpy as np


def rotate_vector_around_axis(v, axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return v * cos_a + np.cross(axis, v) * sin_a + axis * (axis.dot(v)) * (1 - cos_a)


def compute_single_axis_panel_angle(sun_vec, hinge_axis, nominal_normal):
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
    sun_proj = sun_vec - hinge_axis * np.dot(sun_vec, hinge_axis)
    nom_proj = nominal_normal - hinge_axis * np.dot(nominal_normal, hinge_axis)
    if np.linalg.norm(sun_proj) < 1e-9 or np.linalg.norm(nom_proj) < 1e-9:
        return 0.0
    sun_p = sun_proj / np.linalg.norm(sun_proj)
    nom_p = nom_proj / np.linalg.norm(nom_proj)
    cross = np.cross(nom_p, sun_p)
    sin_theta = np.dot(cross, hinge_axis)
    cos_theta = np.dot(nom_p, sun_p)
    return np.arctan2(sin_theta, cos_theta)


def apply_single_axis_tracking(
    surfaces, sun_vec_body, hinge_axis=(0, 1.0, 0), max_angle_deg=360.0
):
    """
    Only rotates surfaces where `s.is_solar_panel == True`.
    All surfaces remain present in surfaces and will contribute to intensity.
    """
    max_rad = np.deg2rad(abs(max_angle_deg))
    for s in surfaces:
        if not getattr(s, "is_solar_panel", False):
            continue
        nominal = getattr(s, "nominal_normal", None)
        if nominal is None:
            nominal = np.array(s.normal, dtype=float)
            s.nominal_normal = nominal.copy()
        angle = compute_single_axis_panel_angle(sun_vec_body, hinge_axis, nominal)
        angle = np.clip(angle, -max_rad, max_rad)
        s.normal = rotate_vector_around_axis(nominal, hinge_axis, angle)


def apply_rigid_body_spin(
    surfaces,
    spin_axis_body=(0.0, 1.0, 0.0),
    spin_hz=1.0,
    dt_sec=None,
    t_sec=None,
    *,
    phase0_rad=0.0,
    state_attr="_rigid_spin_phase_rad",
    cache_attr="nominal_normal",
):
    """
    Rotate ALL surfaces as a single rigid body spinning about spin_axis_body.

    Spin is specified in Hz (revolutions per second):
        omega = 2*pi*spin_hz   [rad/s]
        angle(t) = omega*t + phase0_rad

      - dt_sec: increments an internal phase accumulator stored on the surfaces list
      - t_sec: uses absolute time since start (no accumulator)

    Parameters
    surfaces : list[Surface]
        List of lumos.geometry.Surface objects (or similar) with .normal (3,)
    spin_axis_body : (3,)
        Rotation axis in the SAME frame as s.normal (your LVLH/body frame).
    spin_hz : float
        Spins per second (revolutions per second). Example: 3.0 means 3 full rotations per second.
    dt_sec : float | None
        Time step in seconds.
    t_sec : float | None
        Absolute time in seconds since start.
    phase0_rad : float
        Initial phase offset (radians).
    state_attr : str
        Attribute name stored on the surfaces list for accumulated phase (dt mode).
    cache_attr : str
        Per-surface attribute storing the reference normal (defaults to nominal_normal).

    Returns
    angle_rad : float
        The rotation angle used for this call (radians).
    """
    axis = np.asarray(spin_axis_body, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        raise ValueError("spin_axis_body must be non-zero.")
    axis = axis / n

    spin_hz = float(spin_hz)
    omega = 2.0 * np.pi * spin_hz  # rad/s

    # determine angle
    if (dt_sec is None) == (t_sec is None):
        raise ValueError("Provide exactly one of dt_sec or t_sec (not both).")

    if t_sec is not None:
        angle = omega * float(t_sec) + float(phase0_rad)
    else:
        # store phase on the list object itself (works if you reuse the same list instance)
        if not hasattr(surfaces, state_attr):
            setattr(surfaces, state_attr, float(phase0_rad))
        phase = getattr(surfaces, state_attr) + omega * float(dt_sec)
        setattr(surfaces, state_attr, phase)
        angle = phase

    # apply same rotation to all surfaces (from fixed reference)
    for s in surfaces:
        if getattr(s, cache_attr, None) is None:
            setattr(s, cache_attr, np.array(s.normal, dtype=float))

        n0 = np.array(getattr(s, cache_attr), dtype=float)
        s.normal = rotate_vector_around_axis(n0, axis, angle)

    return angle
