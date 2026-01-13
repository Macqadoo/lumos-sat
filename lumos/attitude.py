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
    surfaces, sun_vec, hinge_axis=(0, 1.0, 0), max_angle_deg=360.0
):
    """
    Only rotates surfaces where `s.is_solar_panel == True`.
    All surfaces remain present in surfaces and will contribute to intensity.
    """
    max_rad = np.deg2rad(abs(max_angle_deg))
    for s in surfaces:
        # only rotate true solar panels
        if not getattr(s, "is_solar_panel", False):
            continue
        nominal = getattr(s, "nominal_normal", None)
        if nominal is None:
            nominal = np.array(s.normal, dtype=float)
            s.nominal_normal = nominal.copy()
        angle = compute_single_axis_panel_angle(sun_vec, hinge_axis, nominal)
        angle = np.clip(angle, -max_rad, max_rad)
        s.normal = rotate_vector_around_axis(nominal, hinge_axis, angle)
