# Imports
from lumos.geometry import Surface
from lumos.spectral import diffuse_spectral_energy_function
from lumos.brdf.library import BINOMIAL, PHONG
import numpy as np
import os
import pandas as pd

# A simple satellite model using GaAs solar cells

# - Coordinate convention assumed here:
#   -Z = nadir, +Z = zenith
#   +X / -X = bus fore/aft
#   +Y / -Y = port/starboard

A_flat_override = 2.0 * 2.0

# Solar wing overall size
wing_L = 10.0  # m
wing_W = 2.0  # m
A_wing = wing_L * wing_W  # per wing

material_dir = r"~/lumos-sat/data/material-spectral-data"

# Black paint (main bus face)
black_DHR_df = pd.read_csv(
    os.path.join(material_dir, "black-simple-dhr.csv"), comment="#"
)
composite_lam = black_DHR_df.iloc[:, 0].to_numpy(dtype=float)
composite_rho = black_DHR_df.iloc[:, 1].to_numpy(dtype=float)
Composite_DHR_df = pd.DataFrame({"lambda_nm": composite_lam, "R": composite_rho})

# GaAs solar cells
GaAs_DHR_df = pd.read_csv(os.path.join(material_dir, "gaAs-dhr.csv"), comment="#")
GaAs_lam = GaAs_DHR_df.iloc[:, 0].to_numpy(dtype=float)
GaAs_rho = GaAs_DHR_df.iloc[:, 1].to_numpy(dtype=float)
GaAs_DHR_df = pd.DataFrame({"lambda_nm": GaAs_lam, "R": GaAs_rho})

bus_BRDF = PHONG(Kd=0.25, Ks=0.04, n=10.0)
panel_BRDF = PHONG(Kd=0.05, Ks=0.25, n=70.0)

# Normals
n_posX = np.array([+1.0, 0.0, 0.0])
n_negX = np.array([-1.0, 0.0, 0.0])
n_posY = np.array([0.0, +1.0, 0.0])
n_negY = np.array([0.0, -1.0, 0.0])
n_posZ = np.array([0.0, 0.0, +1.0])
n_negZ = np.array([0.0, 0.0, -1.0])

# Panel normals
n_wing_port = n_negZ
n_wing_starboard = n_negZ

# BRDF slots
BRDFS = {
    # Nadir "flat face"
    "bus_black_face": diffuse_spectral_energy_function(
        bus_BRDF,
        Composite_DHR_df,
        normalize_incident_deg=8.0,
        rho_samples=30000,
        rho_seed=42,
    ),
    # GaAs solar cell coverglass: low diffuse, moderate specular
    "solar_gaas": diffuse_spectral_energy_function(
        panel_BRDF,
        GaAs_DHR_df,
        normalize_incident_deg=8.0,
        rho_samples=30000,
        rho_seed=42,
    ),
}


# Helper to avoid typos
def brdf(name: str):
    if name not in BRDFS:
        raise KeyError(f"BRDF '{name}' not found. Available: {list(BRDFS.keys())}")
    return BRDFS[name]


s_bus_nadir = Surface(A_flat_override, n_negZ, brdf("bus_black_face"))

# Solar wings + Dishes
s_wing_port = Surface(A_wing, n_wing_port, brdf("solar_gaas"))
s_wing_star = Surface(A_wing, n_wing_starboard, brdf("solar_gaas"))

# set name/label attributes used by diagnostics
for label, s in [
    ("bus_nadir", s_bus_nadir),
    ("solar_wing_port", s_wing_port),
    ("solar_wing_starboard", s_wing_star),
]:
    s.name = label
    s.label = label


def make_surfaces_infer_brdfs():
    surfaces = [
        s_bus_nadir,
        s_wing_port,
        s_wing_star,
    ]

    for s in surfaces:
        # Heuristic: any surface using the "solar_gaas" BRDF or with wing area -> solar panel
        brdf_obj = getattr(s, "brdf", None)
        brdf_name = getattr(brdf_obj, "name", None) if brdf_obj is not None else None

        if brdf_name == "solar_gaas" or getattr(s, "area", None) == globals().get(
            "A_wing", None
        ):
            s.is_panel = True
            s.is_solar_panel = True
        else:
            s.is_panel = getattr(s, "is_panel", False)
            s.is_solar_panel = getattr(s, "is_solar_panel", False)

        # store the zero-angle reference once
        if getattr(s, "is_panel", False) and getattr(s, "nominal_normal", None) is None:
            s.nominal_normal = np.array(s.normal, dtype=float)

    return surfaces


SURFACES_INFER_BRDFS = make_surfaces_infer_brdfs()
