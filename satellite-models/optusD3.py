# Imports
from lumos.geometry import Surface
from lumos.spectral import diffuse_spectral_energy_function
from lumos.brdf.library import BINOMIAL, PHONG
import numpy as np
import os
import pandas as pd

# Optus D3 (GEOSTAR-2 bus) - baseline surface model
# Notes:
# - Coordinate convention assumed here:
#   +Z = nadir, -Z = zenith
#   +X / -X = bus fore/aft
#   +Y / -Y = port/starboard

# GEOSTAR-2 bus dimensions
bus_H = 2.4  # m  (along Z)
bus_W = 2.2  # m  (along Y)
bus_L = 2.8  # m  (along X)

# face areas (bus faces)
A_x_face = bus_W * bus_H  # faces perpendicular to X (Y x Z)
A_y_face = bus_L * bus_H  # faces perpendicular to Y (X x Z)
A_z_face = bus_L * bus_W  # faces perpendicular to Z (X x Y)

# Suggested main face dimension from Optus conversation
A_flat_override = 2.2 * 2.4

# Solar wing overall size
wing_L = 9.8  # m
wing_W = 2.0  # m
A_wing = wing_L * wing_W  # per wing

# Dish geometry (2x Gregorian; assuming 2.4 m diameter each)
dish_D = 2.4  # m
A_dish = np.pi * (dish_D / 2.0) ** 2  # projected area approximation

# Diffuse reflectance arrays
# Black c-c composite
# Aluminum anodised surface
# GaAs solar cell
material_dir = r"~/lumos-sat/data/material-spectral-data"

# Black Composite (main bus face)
black_DHR_df = pd.read_csv(
    os.path.join(material_dir, "black-simple-dhr.csv"), comment="#"
)
composite_lam = black_DHR_df.iloc[:, 0].to_numpy(dtype=float)
composite_rho = black_DHR_df.iloc[:, 1].to_numpy(dtype=float) / 100
Composite_DHR_df = pd.DataFrame({"lambda_nm": composite_lam, "R": composite_rho})

# Aluminum mirror like surface
al_6061_DHR_df = pd.read_csv(os.path.join(material_dir, "al6061-dhr.csv"), comment="#")
al_6061_lam = al_6061_DHR_df.iloc[:, 0].to_numpy(dtype=float)
al_6061_rho = al_6061_DHR_df.iloc[:, 1].to_numpy(dtype=float) / 100
al_6061_DHR_df = pd.DataFrame({"lambda_nm": al_6061_lam, "R": al_6061_rho})

# Reflectors (aluminum)
Al_DHR_df = pd.read_csv(os.path.join(material_dir, "aluminum-dhr.csv"), comment="#")
Al_lam = Al_DHR_df.iloc[:, 0].to_numpy(dtype=float)
Al_rho = Al_DHR_df.iloc[:, 1].to_numpy(dtype=float) / 100
Al_DHR_df = pd.DataFrame({"lambda_nm": Al_lam, "R": Al_rho})

# GaAs solar cells
GaAs_DHR_df = pd.read_csv(os.path.join(material_dir, "gaAs-dhr.csv"), comment="#")
GaAs_lam = GaAs_DHR_df.iloc[:, 0].to_numpy(dtype=float)
GaAs_rho = GaAs_DHR_df.iloc[:, 1].to_numpy(dtype=float) / 100
GaAs_DHR_df = pd.DataFrame({"lambda_nm": GaAs_lam, "R": GaAs_rho})

bus_BRDF = PHONG(Kd=0.25, Ks=0.04, n=10.0)
bus_mirror_BRDF = PHONG(Kd=0.8, Ks=0.95, n=100.0)
dish_BRDF = PHONG(Kd=0.15, Ks=0.25, n=60.0)
panel_BRDF = PHONG(Kd=0.05, Ks=0.25, n=80.0)

# -----------------------
# Normals
# -----------------------
n_posX = np.array([+1.0, 0.0, 0.0])
n_negX = np.array([-1.0, 0.0, 0.0])
n_posY = np.array([0.0, +1.0, 0.0])
n_negY = np.array([0.0, -1.0, 0.0])
n_posZ = np.array([0.0, 0.0, +1.0])
n_negZ = np.array([0.0, 0.0, -1.0])

# Panel normals
n_wing_port = n_negZ
n_wing_starboard = n_negZ

# Static Dish normals
n_dish_1 = [+0.259, 0, -0.966]
n_dish_2 = [-0.259, 0, -0.966]

# -----------------------
# BRDF slots
# -----------------------
#
# Placeholder BRDF priors (tune/replace)
BRDFS = {
    # Bus thermal blanket/dark surfaces
    "bus_mirror_face": diffuse_spectral_energy_function(
        bus_mirror_BRDF,
        al_6061_DHR_df,
        normalize_incident_deg=8.0,
        rho_samples=30000,
        rho_seed=42,
    ),
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
    # Aluminium anodised
    "al_anodised": PHONG(Kd=0.18, Ks=0.08, n=30.0),
    # Reflector surfaces (dual-shell/gridded shaped reflectors): moderately specular
    "dish_reflector": diffuse_spectral_energy_function(
        dish_BRDF,
        al_6061_DHR_df,
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
s_bus_posZ = Surface(A_z_face, n_posZ, brdf("bus_black_face"))
s_bus_posX = Surface(A_x_face, n_posX, brdf("bus_black_face"))
s_bus_negX = Surface(A_x_face, n_negX, brdf("bus_black_face"))
s_bus_posY = Surface(A_y_face, n_posY, brdf("bus_mirror_face"))
s_bus_negY = Surface(A_y_face, n_negY, brdf("bus_mirror_face"))

# Solar wings + Dishes
s_wing_port = Surface(A_wing, n_wing_port, brdf("solar_gaas"))
s_wing_star = Surface(A_wing, n_wing_starboard, brdf("solar_gaas"))
s_dish_1 = Surface(A_dish, n_dish_1, brdf("dish_reflector"))
s_dish_2 = Surface(A_dish, n_dish_2, brdf("dish_reflector"))

# set name/label attributes used by diagnostics
for label, s in [
    ("bus_nadir", s_bus_nadir),
    ("bus_posZ", s_bus_posZ),
    ("bus_posX", s_bus_posX),
    ("bus_negX", s_bus_negX),
    ("bus_posY", s_bus_posY),
    ("bus_negY", s_bus_negY),
    ("solar_wing_port", s_wing_port),
    ("solar_wing_starboard", s_wing_star),
    ("dish_1", s_dish_1),
    ("dish_2", s_dish_2),
]:
    s.name = label
    s.label = label


def make_surfaces_infer_brdfs():
    surfaces = [
        s_bus_nadir,
        s_bus_posZ,
        s_bus_posX,
        s_bus_negX,
        s_bus_posY,
        s_bus_negY,
        s_wing_port,
        s_wing_star,
        s_dish_1,
        s_dish_2,
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
