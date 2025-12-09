# Imports
from lumos.geometry import Surface
from lumos.brdf.library import BINOMIAL, PHONG
import numpy as np

# This code implements a satellite model
# The model is for the optus D3 satellite

#there are four solar panels per array, with Gallium arsenide cells, The DHR plot used has both specular and diffuse reflectance components, notably, the diffuse reflectance for the panels is very low.
#the dishes on the satellite are two dual 2.3m grid shaped deployable reflectors
#The satellite uses the GEOSTAR-2 bus, and at this time, has a Mission Extension Pod (MEP) attached to it.
#Bus Dimensions (HxWxL): 1.75 m x 1.7 m x 1.8 m
#Construction: Composite/Al


# Constants
chassis_area = 3.65 # m^2
solar_array_area = 22 # m^2

chassis_normal = np.array([0, 0, -1])
solar_array_normal = np.array([0, 1, 0])

B = np.array([[3.34, -98.085]])
C = np.array([[-999.999, 867.538, 1000., 1000., -731.248, 618.552, -294.054, 269.248, -144.853, 75.196]])
lab_chassis_brdf = BINOMIAL(B, C, d = 3.0, l1 = -5)

SURFACES_INFER_BRDFS = [
    Surface(1.0, chassis_normal, PHONG(0.34, 0.40, 8.9)),
        Surface(1.0, solar_array_normal, PHONG(0.15, 0.25, 0.26))
        ]