# (c) 2019 Thomas Mansencal
# https://stackoverflow.com/questions/58722583/how-do-i-convert-srm-to-lab-using-e-308-as-an-algorithm

import colour
import numpy as np

EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
TRANSMISSION_PATH_CM = 15.0
ASBC_SHAPE = colour.SpectralShape(380, 780, 5)
OBSERVER = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
ILLUMINANT = colour.SDS_ILLUMINANTS['C']
ILLUMINANT_XY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']

def beer_transmission_sd(srm, path_cm=TRANSMISSION_PATH_CM, shape=ASBC_SHAPE):
    e = np.exp(1)
    wl = shape.range()
    values = np.exp(-(srm / 12.7) * (0.018747 * e**(-(wl - 430) / 13.374) + 0.98226 * e** (-(wl - 430) / 80.514)) * path_cm)
    return colour.SpectralDistribution(values, wl)

def ebc_to_rgb(ebc):
    srm = ebc / 1.97
    xyz = colour.sd_to_XYZ(beer_transmission_sd(srm), cmfs=OBSERVER, illuminant=ILLUMINANT) / 100.0
    rgb = np.clip(colour.XYZ_to_sRGB(xyz, illuminant=ILLUMINANT_XY), 0, 1) * 255.0
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))