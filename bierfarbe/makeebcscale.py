# Beer SRM/EBC to sRGB Model Generator
# Copyright 2022 Thomas Ascher
# SPDX-License-Identifier: MIT
# The following dependencies are required: numpy, matplotlib, colour-science
# Run within Jupyter Notebook for better visualisation
import math
import numpy as np
import matplotlib.pyplot as plt
import colour
import colour.plotting

# For more information about the computational methods see:
# https://www.homebrewtalk.com/threads/on-the-calculation-of-srm-rgb-values-in-the-srgb-color-space.413581/
# https://www.homebrewtalk.com/threads/ebc-or-srm-to-color-rgb.78018
# https://stackoverflow.com/questions/58722583/how-do-i-convert-srm-to-lab-using-e-308-as-an-algorithm
# deLange, A.J. (2016). Color. Brewing Materials and Processes. Elsevier. https://doi.org/10.1016/b978-0-12-799954-8.00011-3

# Adjust the following constants to alter the generated model
BEER_GLAS_DIAMETER_CM = 7.5 # 7.5 is average diameter of a Teku glas
USE_EBC_SCALE = False
MAX_SRM_SCALE_VALUE = 40
POLY_DEGREE_R = 4
POLY_DEGREE_G = 4
POLY_DEGREE_B = 6
OBSERVER = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
ILLUMINANT = colour.SDS_ILLUMINANTS['C']
ILLUMINANT_XY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']
ASBC_SHAPE = colour.SpectralShape(380, 780, 5)

# TODO: implement sampling distance for scale and fix tick labels in scale figure

if USE_EBC_SCALE == True:
    unit_name = 'EBC'
    unit_conversion = 1 / 1.94
    max_scale_value = MAX_SRM_SCALE_VALUE * 2
else:
    unit_name = 'SRM'
    unit_conversion = 1.0
    max_scale_value = MAX_SRM_SCALE_VALUE

def transmission_sd(srm, path_cm):
    wl = ASBC_SHAPE.range()
    values = 10**(-(srm / 12.7) * (0.018747 * math.e**(-(wl - 430.0) / 13.374) + 0.98226 * math.e**(-(wl - 430.0) / 80.514)) * path_cm)
    return colour.SpectralDistribution(values, wl)

scale = np.arange(start=0, stop=max_scale_value+1, dtype='int')
rgb = []
for i in scale:
    xyz = colour.sd_to_XYZ(transmission_sd(i * unit_conversion, BEER_GLAS_DIAMETER_CM), cmfs=OBSERVER, illuminant=ILLUMINANT) / 100.0
    rgb.append(colour.XYZ_to_sRGB(xyz, illuminant=ILLUMINANT_XY))

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in rgb], **{'standalone': False})
ax_scale.xaxis.set_label_text(unit_name)
ax_scale.xaxis.set_ticks_position('bottom')

r = [ 255 * i[0] for i in rgb]
g = [ 255 * i[1] for i in rgb]
b = [ 255 * i[2] for i in rgb]

r_coeff = np.polyfit(scale, r, POLY_DEGREE_R)
g_coeff = np.polyfit(scale, g, POLY_DEGREE_G)
b_coeff = np.polyfit(scale, b, POLY_DEGREE_B)

fig_model = plt.figure()
ax_model = fig_model.subplots(1)
ax_model.set_title(unit_name + ' to sRGB Model for ' + '{:.1f}'.format(BEER_GLAS_DIAMETER_CM) + ' cm Glas Diameter')
ax_model.xaxis.set_label_text(unit_name)
ax_model.yaxis.set_label_text('Channel Intensity')

def plot_channel(values, coeff, color, label):
    ax_model.plot(scale, values, color=color, label=label)
    ax_model.plot(scale, np.poly1d(coeff)(scale), color=color, label='Model ' + label, linestyle=':')

plot_channel(r, r_coeff, '#ff0000', 'R')
plot_channel(g, g_coeff, '#00ff00', 'G')
plot_channel(b, b_coeff, '#0000ff', 'B')
ax_model.legend()

def format_poly_const(val):
    return '{:.3e}'.format(val)

def print_poly(name, unit_name, coeff):
    var_name = unit_name.lower()
    text=''
    for i in reversed(coeff[1:]):
        text += format_poly_const(i) + '+' + var_name + '*('
    text += format_poly_const(coeff[0])
    text = name + '=round(max(0, min(255, ' + text + ')' * (len(coeff) -1 + 3)
    print(text)

print('# ' + unit_name + ' to sRGB model for ' + str(BEER_GLAS_DIAMETER_CM) + ' cm glas diameter')
print_poly('r', unit_name, r_coeff)
print_poly('g', unit_name, g_coeff)
print_poly('b', unit_name, b_coeff)