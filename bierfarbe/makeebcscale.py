# Beer EBC to sRGB Model Generator
# Copyright 2022 Thomas Ascher
# SPDX-License-Identifier: MIT
# Run within Jupyter Notebook
import math
import numpy as np
import matplotlib.pyplot as plt
import colour
import colour.plotting

# Adjust the following constants to alter the generated model
BEER_GLAS_DIAMETER_CM = 7.5 # 7.5 is average diameter of a Teku glas
OBSERVER_NAME = 'CIE 1931 2 Degree Standard Observer'
ILLUMINANT_NAME = 'C'
MAX_EBC = 80
POLY_DEGREE_R = 4
POLY_DEGREE_G = 4
POLY_DEGREE_B = 6

OBSERVER = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
ILLUMINANT = colour.SDS_ILLUMINANTS[ILLUMINANT_NAME]
ILLUMINANT_XY = colour.CCS_ILLUMINANTS[OBSERVER_NAME][ILLUMINANT_NAME]
ASBC_SHAPE = colour.SpectralShape(380, 780, 5)

def transmission_sd(ebc, path_cm):
    srm = ebc / 1.94
    wl = ASBC_SHAPE.range()
    values = 10**(-(srm / 12.7) * (0.018747 * math.e**(-(wl - 430.0) / 13.374) + 0.98226 * math.e** (-(wl - 430.0) / 80.514)) * path_cm)
    return colour.SpectralDistribution(values, wl)

EBC = np.arange(0, MAX_EBC, dtype='int')
RGB = []
for i in EBC:
    XYZ = colour.sd_to_XYZ(transmission_sd(i, BEER_GLAS_DIAMETER_CM), cmfs=OBSERVER, illuminant=ILLUMINANT) / 100.0
    RGB.append(colour.XYZ_to_sRGB(XYZ, illuminant=ILLUMINANT_XY))

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in RGB], columns=len(EBC), **{'standalone': False})
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
#fig_scale.savefig('colorscale.pdf', format='pdf')

R = [ 255 * i[0] for i in RGB]
G = [ 255 * i[1] for i in RGB]
B = [ 255 * i[2] for i in RGB]

R_COEFF = np.polyfit(EBC, R, POLY_DEGREE_R)
G_COEFF = np.polyfit(EBC, G, POLY_DEGREE_G)
B_COEFF = np.polyfit(EBC, B, POLY_DEGREE_B)

fig_model = plt.figure()
ax_model = fig_model.subplots(1)
ax_model.set_title('sRGB Spectrum for Average Beers at ' + '{:.1f}'.format(BEER_GLAS_DIAMETER_CM)
+ ' cm Glas Diameter\nwith ' + OBSERVER_NAME + ' and ' + ILLUMINANT_NAME + ' Illuminant')
ax_model.xaxis.set_label_text('EBC')
ax_model.yaxis.set_label_text('Channel Intensity')

def plot_channel(values, coeff, color, label):
    ax_model.plot(EBC, values, color=color, label=label)
    ax_model.plot(EBC, np.poly1d(coeff)(EBC), color=color, label='Model ' + label, linestyle=':')

plot_channel(R, R_COEFF, '#ff0000', 'R')
plot_channel(G, G_COEFF, '#00ff00', 'G')
plot_channel(B, B_COEFF, '#0000ff', 'B')
ax_model.legend()

def print_poly(name, coeff):
    parts = []
    exp = len(coeff) - 1
    for i in coeff:
        multiplier = ''
        if exp == 1:
            multiplier = '*EBC'
        elif exp > 1:
            multiplier = '*EBC**' + str(exp)
        parts.append('(' + '{:.4E}'.format(i) + multiplier + ')')
        exp = exp - 1
    print(name + '=round(max(0, min(255, ' + '+'.join(parts) + ')))')

print('Model:')
print('# observer=' + OBSERVER_NAME + ', illuminant=' + ILLUMINANT_NAME + ', path=' + str(BEER_GLAS_DIAMETER_CM) + 'cm')
print_poly('R', R_COEFF)
print_poly('G', G_COEFF)
print_poly('B', B_COEFF)