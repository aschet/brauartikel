# Beer SRM/EBC to sRGB Model Generator
# Copyright 2022 Thomas Ascher
# sRGB value generation was partially derived from an implementation by Thomas Mansencal
# For more information about the computational methods see:
# deLange, A.J. (2016). Color. Brewing Materials and Processes. Elsevier. https://doi.org/10.1016/b978-0-12-799954-8.00011-3
# The following dependencies are required: numpy, matplotlib, colour-science
# Run within Jupyter Notebook for better visualisation
# SPDX-License-Identifier: MIT
import math
import numpy as np
import matplotlib.pyplot as plt
import colour
import colour.plotting

# Adjust the following constants to alter the generated model
BEER_GLAS_DIAMETER_CM = 7.5
USE_EBC_SCALE = False
MAX_SCALE_VALUE = 50
POLY_DEGREE_R = 5
POLY_DEGREE_G = 5
POLY_DEGREE_B = 7
OBSERVER = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
ILLUMINANT = colour.SDS_ILLUMINANTS['C']
ILLUMINANT_XY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']

if USE_EBC_SCALE == True:
    unit_name = 'EBC'
    unit_conversion = 1 / 1.94
else:
    unit_name = 'SRM'
    unit_conversion = 1.0

def poly_const_to_text(val):
    return '{:.4e}'.format(val)

def poly_to_text(coeff, input_name):
    text=''
    for i in reversed(coeff[1:]):
        text += poly_const_to_text(i) + '+' + input_name + '*('
    text += poly_const_to_text(coeff[0])
    text = text + ')' * (len(coeff) - 1)
    return text

def compile_poly(coeff, input_name):
    poly_text = poly_to_text(coeff, input_name.lower())
    code = compile(poly_text, 'model', 'eval')
    return poly_text, code

def eval_poly(code):
    srm = scale
    return eval(code)

def calc_r2(actual, predicted):
    corr_matrix = np.corrcoef(actual, predicted)
    corr = corr_matrix[0, 1]
    r2 = corr**2
    return r2

# Generate sRGB input data for model fit
scale = np.arange(start=0, stop=MAX_SCALE_VALUE+1, dtype='int')
wl = colour.SpectralShape(380, 780, 5).range()
rgb = []
for i in scale:
    srm = i * unit_conversion
    values = 10**(-(srm / 12.7) * (0.018747 * math.e**(-(wl - 430.0) / 13.374) + 0.98226 * math.e**(-(wl - 430.0) / 80.514)) * BEER_GLAS_DIAMETER_CM)
    xyz = colour.sd_to_XYZ(colour.SpectralDistribution(values, wl), cmfs=OBSERVER, illuminant=ILLUMINANT) / 100.0
    rgb.append(colour.XYZ_to_sRGB(xyz, illuminant=ILLUMINANT_XY))

r = [i[0] for i in rgb]
g = [i[1] for i in rgb]
b = [i[2] for i in rgb]

# Fit data
r_coeff = np.polyfit(scale, r, POLY_DEGREE_R)
g_coeff = np.polyfit(scale, g, POLY_DEGREE_G)
b_coeff = np.polyfit(scale, b, POLY_DEGREE_B)

# Generate and compile model code
r_text, r_code = compile_poly(r_coeff, unit_name)
g_text, g_code = compile_poly(g_coeff, unit_name)
b_text, b_code = compile_poly(b_coeff, unit_name)

# Generate sRGB output
r_new = eval_poly(r_code)
g_new = eval_poly(g_code)
b_new = eval_poly(b_code)

for i in zip(r_new, g_new, b_new):
    rgb.append(i)

# Print model
print('# ' + unit_name + ' to sRGB model fitted up to ' + str(MAX_SCALE_VALUE) + ' ' + unit_name + ' for a ' + str(BEER_GLAS_DIAMETER_CM) + ' cm glas diameter' )
print('# Multiply outputs by 255 and clip between 0 and 255')
print('r=' + r_text)
print('g=' + g_text)
print('b=' + b_text)

# Plot figures
fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in rgb], columns=len(scale), **{'standalone': False})
ax_scale.xaxis.set_label_text(unit_name)
ax_scale.xaxis.set_ticks_position('bottom')

fig_model = plt.figure()
ax_model = fig_model.subplots(1)
ax_model.set_title(unit_name + ' to sRGB Model for ' + '{:.1f}'.format(BEER_GLAS_DIAMETER_CM) + ' cm Glas Diameter')
ax_model.xaxis.set_label_text(unit_name)
ax_model.yaxis.set_label_text('Intensity')

def plot_channel(values, new_values, color, label):
    r2 = calc_r2(values, new_values)
    ax_model.plot(scale, new_values, color=color, label=label + ' Fit ' +  ' (R²=' + '%.3f'%r2 + ')', linestyle=':')    
    ax_model.plot(scale, values, color=color, label=label + ' Data')

plot_channel(r, r_new, '#ff0000', 'R')
plot_channel(g, g_new, '#00ff00', 'G')
plot_channel(b, b_new, '#0000ff', 'B')
ax_model.legend()