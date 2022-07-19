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
GLAS_DIAMETER_CM = 7.5
OBSERVER_NAME = 'CIE 1964 10 Degree Standard Observer'
ILLUMINANT_NAME = 'D65'
USE_EBC_SCALE = True
MAX_SCALE_VALUE = 80
POLY_DEGREE_R = 3
POLY_DEGREE_G = 3
POLY_DEGREE_B = 3

observer = colour.MSDS_CMFS[OBSERVER_NAME]
illuminant = colour.SDS_ILLUMINANTS[ILLUMINANT_NAME]
illuminant_xy = colour.CCS_ILLUMINANTS[OBSERVER_NAME][ILLUMINANT_NAME]

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
    if USE_EBC_SCALE == True:
        ebc = scale
        return eval(code)
    else:
        srm = scale
        return eval(code)

def fit_poly(channel, degree):
    idx = np.isfinite(channel)
    return np.polyfit(scale[idx], channel[idx], degree)

def cutoff_signal(signal):
    signal[(signal < 0.0)] = np.nan
    return signal

def cutoff_channel(rgb, channel):
    return cutoff_signal(np.array([i[channel] for i in rgb]))

def calc_r2(actual, predicted):
    idx = np.isfinite(actual) & np.isfinite(predicted)
    corr_matrix = np.corrcoef(actual[idx], predicted[idx])
    corr = corr_matrix[0, 1]
    r2 = corr**2
    return r2

def print_comment(text):
    print('# ' + text)

def print_tag_comment(tag, value):
    print_comment(tag + ': ' + value)

# Generate sRGB input data for model fit
scale = np.arange(start=0, stop=MAX_SCALE_VALUE+1,step=1)
wl = colour.SpectralShape(380, 780, 5).range()
rgb = []
for i in scale:
    srm = i * unit_conversion
    values = 10**(-(srm / 12.7) * (0.018747 * math.e**(-(wl - 430.0) / 13.374) + 0.98226 * math.e**(-(wl - 430.0) / 80.514)) * GLAS_DIAMETER_CM)
    xyz = colour.sd_to_XYZ(colour.SpectralDistribution(values, wl), cmfs=observer, illuminant=illuminant) / 100.0
    rgb.append(colour.XYZ_to_sRGB(xyz, illuminant=illuminant_xy))

# Fit data
r = cutoff_channel(rgb, 0)
r_coeff = fit_poly(r, POLY_DEGREE_R)
g = cutoff_channel(rgb, 1)
g_coeff = fit_poly(g, POLY_DEGREE_G)
b = cutoff_channel(rgb, 2)
b_coeff = fit_poly(b, POLY_DEGREE_B)

# Generate and compile model code
r_text, r_code = compile_poly(r_coeff, unit_name)
g_text, g_code = compile_poly(g_coeff, unit_name)
b_text, b_code = compile_poly(b_coeff, unit_name)

# Generate sRGB output
r_new = cutoff_signal(eval_poly(r_code))
g_new = cutoff_signal(eval_poly(g_code))
b_new = cutoff_signal(eval_poly(b_code))

for i in zip(r_new, g_new, b_new):
    rgb.append(i)

# Print model
print_comment(unit_name + ' to sRGB model, multiply outputs by 255 and clip between 0 and 255')
print_tag_comment('glas diameter', str(GLAS_DIAMETER_CM) + ' cm')
print_tag_comment('observer', OBSERVER_NAME)
print_tag_comment('illuminant', ILLUMINANT_NAME)
print_tag_comment('scale', str(MAX_SCALE_VALUE) + ' ' + unit_name)
print('r=' + r_text)
print('g=' + g_text)
print('b=' + b_text)

# Plot figures
fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in rgb], columns=len(scale), **{'standalone': False})
ax_scale.xaxis.set_label_text(unit_name)
ax_scale.xaxis.set_ticks_position('bottom')

fig_model = plt.figure()
ax_model = fig_model.subplots(1)
ax_model.set_title(unit_name + ' to sRGB Model')
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