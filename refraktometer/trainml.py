#!/usr/bin/env python3
# Refractometer Correlation Model Generator
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import numpy as np
import pandas as pa
from sklearn.linear_model import LinearRegression

def correct_bx(bx, wcf):
    return bx / wcf

# Novotný correleation functions implemented according to:
# Petr Novotný. Počítáme: Nová korekce refraktometru. 2017.
# URL: http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = -0.002349 * oe + 0.006276 * bxfc + 1.0
    return oe, bxfc, fg

def cor_novotny_quadratic(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * bxfc + \
        2.916 * 10.0**-5 * bxfc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * bxfc + 1.0
    return oe, bxfc, fg

# Terrill correleation functions implemented according to:
# Sean Terrill. Refractometer FG Results. 2011.
# URL: http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.0 - 0.000856829 * oe + 0.00349412 * bxfc
    return oe, bxfc, fg

def cor_terrill_cubic(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * bxfc - \
        0.00127169 * bxfc**2 + 0.0000632929 * bxfc**3
    return oe, bxfc, fg

models = [ cor_novotny_linear, cor_novotny_quadratic, cor_terrill_linear, cor_terrill_cubic]
data = list()

wcf = 1.0
for wcf_part in range(0, 7):
    wcf = 1.0 + wcf_part / 100.0
    for bxi in range(21, 6, -1):
        for bxf in range(bxi-1, 4, -1):
            for model in models:
                oe, bxfc, fg = model(bxi, bxf, wcf)
                if fg >= 1.0:
                    data.append([oe, bxfc, fg])

col_name_bxi = 'BXI'
col_name_bxf = 'BXF'
col_name_fg = 'FG'

df = pa.DataFrame(data, columns=[col_name_bxi, col_name_bxf, col_name_fg])

def print_equation(feature_name, variable_name):
    reg_xdata = df[[col_name_bxi, col_name_bxf]].to_numpy()
    reg_ydata = df[feature_name].to_numpy()
    reg = LinearRegression().fit(reg_xdata, reg_ydata)
    print(variable_name + ' = ' + '%.6f'%reg.intercept_ + ' + ' + '%.6f'%reg.coef_[0] + ' * bxic + ' + '%.6f'%reg.coef_[1] +  ' * bxfc')

print_equation(col_name_fg, 'fg')
