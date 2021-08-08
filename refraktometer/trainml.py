#!/usr/bin/env python3
# Refractometer Correlation Model Generator
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import numpy as np
import pandas as pa
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

def correct_ri(ri, wcf):
    return ri / wcf

def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

def plato_to_sg(se):
    return 1.0 + (se / (258.6 - ((se / 258.2) * 227.1)))

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(rii, rif, wcf):
    riic = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return riic, rifc, -0.002349 * riic + 0.006276 * rifc + 1.0

def cor_novotny_quadratic(rii, rif, wcf):
    riic = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return riic, rifc, 1.335 * 10.0**-5 * riic**2 - \
        3.239 * 10.0**-5 * riic * rifc + \
        2.916 * 10.0**-5 * rifc**2 - \
        2.421 * 10.0**-3 * riic + \
        6.219 * 10.0**-3 * rifc + 1.0

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(rii, rif, wcf):
    riic = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)          
    return riic, rifc, 1.0 - 0.000856829 * riic + 0.00349412 * rifc

def cor_terrill_cubic(rii, rif, wcf):
    riic = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)         
    return riic, rifc, 1.0 - 0.0044993 * riic + 0.000275806 * riic**2 - \
        0.00000727999 * riic**3 + 0.0117741 * rifc - \
        0.00127169 * rifc**2 + 0.0000632929 * rifc**3

models = [ cor_novotny_linear, cor_novotny_quadratic, cor_terrill_linear, cor_terrill_cubic]
data = list()

wcf = 1.0
for wcf_part in range(0, 7):
    wcf = 1.0 + wcf_part / 100.0
    for rii in range(21, 6, -1):
        for rif in range(rii-1, 4, -1):
            for model in models:
                riic, rifc, fg =  model(rii, rif, wcf)
                if fg >= 1.0:
                    data.append([riic, rifc, fg])

col_name_oe = 'OE'
col_name_rii = 'RII'
col_name_rif = 'RIF'
col_name_fg = 'FG'
col_name_ae = 'AE'

measured_data = pa.read_csv('data.csv', delimiter=',')
median_wcf = (measured_data[col_name_rii] / measured_data[col_name_oe]).median()
measured_data[col_name_rii] = measured_data[col_name_rii] / median_wcf
measured_data[col_name_rif] = measured_data[col_name_rii] / median_wcf
measured_data[col_name_fg] = plato_to_sg(measured_data[col_name_ae])
threshold = median_absolute_error(measured_data[col_name_oe], measured_data[col_name_rii])
measured_data = measured_data[(abs(measured_data[col_name_rii] - measured_data[col_name_oe]) <= threshold)]
measured_data = measured_data[[col_name_rii, col_name_rif, col_name_fg]]

data = np.concatenate((data, measured_data.to_numpy()), axis=0)

df = pa.DataFrame(data, columns=['RII', 'RIF', 'FG'])
reg_xdata = df[[col_name_rii, col_name_rif]].to_numpy()
reg_ydata = df[col_name_fg].to_numpy()
reg = LinearRegression().fit(reg_xdata, reg_ydata)
print('fg = ' + '%.6f'%reg.intercept_ + ' + ' + '%.6f'%reg.coef_[0] + ' * riic + ' + '%.6f'%reg.coef_[1] +  ' * rifc' )
