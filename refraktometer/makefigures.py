#!/usr/bin/env python3
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import iqr

def correct_bx(bx, wcf):
    return bx / wcf

# Alcohol content estimation and Plato/SG conversion implemented according to
# G. Spedding. "Alcohol and Its Measurement". In: Brewing Materials and Processes. Elsevier,
# 2016, S. 123-149. DOI: 10.1016/b978-0-12-799954-8.00007-1.

def sg_to_p(sg):
    return sg**2 * -205.347 + 668.72 * sg - 463.37

def p_to_sg(p):
    return p / (258.6 - (p / 258.2 * 227.1)) + 1.0

def calc_re(oe, ae):
    return 0.1948 * oe + 0.8052 * ae

def calc_abw(oe, re):
    return (oe - re) / (2.0665 - (1.0665 * oe / 100.0))

def calc_abv(abw, fg):
    return abw * fg / 0.7907

def calc_abv_simple(oe, ae):
    return calc_abv(calc_abw(oe, calc_re(oe, ae)), p_to_sg(ae))

# Bonham (Standard) correlation function implemented according to:
# Louis K. Bonham. "The Use of Handheld Refractometers by Homebrewers".
# In: Zymurgy 24.1 (2001), S. 43-46.

def cor_bonham(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    fg = 1.001843 - 0.002318474 * oe - 0.000007775 * oe**2 - \
        0.000000034 * oe**3 + 0.00574 * bxf + \
        0.00003344 * bxf**2 + 0.000000086 * bxf**3
    return oe, sg_to_p(fg), fg

# Gardner correlation function implemented according to:
# Louis K. Bonham. "The Use of Handheld Refractometers by Homebrewers".
# In: Zymurgy 24.1 (2001), S. 43-46.

def cor_gardner(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    ae = 1.53 * bxf - 0.59 * oe
    return oe, ae, p_to_sg(ae)

# Gossett correlation function implemented according to:
# James M. Gossett. Derivation and Explanation of the Brix-Based Calculator For Estimating
# ABV in Fermenting and Finished Beers. 2012.
# URL: http://www.ithacoin.com/brewing/Derivation.htm

def abw_gosett(bxi, bxf, wcf):
    k = 0.445
    c = 100.0 * (bxi - bxf) / (100.0 - 48.4 * k - 0.582 * bxf)
    return 48.4 * c / (100 - 0.582 * c)

# The Gossett correlation is for abw and not fg. For abv calculation Gossett utilizes the
# Bonham correlation. Here the fg is derived from the abw equation instead.
def cor_gossett(bxi, bxf, wcf):
    abw = abw_gosett(bxi, bxf, wcf)
    ae = bxi - (abw * (2.0665 - 1.0665 * bxi / 100.0)) / 0.8052
    return bxi, ae, p_to_sg(ae)

# Novotný correlation functions implemented according to:
# Petr Novotný. Počítáme: Nová korekce refraktometru. 2017.
# URL: http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html

def cor_novotny_linear(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = -0.002349 * oe + 0.006276 * bxfc + 1.0
    return oe, sg_to_p(fg), fg

def cor_novotny_quadratic(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * bxfc + \
        2.916 * 10.0**-5 * bxfc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * bxfc + 1.0
    return oe, sg_to_p(fg), fg

# Terrill correlation functions implemented according to:
# Sean Terrill. Refractometer FG Results. 2011.
# URL: http://seanterrill.com/2011/04/07/refractometer-fg-results/

def cor_terrill_linear(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.0 - 0.000856829 * oe + 0.00349412 * bxfc
    return oe, sg_to_p(fg), fg

def cor_terrill_cubic(bxi, bxf, wcf):
    oe = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    fg = 1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * bxfc - \
        0.00127169 * bxfc**2 + 0.0000632929 * bxfc**3
    return oe, sg_to_p(fg), fg

# Sean Terrill's website issues. 2020.
# URL: https://www.reddit.com/r/Homebrewing/comments/bs3af9/sean_terrills_website_issues

def cor_novotrill(bxi, bxf, wcf):
    oe1, ae1, fg1 = cor_terrill_linear(bxi, bxf, wcf)
    oe2, ae2, fg2 = cor_novotny_linear(bxi, bxf, wcf)
    fg_mean = (fg1 + fg2) / 2.0
    fg = np.where(fg_mean < 1.014, fg1, fg2)
    return oe1, sg_to_p(fg), fg

class RefracModel:
    def __init__(self, name, short_name, cor_model):
        self.name = name
        self.short_name = short_name
        self.cor_model = cor_model

    def calc_ae(self, bxi, bxf, wcf):
        oe, ae, fg = self.cor_model(bxi, bxf, wcf)
        return ae

refrac_models = [
    RefracModel('Bonham', 'BO', cor_bonham),
    RefracModel('Gardner', 'GA', cor_gardner),
    RefracModel('Gossett', 'GO', cor_gossett),
    RefracModel('Novotný Linear', 'NL', cor_novotny_linear),
    RefracModel('Novotný Quadratisch', 'NQ', cor_novotny_quadratic),        
    RefracModel('Terrill Kubisch', 'TK', cor_terrill_cubic),
    RefracModel('Terrill Linear', 'TL', cor_terrill_linear),
    RefracModel('Terrill+Novotný', 'TN', cor_novotrill)
]

#model_short_names = list(map(lambda model: model.short_name, refrac_models))

col_name_wcf = 'WCF'
col_name_fg = 'FG'
col_name_ae = 'AE'
col_name_bxi = 'BXI'
col_name_bxf = 'BXF'
col_name_measurement = 'Messung'
col_name_hydrometer = 'Bierspindel'
col_name_statistic = 'Statistik'

stats_caps = ['Max. Abweichung [g/100g]', 'Mittlere Abweichung [g/100g]', 'Standardabweichung [g/100g] ',
'Abweichungen < 0,25 g/100g [%]', 'Abweichungen < 0,50 g/100g [%]', 'Abweichungen < 1,00 g/100g [%]']

def calc_stats(devs):
    devs_abs = devs.abs()
    max = dev[devs_abs.idxmax()]
    mean = devs.mean()
    std = devs.std()
    below_point_one = devs_abs.le(0.25).sum() / len(devs_abs) * 100.0
    below_point_two = devs_abs.le(0.5).sum() / len(devs_abs) * 100.0
    below_point_five = devs_abs.le(1.0).sum() / len(devs_abs) * 100.0
    return [ max, mean, std, below_point_one, below_point_two, below_point_five ]   

data_ferm = pa.read_csv('data_fermentation.csv', delimiter=',')
data_ferm_dev = pa.DataFrame()
data_ferm_graph = pa.DataFrame()

data_ferm_graph[col_name_measurement] = list(range(1, data_ferm.shape[0] + 1))
data_ferm_graph[col_name_hydrometer] = data_ferm[col_name_ae]

for model in refrac_models:
    data_ferm_graph[model.name] = model.calc_ae(data_ferm[col_name_bxi], data_ferm[col_name_bxf], data_ferm[col_name_wcf])
    data_ferm_dev[model.name] = data_ferm_graph[model.name] - data_ferm[col_name_ae]

data_ferm_table = pa.DataFrame()
data_ferm_table[col_name_statistic] = ['Endabweichung [g/100g]'] + stats_caps

for model in refrac_models:
    last = data_ferm_dev.iloc[-1][model.name]
    dev = data_ferm_dev[model.name]
    data_ferm_table[model.short_name] = [ last ] + calc_stats(dev)

data_ferm_table.to_latex('table_fermentation.tex', index=False, float_format='%.1f', decimal=',')

plot_cols = 2
plot_rows = len(refrac_models) // plot_cols + len(refrac_models) % plot_cols

fig_ferm = plt.figure(constrained_layout=True, figsize=(8, 12))
axes = fig_ferm.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
for i, model in enumerate(refrac_models):
    plot_row = i // plot_cols
    plot_col = i % plot_cols
    if plot_rows > 1:
        ax = axes[plot_row][plot_col]
    else:
        ax = axes[plot_col]
    ax.set_ylim([2,18])
    ax.plot(data_ferm_graph[col_name_measurement], data_ferm_graph[col_name_hydrometer], label=col_name_hydrometer)
    ax.plot(data_ferm_graph[col_name_measurement], data_ferm_graph[model.name], label=model.name)
    r2 = r2_score(data_ferm_graph[col_name_hydrometer], data_ferm_graph[model.name])
    ax.set_title(model.name + ' (R²=' + '%.3f'%r2 + ')')
    ax.legend(loc='best')  
    ax.set_ylabel('Scheinb. Restex. [g/100g]')

fig_ferm.savefig('graph_fermentation.pdf', format='pdf')

default_wcf = 1.04

data_ae = pa.read_csv('data.csv', delimiter=',')
data_ae_abs = pa.DataFrame()
data_ae_dev = pa.DataFrame()

data_ae[col_name_ae] = np.where(np.isnan(data_ae[col_name_ae]), sg_to_p(data_ae[col_name_fg]), data_ae[col_name_ae])
data_ae[col_name_wcf] = np.where(np.isnan(data_ae[col_name_wcf]), default_wcf, data_ae[col_name_wcf])

data_ae_abs[col_name_hydrometer] = data_ae[col_name_ae]
for model in refrac_models:
    data_ae_abs[model.name] = model.calc_ae(data_ae[col_name_bxi], data_ae[col_name_bxf], data_ae[col_name_wcf])
    data_ae_dev[model.name] = data_ae_abs[model.name] - data_ae[col_name_ae]

filter_outliers = True
if filter_outliers == True:
    row_criteria = data_ae_dev.abs().max(axis=1)
    threshold = iqr(row_criteria) * 3
    print('Filter threshold is %.2f'% threshold)
    filter = row_criteria <= threshold
    data_ae_abs = data_ae_abs.where(filter).dropna()
    data_ae_dev = data_ae_dev.where(filter).dropna()

data_ae_table = pa.DataFrame()
data_ae_table[col_name_statistic] = stats_caps

for model in refrac_models:
    dev = data_ae_dev[model.name]
    data_ae_table[model.short_name] = calc_stats(dev)

data_ae_table.to_latex('table_ae.tex', index=False, float_format='%.1f', decimal=',')

fig_ae = plt.figure(constrained_layout=True, figsize=(8, 12))
axes = fig_ae.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
for i, model in enumerate(refrac_models):
    plot_row = i // plot_cols
    plot_col = i % plot_cols
    if plot_rows > 1:
        ax = axes[plot_row][plot_col]
    else:
        ax = axes[plot_col]
    data_ae_dev[model.name].plot.hist(density=True, xlim=[-1.5,1.5], bins=15, ax=ax)
    data_ae_dev[model.name].plot.density(ax=ax)
    r2 = r2_score(data_ae_abs[col_name_hydrometer], data_ae_abs[model.name])
    ax.set_title(model.name + ' (R²=' + '%.3f'%r2 + ')')
    ax.set_xlabel('Abw. scheinbarer Restextrakt [g/100g]')
    ax.set_ylabel('Dichte')

fig_ae.savefig('graph_ae.pdf', format='pdf')

#plt.show()