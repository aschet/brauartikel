#!/usr/bin/env python3
# Refractometer Correlation Model Evaluation
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import pandas as pa
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import iqr

# ABV = alcohol by volume in %
# ABW = alcohol bei weight in %
# AE = apparent extract in °P
# FG = final gravity in SG
# OE = original extract in °P
# RE = real extract in °P
# RII = initial refractive index in °Bx
# RIF = final refractive index in °Bx
# SG = specific gravity
# WCF = wort correction factor

default_wcf = 1.04
recalc_default_wcf = False
measurement_specific_wcf = False
filter_oe_outliers = True
reference_filter = 'PBA-B M'
refractometer_filter = 'ORA 32BA'
plot_ae_dev = False
plot_abv_dev = True

def correct_ri(ri, wcf):
    return ri / wcf

# Alcohol content estimation and Plato/SG conversion implemented according to
# G. Spedding. "Alcohol and Its Measurement". In: Brewing Materials and Processes. Elsevier,
# 2016, S. 123-149. DOI: 10.1016/b978-0-12-799954-8.00007-1

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

def abv_common(cor_model, rii, rif, wcf):
    oe, ae, fg = cor_model(rii, rif, wcf)
    return calc_abv(calc_abw(oe, calc_re(oe, ae)), fg)

# Bonham correleation function implemented according to:
# Louis K. Bonham. "The Use of Handheld Refractometers by Homebrewers".
# In: Zymurgy 24.1 (2001), S. 43-46.
def cor_bonham(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    fg = 1.001843 - 0.002318474 * oe - 0.000007775 * oe**2 - \
        0.000000034 * oe**3 + 0.00574 * rif + \
        0.00003344 * rif**2 + 0.000000086 * rif**3
    return oe, sg_to_p(fg), fg

# Gardner correleation function implemented according to:
# Louis K. Bonham. "The Use of Handheld Refractometers by Homebrewers".
# In: Zymurgy 24.1 (2001), S. 43-46.
def cor_gardner(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    ae = 1.53 * rif - 0.59 * oe
    return oe, ae, p_to_sg(ae)

# Gossett correleation function implemented according to:
# James M. Gossett. Derivation and Explanation of the Brix-Based Calculator For Estimating
# ABV in Fermenting and Finished Beers. 2012.
# URL: http://www.ithacoin.com/brewing/Derivation.htm
def abv_gosett(cor_model, rii, rif, wcf):
    k = 0.445
    c = 100.0 * (rii - rif) / (100.0 - 48.4 * k - 0.582 * rif)
    abw = 48.4 * c / (100 - 0.582 * c)
    oe, ae, fg = cor_model(rii, rif, wcf)
    return calc_abv(abw, fg)

# Novotný correleation functions implemented according to:
# Petr Novotný. Počítáme: Nová korekce refraktometru. 2017.
# URL: http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    fg = -0.002349 * oe + 0.006276 * rifc + 1.0
    return oe, sg_to_p(fg), fg

def cor_novotny_quadratic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    fg = 1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * rifc + \
        2.916 * 10.0**-5 * rifc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * rifc + 1.0
    return oe, sg_to_p(fg), fg

# Terrill correleation functions implemented according to:
# Sean Terrill. Refractometer FG Results. 2011.
# URL: http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    fg = 1.0 - 0.000856829 * oe + 0.00349412 * rifc
    return oe, sg_to_p(fg), fg

def cor_terrill_cubic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    fg = 1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * rifc - \
        0.00127169 * rifc**2 + 0.0000632929 * rifc**3
    return oe, sg_to_p(fg), fg

# Obtained by fit into data generated from Terrill and Novotný equations
def cor_ascher(rii, rif, wcf):
    riic = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    fg = 0.991845 + -0.001637 * riic + 0.006053 * rifc
    return riic, sg_to_p(fg), fg

def print_stats(name, stats, is_deviation):
    full_name = name
    if is_deviation == True:
        full_name += ' Deviation'
    print(full_name + ' Statistics:')
    print(stats)
    print()

class RefracModel:
    def __init__(self, name, cor_model, abv_model):
        self.name = name
        self.cor_model = cor_model
        self.abv_model = abv_model

    def calc_ae(self, rii, rif, wcf):
        oe, ae, fg = self.cor_model(rii, rif, wcf)
        return ae

    def calc_abv(self, rii, rif, wcf):
        return self.abv_model(self.cor_model, rii, rif, wcf)

refrac_models = [
    RefracModel('Terrill Linear', cor_terrill_linear, abv_common),
    RefracModel('Terrill Cubic', cor_terrill_cubic, abv_common),
    RefracModel('Novotny Linear', cor_novotny_linear, abv_common),
    RefracModel('Novotny Quadratic', cor_novotny_quadratic, abv_common),
    RefracModel('Bonham', cor_bonham, abv_common),
    RefracModel('Gardner', cor_gardner, abv_common),
    RefracModel('Gossett', cor_bonham, abv_gosett),
    RefracModel('Ascher', cor_ascher, abv_common)    
]

model_names = list(map(lambda model: model.name, refrac_models))

col_name_abv = 'ABV'
col_name_wcf = 'WCF'
col_name_oe = 'OE'
col_name_ae = 'AE'
col_name_rii = 'RII'
col_name_rif = 'RIF'
col_name_reference = 'Reference'
col_name_refractometer = 'Refractometer'
row_name_square = 'r2score'

def model_col_name(section, name):
    return section + ' ' + name

data = pa.read_csv('data.csv', delimiter=',')
data_abv_dev = pa.DataFrame()
data_ae_dev = pa.DataFrame()

if len(reference_filter) > 0:
    data = data[data[col_name_reference] == reference_filter] 

if len(refractometer_filter) > 0:
    data = data[data[col_name_refractometer] == refractometer_filter]    

data[col_name_wcf] = data[col_name_rii] / data[col_name_oe]
data[col_name_abv] = calc_abv_simple(data[col_name_oe], data[col_name_ae])

wcf_stats = data[col_name_wcf].describe()
if recalc_default_wcf == True:
    default_wcf = wcf_stats['75%']
    print('Updating default WCF to ' + str(default_wcf) + '\n')
print_stats(col_name_wcf, wcf_stats, False)

if measurement_specific_wcf == False:
    data[col_name_wcf] = default_wcf

if filter_oe_outliers == True:
    riic = correct_ri(data[col_name_rii], data[col_name_wcf])
    rii_dev = riic - data[col_name_oe]
    threshold = abs(iqr(rii_dev) * 1.5)
    print('Filtering ' + col_name_rii + ' outliers over ' + str(threshold) + '\n')
    data = data[(abs(riic - data[col_name_oe]) <= threshold)]

for model in refrac_models:
    model_col_name_ae = model_col_name(col_name_ae, model.name)
    data[model_col_name_ae] = model.calc_ae(data[col_name_rii], data[col_name_rif], data[col_name_wcf])
    data_ae_dev[model.name] = data[model_col_name_ae] - data[col_name_ae]
    model_col_name_abv = model_col_name(col_name_abv, model.name)
    data[model_col_name_abv] = model.calc_abv(data[col_name_rii], data[col_name_rif], data[col_name_wcf])   
    data_abv_dev[model.name] = data[model_col_name_abv] - data[col_name_abv]
 
data.to_csv('data_eval.csv', index=False)

def create_stats(devs, col_name):
    stats = devs.describe()
    stats.loc[row_name_square] = list(map(lambda name: r2_score(data[col_name], data[model_col_name(col_name, name)]), model_names))
    return stats

stats_ae_dev = create_stats(data_ae_dev, col_name_ae)
stats_ae_dev.to_csv('stats_ae_dev.csv', index=True)
stats_abv_dev = create_stats(data_abv_dev, col_name_abv)
stats_abv_dev.to_csv('stats_abv_dev.csv', index=True)

print_stats(col_name_ae, stats_ae_dev, True)
print_stats(col_name_abv, stats_abv_dev, True)

def plot_devs(col_name, data_dev, stats_dev):
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    reference = ', '.join(list(data[col_name_reference].unique()))
    refractometer = ', '.join(list(data[col_name_refractometer].unique()))
    fig.suptitle('Refractometer Correlation Model Comparison: ' + col_name + ' (' + reference + ' with ' + refractometer + ', ' + str(data_dev.shape[0]) + ' Measurements)')
    subfigs = fig.subfigures(1, 2)

    ax_quantils = subfigs[0].subplots(1, 1)
    ax_quantils.axhline(0.0, linestyle='--', c='#000000', linewidth=1)
    dev_caption = col_name + ' Deviation at WCF='
    if measurement_specific_wcf == True:
        dev_caption += 'Auto'
    else:
        dev_caption += '%.2f'%default_wcf
    ax_quantils.set_ylabel(dev_caption)
    data_dev.boxplot(model_names, ax=ax_quantils, rot=45, grid=False, showmeans=True)

    cols = 2
    rows = len(refrac_models) // cols + len(refrac_models) % cols
    ax_densities = subfigs[1].subplots(rows, cols, sharex=True, sharey=True)
    for i, model in enumerate(refrac_models):
        row = i // cols
        col = i % cols
        if rows > 1:
            ax_desnity = ax_densities[row][col]
        else:
            ax_desnity = ax_densities[col]
        rsquare = stats_dev[model.name][row_name_square]
        ax_desnity.set_title(model.name + ' (R²=' + '%.3f'%rsquare + ')')    
        ax_desnity.set_xlabel(dev_caption)
        data_dev[model.name].plot.hist(density=True, xlim=[-1,1], ax=ax_desnity)
        try:
            data_dev[model.name].plot.density(ax=ax_desnity)
        except:
            pass
    
    return fig

if plot_ae_dev:
    fig_ae = plot_devs(col_name_ae, data_ae_dev, stats_ae_dev)
    fig_ae.savefig('stats_ae_dev.svg')
if plot_abv_dev:
    fig_abv = plot_devs(col_name_abv, data_abv_dev, stats_abv_dev)
    fig_abv.savefig('stats_abv_dev.svg')
if plot_ae_dev == True or plot_abv_dev == True:
    plt.show()