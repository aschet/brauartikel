#!/usr/bin/env python3
# Refractometer Correlation Model Evaluation
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import pandas as pa
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, r2_score

# ABV = alcohol by volume in %
# ABW = alcohol bei weight in %
# AE = apparent extract in °P
# FG = final gravity in SG
# OE = original extract in °P
# RII = initial refractive index in °Bx
# RIF = final refractive index in °Bx
# SG = specific gravity
# WCF = wort correction factor

default_wcf = 1.04
recalc_default_wcf = True
measurement_specific_wcf = False
filter_outliers = True
reference_filter = 'PBA-B M'
plot_ae_dev = True
plot_abv_dev = True

def correct_ri(ri, wcf):
    return ri / wcf

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def plato_to_sg(se):
    return 1.0 + (se / (258.6 - ((se / 258.2) * 227.1)))

# https://www.brewersjournal.info/science-basic-beer-alcohol-extract-determinations/
def calc_abw(oe, ae):
    return (0.8052 * (oe - ae)) / (2.0665 - (1.0665 * oe / 100.0))

# https://www.brewersjournal.info/science-basic-beer-alcohol-extract-determinations/
def calc_abv(abw, fg):
    return abw * fg / 0.7907

def calc_abv_simple(oe, ae):
    return calc_abv(calc_abw(oe, ae), plato_to_sg(ae))

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_bonham(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    return oe, sg_to_plato(1.001843 - 0.002318474 * oe - 0.000007775 * oe**2 - \
        0.000000034 * oe**3 + 0.00574 * rif + \
        0.00003344 * rif**2 + 0.000000086 * rif**3)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_gardner(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    return oe, 1.53 * rif - 0.59 * oe

# http://www.ithacoin.com/brewing/Derivation.htm
def calc_abv_gosett(rii, rif, wcf):
    k = 0.445
    c = 100.0 * (rii - rif) / (100.0 - 48.4 * k - 0.582 * rif)
    abw = 48.4 * c / (100 - 0.582 * c)
    oe, ae = cor_bonham(rii, rif, wcf)
    fg = plato_to_sg(ae)
    return calc_abv(abw, fg)

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return oe, sg_to_plato(-0.002349 * oe + 0.006276 * rifc + 1.0)

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html     
def cor_novotny_quadratic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return oe, sg_to_plato(1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * rifc + \
        2.916 * 10.0**-5 * rifc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * rifc + 1.0)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)          
    return oe, sg_to_plato(1.0 - 0.000856829 * oe + 0.00349412 * rifc)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_cubic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)         
    return oe, sg_to_plato(1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * rifc - \
        0.00127169 * rifc**2 + 0.0000632929 * rifc**3)

def print_stats(name, stats, is_deviation):
    full_name = name
    if is_deviation == True:
        full_name += ' Deviation'
    print(full_name + ' Statistics:')
    print(stats)
    print()

class RefracModel:
    def __init__(self, name, cor_model, abv_model=None):
        self.name = name
        self.cor_model = cor_model
        self.abv_model = abv_model

    def calc_ae(self, rii, rif, wcf):
        _, ae = self.cor_model(rii, rif, wcf)
        return ae

    def calc_abv(self, rii, rif, wcf):
        if not (self.abv_model is None):
            return self.abv_model(rii, rif, wcf)
        else:
            oe, ae = self.cor_model(rii, rif, wcf)
            return calc_abv_simple(oe, ae)

refrac_models = [
    RefracModel('Terrill Linear', cor_terrill_linear),
    RefracModel('Terrill Cubic', cor_terrill_cubic),
    RefracModel('Novotny Linear', cor_novotny_linear),
    RefracModel('Novotny Quadratic', cor_novotny_quadratic),
    RefracModel('Bonham', cor_bonham),
    RefracModel('Gardner', cor_gardner),
    RefracModel('Gossett', cor_bonham, calc_abv_gosett)
]

model_names = list(map(lambda model: model.name, refrac_models))

col_name_abv = 'ABV'
col_name_wcf = 'WCF'
col_name_oe = 'OE'
col_name_ae = 'AE'
col_name_rii = 'RII'
col_name_rif = 'RIF'
col_name_reference = 'Reference'
row_name_square = 'r2score'

def model_col_name(section, name):
    return section + ' ' + name

data = pa.read_csv('data.csv', delimiter=',')
data_abv_dev = pa.DataFrame()
data_ae_dev = pa.DataFrame()

if len(reference_filter) > 0:
    data = data[data[col_name_reference] == reference_filter] 

data[col_name_wcf] = data[col_name_rii] / data[col_name_oe]
data[col_name_abv] = calc_abv_simple(data[col_name_oe], data[col_name_ae])

wcf_stats = data[col_name_wcf].describe()
if recalc_default_wcf == True:
    default_wcf = wcf_stats['75%']
print_stats(col_name_wcf, wcf_stats, False)

if measurement_specific_wcf == False:
    data[col_name_wcf] = default_wcf

if filter_outliers == True:
    riic = correct_ri(data[col_name_rii], data[col_name_wcf])
    mae = median_absolute_error(data[col_name_oe], riic) + 1.776357e-15
    print('Filtering ' + col_name_rii + ' outliers over ' + str(mae))
    print()
    data = data[ (abs(data[col_name_oe] - riic) <= mae)]

for model in refrac_models:
    model_col_name_ae = model_col_name(col_name_ae, model.name)
    data[model_col_name_ae] = model.calc_ae(data[col_name_rii], data[col_name_rif], data[col_name_wcf])
    data_ae_dev[model.name] = data[col_name_ae] - data[model_col_name_ae]
    model_col_name_abv = model_col_name(col_name_abv, model.name)
    data[model_col_name_abv] = model.calc_abv(data[col_name_rii], data[col_name_rif], data[col_name_wcf])   
    data_abv_dev[model.name] = data[col_name_abv] - data[model_col_name_abv]
 
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
    fig.suptitle('Refractometer Correlation Model Evaluation: ' + col_name)
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
        data_dev[model_names[i]].plot.hist(density=True, xlim=[-1,1], ax=ax_desnity)
        data_dev[model_names[i]].plot.density(ax=ax_desnity)  
    
    return fig

if plot_ae_dev:
    fig_ae = plot_devs(col_name_ae, data_ae_dev, stats_ae_dev)
    fig_ae.savefig('stats_ae_dev.svg')
if plot_abv_dev:
    fig_abv = plot_devs(col_name_abv, data_abv_dev, stats_abv_dev)
    fig_abv.savefig('stats_abv_dev.svg')
if plot_ae_dev == True or plot_abv_dev == True:
    plt.show()