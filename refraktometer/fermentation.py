#!/usr/bin/env python3
# Refractometer Correlation Model Evaluation: Active Fermentation
# Copyright 2021 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import pandas as pa
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# BXF = final refractometer reading in °Bx
# BXI = initial refractometer reading in °Bx
# FG = final gravity in SG
# OE = original extract in °P
# RE = real extract in °P
# SG = specific gravity
# WCF = wort correction factor

default_wcf = 1.0

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

def cor_from_abw(abw, bxi, wcf):
    oe = correct_bx(bxi, wcf)
    ae = oe - (abw * (2.0665 - 1.0665 * oe / 100.0)) / 0.8052
    return oe, ae, p_to_sg(ae)

# The Gossett correlation is for abw and not fg. For abv calculation Gossett utilizes the
# Bonham correlation. Here the fg is derived from the abw equation instead.
def cor_gossett(bxi, bxf, wcf):
    return cor_from_abw(abw_gosett(bxi, bxf, wcf), bxi, wcf)

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

class RefracModel:
    def __init__(self, name, cor_model):
        self.name = name
        self.cor_model = cor_model

    def calc_ae(self, bxi, bxf, wcf):
        oe, ae, fg = self.cor_model(bxi, bxf, wcf)
        return ae

refrac_models = [
    RefracModel('Gardner', cor_gardner),
    RefracModel('Bonham', cor_bonham),
    RefracModel('Terrill Linear', cor_terrill_linear),
    RefracModel('Terrill Kubisch', cor_terrill_cubic),
    RefracModel('Gossett', cor_gossett),
    RefracModel('Novotný Linear', cor_novotny_linear),
    RefracModel('Novotný Quadratisch', cor_novotny_quadratic)
]

model_names = list(map(lambda model: model.name, refrac_models))

col_name_ae = 'AE'
col_name_bxi = 'BXI'
col_name_bxf = 'BXF'
col_name_measurement = 'Measurement'
col_name_hydrometer = 'Aräometer'

def model_col_name(section, name):
    return section + ' ' + name

data = pa.read_csv('fermentation_data.csv', delimiter=',')
data_dev = pa.DataFrame()
data_graph = pa.DataFrame()

data_graph[col_name_measurement] = list(range(1, data.shape[0] + 1))
data_graph[col_name_hydrometer] = data[col_name_ae]

for model in refrac_models:
    data_graph[model.name] = model.calc_ae(data[col_name_bxi], data[col_name_bxf], default_wcf)
    data_dev[model.name] = data_graph[model.name] - data[col_name_ae]

data_graph.to_csv('fermentation_graph.csv', index=False)

print("Final deviation:")
print(data_dev.iloc[-1])
print()
print("Deviation statistics:")
stats = data_dev.describe()
stats.loc['rscore'] = list(map(lambda name: r2_score(data_graph[col_name_hydrometer], data_graph[name]), model_names))
print(stats)

fig = plt.figure(constrained_layout=True, figsize=(5, 5))
#fig.suptitle('Novotny Dataset: Active Fermentation with OE of 17 °P')
ax = fig.subplots(1, 1)
ax.set_xlabel('Messung')
ax.set_ylabel('Scheinbarer Restextrakt (°P)')
ax.plot(data_graph[col_name_measurement], data_graph[col_name_hydrometer], label=col_name_hydrometer, marker='.')
for model in refrac_models:
    ax.plot(data_graph[col_name_measurement], data_graph[model.name], linestyle=':', label=model.name)
ax.legend(loc='best')

plt.savefig("fermentation_graph.pdf", format="pdf")
plt.show()