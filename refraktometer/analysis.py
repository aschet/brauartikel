#!/usr/bin/env python3
from numpy.core.fromnumeric import mean
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt

# Refractometer Correlation Function Evaluation
# greetings, Thomas Ascher

# aat = apparent attenuation in %
# ae = apparent extract in °P
# oe = original extract in °P
# rii = refractive index initial
# rif = refractive index final
# sg = specific gravity
# wcf = wort correction factor

wcf = 1.0
filter_outliers = False

def correct_ri(ri, wcf):
    return ri / wcf

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def plato_to_sg(se):
    return 1.0 + (se / (258.6 - ((se / 258.2) * 227.1)))

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_bonham(rii, rif, wcf):
    rifc = correct_ri(rii, wcf)
    return sg_to_plato(1.001843 - 0.002318474 * rifc - 0.000007775 * rifc**2 - \
        0.000000034 * rifc**3 + 0.00574 * rif + \
        0.00003344 * rif**2 + 0.000000086 * rif**3)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_gardner(rii, rif, wcf):
    rifc = correct_ri(rii, wcf)
    return 1.53 * rif - 0.59 * rifc

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return sg_to_plato(-0.002349 * oe + 0.006276 * rifc + 1.0)

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html     
def cor_novotny_quadratic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)
    return sg_to_plato(1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * rifc + \
        2.916 * 10.0**-5 * rifc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * rifc + 1.0)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)           
    return sg_to_plato(1.0 - 0.000856829 * oe + 0.00349412 * rifc)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_cubic(rii, rif, wcf):
    oe = correct_ri(rii, wcf)
    rifc = correct_ri(rif, wcf)          
    return sg_to_plato(1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * rifc - \
        0.00127169 * rifc**2 + 0.0000632929 * rifc**3)

cor_models = [
    ('Bonham', cor_bonham, '#a9f693'),
    ('Gardner', cor_gardner, '#00c295'),
    ('Novotny Linear', cor_novotny_linear, '#5f5959'),
    ('Novotny Quadratic', cor_novotny_quadratic, '#ff0043'),
    ('Terrill Linear', cor_terrill_linear, '#ff795b'),
    ('Terrill Cubic', cor_terrill_cubic, '#fddb85'),    
]

# Revisiting ABV Calculations, Zymurgy July/August 2019 p. 48
def calc_abv(oe, ae):
    og = plato_to_sg(oe)
    fg = plato_to_sg(ae)
    return fg * (5118.0 * (og**2 - fg**2) + 16755.0 * (fg - og)) / (8.739 * og**4 \
        - 57.22 * og**3 + 89.09 * og**2 + 14.95 + og - 105.99)

def col_name(section, name):
    return section + ' ' + name

def col_name_ae(name):
    return col_name('AE', name)

def col_name_ae_err(name):
    return col_name('AE Error', name)

def col_name_abv(name):
    return col_name('ABV', name)

def col_name_abv_err(name):
    return col_name('ABV Error', name)

data = pa.read_csv("data.csv", delimiter=',')

col_name_riic_err = 'RIIC Error'
data[col_name_riic_err] = data.apply(lambda row: correct_ri(row.RII, wcf) - row.OE, axis=1)
if filter_outliers == True:
    riic_err_threshold = data[col_name_riic_err].std()
    print('Filtering outliers over ' + col_name_riic_err + ': ' + str(riic_err_threshold))
    print()
    data = data[(abs(data[col_name_riic_err]) <= riic_err_threshold)]

data['AAT'] = (data['OE'] - data['AE']) * 100.0 / data['OE']
data['ABV'] = data.apply(lambda row: calc_abv(row.OE, row.AE), axis=1)
wcf_col_name = 'WCF'
data[wcf_col_name] = data['RII'] / data['OE']

def add_cor_model_data(name, functor):
    data[col_name_ae(name)] = data.apply(lambda row: functor(row.RII, row.RIF, wcf), axis=1)
    data[col_name_ae_err(name)] = data.apply(lambda row: row[col_name_ae(name)] - row.AE, axis=1)
    data[col_name_abv(name)] = data.apply(lambda row: calc_abv(correct_ri(row.RII, wcf), row[col_name_ae(name)]), axis=1)
    data[col_name_abv_err(name)] = data.apply(lambda row: row[col_name_abv(name)] - row.ABV, axis=1)

name_bonham = 'Bonham'
name_gardner = 'Gardner'
name_novotny_linear = 'Novotny Linear'
name_novotny_quadratic = 'Novotny Quadratic'
name_terrill_linear = 'Terrill Linear'
name_terrill_cubic = 'Terrill Cubic'

for model in cor_models:
    add_cor_model_data(model[0], model[1])

wcf_list = data[wcf_col_name]
wcf_stats = pa.DataFrame([(wcf_list.mean(), wcf_list.min(), wcf_list.max(), wcf_list.mad(), wcf_list.std())], columns = ['WCF Mean', 'WCF Min', 'WCF Max', 'WCF MAD', 'WCF STD'])
print(wcf_stats)
print()

def calc_model_stats(name):
    abv_err = data[col_name_abv_err(name)]
    abv_err_abs = abv_err.abs()
    abv_err_below = abv_err_abs.le(0.5).sum() / len(abv_err_abs) * 100.0
    return name, abv_err_abs.mean(), abv_err[abv_err_abs.idxmin()], abv_err[abv_err_abs.idxmax()], abv_err.mad(), abv_err.std(), abv_err_below

stats_list = []
for model in cor_models:
    stats_list.append(calc_model_stats(model[0]))

data.to_csv("data_ext.csv")

stats = pa.DataFrame(stats_list, columns = ['Name' , 'ABV Error Abs Mean', 'ABV Error Min', 'ABV Error Max', 'ABV Error MAD', 'ABV Error STD', 'ABV Error % Below 0.5'])
stats.to_csv("stats_abv.csv")
print(stats)

def add_plot_part(model_name, ax, col_name, functor, color):
    return data.plot.scatter(x=col_name, y=functor(model_name), label=model_name, c=color, ax=ax)

def add_plot(col_name, functor):
    first_model = cor_models[0]
    ax = add_plot_part(first_model[0], None, col_name, functor, first_model[2])
    if len(cor_models) > 1:
        for model in cor_models[1:]:
            add_plot_part(model[0], ax, col_name, functor, model[2])
    x = data[col_name]
    plt.plot(x, x, c='#000000', linewidth=1)
    plt.xlabel('Reference ' + col_name)
    plt.ylabel('Refractometer ' + col_name + ' at WCF=' + str(wcf))

add_plot("ABV", col_name_abv)
plt.savefig("stats_abv.png")
plt.show()