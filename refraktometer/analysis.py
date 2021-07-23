#!/usr/bin/env python3
from numpy.core.fromnumeric import mean
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt

# Refractometer Correlation Model Evaluation
# greetings, Thomas Ascher

# aat = apparent attenuation in %
# ae = apparent extract in °P
# oe = original extract in °P
# rii = refractive index initial
# rif = refractive index final
# sg = specific gravity
# wcf = wort correction factor

wcf = 1.0
filter_outliers = True

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

def col_name_ae_dev(name):
    return col_name('AE Dev', name)

def col_name_abv(name):
    return col_name('ABV', name)

def col_name_abv_dev(name):
    return col_name('ABV Dev', name)

data = pa.read_csv('data.csv', delimiter=',')

col_name_riic_dev = 'RIIC Dev'
data[col_name_riic_dev] = data.apply(lambda row: correct_ri(row.RII, wcf) - row.OE, axis=1)
if filter_outliers == True:
    riic_dev_threshold = data[col_name_riic_dev].std()
    print('Filtering outliers over ' + col_name_riic_dev + ': ' + str(riic_dev_threshold))
    print()
    data = data[(abs(data[col_name_riic_dev]) <= riic_dev_threshold)]

data['AAT'] = (data['OE'] - data['AE']) * 100.0 / data['OE']
data['ABV'] = data.apply(lambda row: calc_abv(row.OE, row.AE), axis=1)
wcf_col_name = 'WCF'
data[wcf_col_name] = data['RII'] / data['OE']

def add_cor_model_data(name, functor):
    data[col_name_ae(name)] = data.apply(lambda row: functor(row.RII, row.RIF, wcf), axis=1)
    data[col_name_ae_dev(name)] = data.apply(lambda row: row[col_name_ae(name)] - row.AE, axis=1)
    data[col_name_abv(name)] = data.apply(lambda row: calc_abv(correct_ri(row.RII, wcf), row[col_name_ae(name)]), axis=1)
    data[col_name_abv_dev(name)] = data.apply(lambda row: row[col_name_abv(name)] - row.ABV, axis=1)

for model in cor_models:
    add_cor_model_data(model[0], model[1])

wcf_list = data[wcf_col_name]
wcf_stats = pa.DataFrame([(wcf_list.min(), wcf_list.max(), wcf_list.mean(), wcf_list.std())], columns = ['WCF Min', 'WCF Max', 'WCF Mean', 'WCF STD'])
print(wcf_stats)
print()

def calc_model_stats(name):
    abv_dev = data[col_name_abv_dev(name)]
    abv_dev_abs = abv_dev.abs()
    abv_dev_below_25 = abv_dev_abs.le(0.25).sum() / len(abv_dev_abs) * 100.0    
    abv_dev_below_50 = abv_dev_abs.le(0.5).sum() / len(abv_dev_abs) * 100.0
    return name, abv_dev_abs.min(), abv_dev_abs.max(), abv_dev_abs.mean(), abv_dev.std(), abv_dev_below_25, abv_dev_below_50

stats_list = []
for model in cor_models:
    stats_list.append(calc_model_stats(model[0]))

data.to_csv("data_ext.csv")

stats_columns = ['Name' , 'ABV Min Dev', 'ABV Max Dev', 'ABV Mean Dev', 'ABV Standard Dev', 'ABV Dev % Below 0.25', 'ABV Dev % Below 0.5']
stats_colors = ['#a9f693', '#00c295', '#ff0043', '#ff795b']
stats = pa.DataFrame(stats_list, columns=stats_columns)
stats.to_csv("stats_abv.csv")
print(stats)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle('Refractometer Correlation Model Evaluation')
ax_stats = axes[0]
ax_data = axes[1]

stats.plot(x='Name', y=stats_columns[1:-2], kind='bar', color=stats_colors, ax=ax_stats)
ax_stats.title.set_text('ABV Deviation Statistics')
ax_stats.set_xlabel('')
ax_stats.set_ylabel('Model ABV Deviation at WCF=' + str(wcf))

def add_data_plot_part(model_name, ax, col_name, functor, color):
    return data.plot.scatter(x=col_name, y=functor(model_name), label=model_name, c=color, ax=ax)

def add_data_plot(col_name, functor, ax):
    x = data[col_name]
    plt.plot(x, x, c='#000000', linewidth=1, axes=ax)
    for model in cor_models:
            add_data_plot_part(model[0], axes[1], col_name, functor, model[2])
    ax.title.set_text(col_name + ' Deviation')
    ax.set_xlabel('Reference ' + col_name)
    ax.set_ylabel('Model ' + col_name + ' at WCF=' + str(wcf))                

add_data_plot('ABV', col_name_abv, ax_data)

plt.savefig('stats_abv.png')
plt.show()