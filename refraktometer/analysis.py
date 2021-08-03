#!/usr/bin/env python3
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

wcf = 1.04
filter_outliers = True

def correct_ri(ri):
    return ri / wcf

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def plato_to_sg(se):
    return 1.0 + (se / (258.6 - ((se / 258.2) * 227.1)))

# Revisiting ABV Calculations, Zymurgy July/August 2019 p. 48
def calc_abv(oe, ae):
    og = plato_to_sg(oe)
    fg = plato_to_sg(ae)
    return fg * (5118.0 * (og**2 - fg**2) + 16755.0 * (fg - og)) / (8.739 * og**4 \
        - 57.22 * og**3 + 89.09 * og**2 + 14.95 + og - 105.99)

def calc_model_abv(rii, rif, functor):
    oe, fe = functor(rii, rif)
    return calc_abv(oe, fe)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_bonham(rii, rif):
    oe = correct_ri(rii)
    return oe, sg_to_plato(1.001843 - 0.002318474 * oe - 0.000007775 * oe**2 - \
        0.000000034 * oe**3 + 0.00574 * rif + \
        0.00003344 * rif**2 + 0.000000086 * rif**3)

def calc_abv_bonham(rii, rif):
    return calc_model_abv(rii, rif, cor_bonham)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_gardner(rii, rif):
    oe = correct_ri(rii)
    return oe, 1.53 * rif - 0.59 * oe

def calc_abv_gardner(rii, rif):
    return calc_model_abv(rii, rif, cor_gardner)

# http://www.ithacoin.com/brewing/Derivation.htm
def calc_abv_gosett(rii, rif):
    k = 0.445
    c = 100.0 * (rii - rif) / (100.0 - 48.4 * k - 0.582 * rif)
    abw = 48.4 * c / (100 - 0.582 * c)
    oe, fe = cor_bonham(rii, rif)
    fg = plato_to_sg(fe)
    return abw * fg / 0.794

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(rii, rif):
    oe = correct_ri(rii)
    rifc = correct_ri(rif)
    return oe, sg_to_plato(-0.002349 * oe + 0.006276 * rifc + 1.0)

def calc_abv_novotny_linear(rii, rif):
    return calc_model_abv(rii, rif, cor_novotny_linear)

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html     
def cor_novotny_quadratic(rii, rif):
    oe = correct_ri(rii)
    rifc = correct_ri(rif)
    return oe, sg_to_plato(1.335 * 10.0**-5 * oe**2 - \
        3.239 * 10.0**-5 * oe * rifc + \
        2.916 * 10.0**-5 * rifc**2 - \
        2.421 * 10.0**-3 * oe + \
        6.219 * 10.0**-3 * rifc + 1.0)

def calc_abv_novotny_quadratic(rii, rif):
    return calc_model_abv(rii, rif, cor_novotny_quadratic)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(rii, rif):
    oe = correct_ri(rii)
    rifc = correct_ri(rif)           
    return oe, sg_to_plato(1.0 - 0.000856829 * oe + 0.00349412 * rifc)

def calc_abv_terrill_linear(rii, rif):
    return calc_model_abv(rii, rif, cor_terrill_linear)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_cubic(rii, rif):
    oe = correct_ri(rii)
    rifc = correct_ri(rif)          
    return oe, sg_to_plato(1.0 - 0.0044993 * oe + 0.000275806 * oe**2 - \
        0.00000727999 * oe**3 + 0.0117741 * rifc - \
        0.00127169 * rifc**2 + 0.0000632929 * rifc**3)

def calc_abv_terrill_cubic(rii, rif):
    return calc_model_abv(rii, rif, cor_terrill_cubic)

color_palette = [
    '#a9f693',
    '#00c295',
    '#84c5ea',
    '#2e709f',
    '#ff0043',
    '#ff795b',
    '#fddb85'
]

abv_models = [
    ('Bonham', calc_abv_bonham, color_palette[0]),
    ('Gardner', calc_abv_gardner, color_palette[1]),
    ('Gossett', calc_abv_gosett, color_palette[2]),    
    ('Novotny Linear', calc_abv_novotny_linear, color_palette[3]),
    ('Novotny Quadratic', calc_abv_novotny_quadratic, color_palette[4]),
    ('Terrill Linear', calc_abv_terrill_linear, color_palette[5]),
    ('Terrill Cubic', calc_abv_terrill_cubic, color_palette[6]),
]

col_name_abv = 'ABV'
col_name_wcf = 'WCF'
col_name_oe = 'OE'
col_name_ae = 'AE'
col_name_rii = 'RII'
col_name_rif = 'RIF'

def model_col_name_abv(name):
    return col_name_abv + ' ' + name

data = pa.read_csv('data.csv', delimiter=',')

if filter_outliers == True:
    rii_dev_threshold = (data[col_name_oe] - data[col_name_rii]).std()
    print('Filtering ' + col_name_rii + ' outliers over ' + str(rii_dev_threshold))
    print()
    data = data[(abs(data[col_name_oe] - data[col_name_rii]) <= rii_dev_threshold)]

data[col_name_abv] = calc_abv(data[col_name_oe], data[col_name_ae])
data[col_name_wcf] = data[col_name_rii] / data[col_name_oe]
for abv_model in abv_models:
    data[model_col_name_abv(abv_model[0])] = abv_model[1](data[col_name_rii], data[col_name_rif])

wcf_list = data[col_name_wcf]
wcf_stats = pa.DataFrame([(wcf_list.min(), wcf_list.max(), wcf_list.mean(), wcf_list.std())], columns = ['Min', 'Max', 'Mean', 'STD'])
print("WCF Statistics:")
print(wcf_stats)
print()

def calc_rsquare(estimations, measureds):
    see =  ((np.array(measureds) - np.array(estimations))**2).sum()
    mmean = (np.array(measureds)).sum() / float(len(measureds))
    derr = ((mmean - measureds)**2).sum()    
    return 1 - (see / derr)

def calc_abv_model_stats(name):
    abv_observed = data[model_col_name_abv(name)]
    abv_reference = data[col_name_abv]
    abv_dev = abv_reference - abv_observed
    abv_dev_abs = abv_dev.abs()
    abv_dev_below_25 = abv_dev_abs.le(0.25).sum() / float(len(abv_dev_abs)) * 100.0    
    abv_dev_below_50 = abv_dev_abs.le(0.5).sum() / float(len(abv_dev_abs)) * 100.0
    rsquare = calc_rsquare(abv_observed, abv_reference)
    return name, abv_dev_abs.min(), abv_dev_abs.max(), abv_dev_abs.median(), abv_dev_abs.mean(), abv_dev.std(), rsquare, abv_dev_below_25, abv_dev_below_50

stats_list = []
for abv_model in abv_models:
    stats_list.append(calc_abv_model_stats(abv_model[0]))

data.to_csv("data_ext.csv")

stats_columns = ['Name' , 'Min', 'Max', 'Median', 'Mean', 'STD', 'R-Squared', '% Below 0.25', '% Below 0.5']
stats_colors = ['#a9f693', '#00c295', '#ff0043', '#ff795b']
stats = pa.DataFrame(stats_list, columns=stats_columns)
stats.to_csv("stats_abvdev.csv")
print("ABV Deviation Statistics:")
print(stats)

fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle('Refractometer Correlation Model Evaluation')
ax_stats = axes[0]
ax_data = axes[1]

wcf_caption_part = 'at WCF=' + '%.2f'%wcf

stats.plot(x='Name', y=stats_columns[1:-3], kind='bar', color=color_palette, ax=ax_stats)
ax_stats.title.set_text('ABV Deviation Statistics')
ax_stats.set_xlabel('')
ax_stats.set_ylabel('Model ABV Deviation ' + wcf_caption_part)

def add_data_plot_part(model_name, ax, col_name, functor, color):
    specific_col_name = functor(model_name)
    rsquare = calc_rsquare(data[specific_col_name], data[col_name])
    label_content = model_name + ' (R²=' + '%.3f'%rsquare + ')'
    return data.plot.scatter(x=col_name, y=specific_col_name, label=label_content, c=color, ax=ax)

regression_line = data[col_name_abv]
plt.plot(regression_line, regression_line, c='#000000', linewidth=1, axes=ax_data)
for abv_model in abv_models:
        add_data_plot_part(abv_model[0], axes[1], col_name_abv, model_col_name_abv, abv_model[2])
ax_data.title.set_text(col_name_abv + ' Deviation')
ax_data.set_xlabel('Reference ' + col_name_abv)
ax_data.set_ylabel('Model ' + col_name_abv + ' ' + wcf_caption_part)                

plt.savefig('stats_abv.png')
plt.show()