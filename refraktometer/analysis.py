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

# https://www.brewersjournal.info/science-basic-beer-alcohol-extract-determinations/
def calc_abw(oe, ae):
    return (0.8052 * (oe - ae)) / (2.0665 - (1.0665 * oe / 100.0))

# https://www.brewersjournal.info/science-basic-beer-alcohol-extract-determinations/
def calc_abv(abw, fg):
    return abw * fg / 0.7907

def calc_abv_simple(oe, ae):
    return calc_abv(calc_abw(oe, ae), plato_to_sg(ae))

def calc_model_abv(rii, rif, functor):
    oe, ae = functor(rii, rif)
    return calc_abv_simple(oe, ae)

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
    oe, ae = cor_bonham(rii, rif)
    fg = plato_to_sg(ae)
    return calc_abv(abw, fg)

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

class ABVModel:
    def __init__(self, name, functor, color):
        self.name = name
        self.functor = functor
        self.color = color

    def calc_abv(self, rii, rif):
        return self.functor(rii, rif)

abv_models = [
    ABVModel('Terrill Linear', calc_abv_terrill_linear, color_palette[5]),
    ABVModel('Terrill Cubic', calc_abv_terrill_cubic, color_palette[6]),
    ABVModel('Novotny Linear', calc_abv_novotny_linear, color_palette[3]),
    ABVModel('Novotny Quadratic', calc_abv_novotny_quadratic, color_palette[4]),    
    ABVModel('Bonham', calc_abv_bonham, color_palette[0]),
    ABVModel('Gardner', calc_abv_gardner, color_palette[1]),
    ABVModel('Gossett', calc_abv_gosett, color_palette[2]),    
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
data_dev = pa.DataFrame()

if filter_outliers == True:
    rii_dev_threshold = (data[col_name_oe] - data[col_name_rii]).std()
    print('Filtering ' + col_name_rii + ' outliers over ' + str(rii_dev_threshold))
    print()
    data = data[(abs(data[col_name_oe] - data[col_name_rii]) <= rii_dev_threshold)]

data[col_name_wcf] = data[col_name_rii] / data[col_name_oe]
data[col_name_abv] = calc_abv_simple(data[col_name_oe], data[col_name_ae])
for abv_model in abv_models:
    data[model_col_name_abv(abv_model.name)] = abv_model.calc_abv(data[col_name_rii], data[col_name_rif])

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
    data_dev[name] = abv_dev
    abv_dev_abs = abv_dev.abs()
    abv_dev_below_25 = abv_dev_abs.le(0.25).sum() / float(len(abv_dev_abs)) * 100.0    
    abv_dev_below_50 = abv_dev_abs.le(0.5).sum() / float(len(abv_dev_abs)) * 100.0
    rsquare = calc_rsquare(abv_observed, abv_reference)
    return name, abv_dev_abs.min(), abv_dev_abs.max(), abv_dev.mean(), abv_dev.std(), rsquare, abv_dev_below_25, abv_dev_below_50

stats_list = []
for abv_model in abv_models:
    stats_list.append(calc_abv_model_stats(abv_model.name))

data.to_csv("data_eval.csv", index=False)

stats_columns = ['Name' , 'Min', 'Max', 'Mean', 'STD', 'R-Squared', '% Below 0.25', '% Below 0.5']
stats = pa.DataFrame(stats_list, columns=stats_columns)
stats.to_csv("stats_abv_dev.csv", index=False)
print("ABV Deviation Statistics:")
print(stats)

abv_model_names = list(map(lambda model: model.name, abv_models))

fig = plt.figure(constrained_layout=True, figsize=(14, 8))
fig.suptitle('Refractometer Correlation Model Evaluation')
subfigs = fig.subfigures(1, 2)

subfigs[0].suptitle('ABV Deviation Quantils')
ax_quantils = subfigs[0].subplots(1, 1)
ax_quantils.axhline(0.0, linestyle='--', c='#000000', linewidth=1)
dev_caption = 'ABV Deviation at WCF=%.2f'%wcf
ax_quantils.set_ylabel(dev_caption)
data_dev.boxplot(abv_model_names, ax=ax_quantils, rot=45, grid=False, showmeans=True)

subfigs[1].suptitle('ABV Deviation Density')
name_indexed_stats = stats.set_index('Name')
cols = 2
rows = len(abv_models) // cols + len(abv_models) % cols
ax_densities = subfigs[1].subplots(rows, cols, sharex=True, sharey=True)
for i, abv_model in enumerate(abv_models):
    row = i // cols
    col = i % cols
    ax_desnity = ax_densities[row][col]
    rsquare = name_indexed_stats.loc[abv_model.name]['R-Squared']
    ax_desnity.set_title(abv_model.name + ' (R²=' + '%.3f'%rsquare + ')')    
    ax_desnity.set_xlabel(dev_caption)
    data_dev[abv_model_names[i]].plot.hist(density=True, xlim=[-1,1], ax=ax_desnity)
    data_dev[abv_model_names[i]].plot.density(ax=ax_desnity)  

plt.savefig('stats_abv_dev.png')
plt.show()