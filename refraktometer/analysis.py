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

wcf = 1.0
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

class RefracModel:
    def __init__(self, name, cor_model, abv_model):
        self.name = name
        self.cor_model = cor_model
        self.abv_model = abv_model

    def calc_ae(self, rii, rif):
        _, ae = self.cor_model(rii, rif)
        return ae

    def calc_abv(self, rii, rif):
        return self.abv_model(rii, rif)

refrac_models = [
    RefracModel('Terrill Linear', cor_terrill_linear, calc_abv_terrill_linear),
    RefracModel('Terrill Cubic', cor_terrill_cubic, calc_abv_terrill_cubic),
    RefracModel('Novotny Linear', cor_novotny_linear, calc_abv_novotny_linear),
    RefracModel('Novotny Quadratic', cor_novotny_quadratic, calc_abv_novotny_quadratic),
    RefracModel('Bonham', cor_bonham, calc_abv_bonham),
    RefracModel('Gardner', cor_gardner, calc_abv_gardner),
    RefracModel('Gossett', cor_bonham, calc_abv_gosett)
]

model_names = list(map(lambda model: model.name, refrac_models))

col_name_abv = 'ABV'
col_name_wcf = 'WCF'
col_name_oe = 'OE'
col_name_ae = 'AE'
col_name_rii = 'RII'
col_name_rif = 'RIF'
row_name_square = 'rsquare'

def model_col_name(section, name):
    return section + ' ' + name

data = pa.read_csv('data.csv', delimiter=',')
data_abv_dev = pa.DataFrame()
data_ae_dev = pa.DataFrame()

if filter_outliers == True:
    rii_dev_threshold = (data[col_name_oe] - data[col_name_rii]).std()
    print('Filtering ' + col_name_rii + ' outliers over ' + str(rii_dev_threshold))
    print()
    data = data[(abs(data[col_name_oe] - data[col_name_rii]) <= rii_dev_threshold)]

data[col_name_wcf] = data[col_name_rii] / data[col_name_oe]
data[col_name_abv] = calc_abv_simple(data[col_name_oe], data[col_name_ae])

for model in refrac_models:
    model_col_name_ae = model_col_name(col_name_ae, model.name)
    data[model_col_name_ae] = model.calc_ae(data[col_name_rii], data[col_name_rif])
    data_ae_dev[model.name] = data[col_name_ae] - data[model_col_name_ae]
    model_col_name_abv = model_col_name(col_name_abv, model.name)
    data[model_col_name_abv] = model.calc_abv(data[col_name_rii], data[col_name_rif])   
    data_abv_dev[model.name] = data[col_name_abv] - data[model_col_name_abv]

data.to_csv("data_eval.csv", index=False)

wcf_stats = data[col_name_wcf].describe()
print("WCF Statistics:")
print(wcf_stats)
print()

def calc_rsquare(estimations, measureds):
    see =  ((np.array(measureds) - np.array(estimations))**2).sum()
    mmean = (np.array(measureds)).sum() / float(len(measureds))
    derr = ((mmean - measureds)**2).sum()    
    return 1 - (see / derr)

def create_stats(devs, col_name):
    stats = devs.describe()
    stats.loc[row_name_square] = list(map(lambda name: calc_rsquare(data[model_col_name(col_name, name)], data[col_name]), model_names))
    return stats

stats_ae_dev = create_stats(data_ae_dev, col_name_ae)
stats_ae_dev.to_csv("stats_ae_dev.csv", index=True)
stats_abv_dev = create_stats(data_abv_dev, col_name_abv)
stats_abv_dev.to_csv("stats_abv_dev.csv", index=True)

print("ABV Deviation Statistics:")
print(stats_abv_dev)

def plot_devs(col_name, data_dev, stats_dev):
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    fig.suptitle('Refractometer Correlation Model Evaluation')
    subfigs = fig.subfigures(1, 2)

    subfigs[0].suptitle(col_name + ' Deviation Quantils')
    ax_quantils = subfigs[0].subplots(1, 1)
    ax_quantils.axhline(0.0, linestyle='--', c='#000000', linewidth=1)
    dev_caption = col_name + ' Deviation at WCF=%.2f'%wcf
    ax_quantils.set_ylabel(dev_caption)
    data_dev.boxplot(model_names, ax=ax_quantils, rot=45, grid=False, showmeans=True)

    subfigs[1].suptitle(col_name + ' Deviation Histogram')
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

#fig_ae = plot_devs(col_name_ae, data_ae_dev, stats_ae_dev)
#fig_ae.savefig('stats_ae_dev.png')
fig_abv = plot_devs(col_name_abv, data_abv_dev, stats_abv_dev)
fig_abv.savefig('stats_abv_dev.png')
plt.show()