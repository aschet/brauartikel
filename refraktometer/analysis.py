#!/usr/bin/env python3
from numpy.core.fromnumeric import mean
import pandas as pa

# Refractometer Correlation Function Evaluation
# greetings, Thomas Ascher

# aat = apparent attenuation in %
# ae = apparent extract in °P
# oe = original extract in °P
# ri = refractive index
# sg = specific gravity
# WCF = wort correction factor

wcf = 1.0

def correct_ri(ri, wcf):
    return ri / wcf

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

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

def calc_abv(oe, ae):
    return (261.1/(261.53-ae))*(81.92*(oe-ae)/(206.65-1.0665*oe))/0.7894

data = pa.read_csv("data.csv", delimiter=',')
data['AAT'] = (data['OE'] - data['AE']) * 100.0 / data['OE']
data['ABV'] = data.apply(lambda row: calc_abv(row.OE, row.AE), axis=1)
wcf_col_name = 'WCF'
data[wcf_col_name] = data['RII'] / data['OE']

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

add_cor_model_data(name_bonham, cor_bonham)
add_cor_model_data(name_gardner, cor_gardner)
add_cor_model_data(name_novotny_linear, cor_novotny_linear)
add_cor_model_data(name_novotny_quadratic, cor_novotny_quadratic)
add_cor_model_data(name_terrill_linear, cor_terrill_linear)
add_cor_model_data(name_terrill_cubic, cor_terrill_cubic)

wcf_list = data[wcf_col_name]
wcf_stats = pa.DataFrame([(wcf_list.mean(), wcf_list.min(), wcf_list.max(), wcf_list.mad(), wcf_list.std())], columns = ['WCF Mean', 'WCF Min', 'WCF Max', 'WCF MAD', 'WCF STD'])
print(wcf_stats)
print()

def calc_model_stats(name):
    abv_err = data[col_name_abv_err(name)]
    abv_err_abs = abv_err.abs()
    abv_err_below = abv_err_abs.le(0.5).sum() / len(abv_err_abs) * 100.0
    return name, abv_err_abs.mean(), abv_err[abv_err_abs.idxmin()], abv_err[abv_err_abs.idxmax()], abv_err.mad(), abv_err.std(), abv_err_below

stats_list = [
calc_model_stats(name_bonham),
calc_model_stats(name_gardner),
calc_model_stats(name_novotny_linear),
calc_model_stats(name_novotny_quadratic),
calc_model_stats(name_terrill_linear),
calc_model_stats(name_terrill_cubic),
]

data.to_csv("data_ext.csv")

stats = pa.DataFrame(stats_list, columns = ['Name' , 'ABV Error Abs Mean', 'ABV Error Min', 'ABV Error Max', 'ABV Error MAD', 'ABV Error STD', 'ABV Error % Below 0.5'])
stats.to_csv("stats.csv")
print(stats)