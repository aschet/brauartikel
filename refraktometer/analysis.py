#!/usr/bin/env python3
from numpy.core.fromnumeric import mean
import pandas as pa

# Refractometer Correlation Function Evaluation
# greetings, Thomas Ascher

# AAT = apparent attenuation in %
# AE = apparent extract in °P
# BX = brix
# OE = original extract in °P
# SE = specific extract in °P
# SG = specific gravity
# WCF = wort correction factor

wcf = 1.0

def correct_bx(bx, wcf):
    return bx / wcf

# https://www.brewersfriend.com/plato-to-sg-conversion-chart
def sg_to_plato(sg):
    return (-1.0 * 616.868) + (1111.14 * sg) - (630.272 * sg**2) + (135.997 * sg**3)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_bonham(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    return sg_to_plato(1.001843 - 0.002318474 * bxic - 0.000007775 * bxic**2 - \
        0.000000034 * bxic**3 + 0.00574 * bxf + \
        0.00003344 * bxf**2 + 0.000000086 * bxf**3)

# The Use of Handheld Refractometers by Homebrewer, Zymurgy January/February 2001 p. 44
def cor_gardner(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    return 1.53 * bxf - 0.59 * bxic

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html
def cor_novotny_linear(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    return sg_to_plato(-0.002349 * bxic + 0.006276 * bxfc + 1.0)

# http://www.diversity.beer/2017/01/pocitame-nova-korekce-refraktometru.html     
def cor_novotny_quadratic(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)
    return sg_to_plato(1.335 * 10.0**-5 * bxic**2 - \
        3.239 * 10.0**-5 * bxic * bxfc + \
        2.916 * 10.0**-5 * bxfc**2 - \
        2.421 * 10.0**-3 * bxic + \
        6.219 * 10.0**-3 * bxfc + 1.0)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_linear(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)           
    return sg_to_plato(1.0 - 0.000856829 * bxic + 0.00349412 * bxfc)

# http://seanterrill.com/2011/04/07/refractometer-fg-results/
def cor_terrill_cubic(bxi, bxf, wcf):
    bxic = correct_bx(bxi, wcf)
    bxfc = correct_bx(bxf, wcf)          
    return sg_to_plato(1.0 - 0.0044993 * bxic + 0.000275806 * bxic**2 - \
        0.00000727999 * bxic**3 + 0.0117741 * bxfc - \
        0.00127169 * bxfc**2 + 0.0000632929 * bxfc**3)

def calc_abv(oe, fe):
    return (261.1/(261.53-fe))*(81.92*(oe-fe)/(206.65-1.0665*oe))/0.7894

data = pa.read_csv("data.csv", delimiter=',')
data['AAT'] = (data['OE'] - data['AE']) * 100.0 / data['OE']
data['ABV'] = data.apply(lambda row: calc_abv(row.OE, row.AE), axis=1)
wcf_col_name = 'WCF'
data[wcf_col_name] = data['BXI'] / data['OE']

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
    data[col_name_ae(name)] = data.apply(lambda row: functor(row.BXI, row.BXF, wcf), axis=1)
    data[col_name_ae_err(name)] = data.apply(lambda row: row[col_name_ae(name)] - row.AE, axis=1)
    data[col_name_abv(name)] = data.apply(lambda row: calc_abv(correct_bx(row.BXI, wcf), row[col_name_ae(name)]), axis=1)
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