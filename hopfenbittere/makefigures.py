#!/usr/bin/env python3
# Copyright 2022 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction

def p_to_sg(p):
    return p / (258.6 - (p / 258.2 * 227.1)) + 1.0

def c_to_k(c):
    return c + 273.15

def calc_alpha_acid_concentration(hop_weight, alpha_acid_rating, cast_wort_volume):
    hop_weight_mg = hop_weight * 1000.0
    alpha_acid_rating_decimal = alpha_acid_rating / 100.0
    return alpha_acid_rating_decimal * hop_weight_mg / cast_wort_volume

class TimeLUT:
    def __init__(self, boil_times, utilizations):
        self.lut = list(zip(boil_times, utilizations))
    
    def lookup(self, boil_time):
        for entry in self.lut:
            if boil_time - entry[0] <= 0:
                return entry[1]
        return self.lut[-1][1]

class GravityLUT:
    def __init__(self, gravities, time_luts):
        self.lut = list(zip(gravities, time_luts))

    def lookup(self, gravity):
        for entry in self.lut:
            if gravity - entry[0] <= 0:
                return entry[1]
        return self.lut[-1][1]

rager_time = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51]
lut_rager = TimeLUT(rager_time, [5.0, 6.0, 8.0, 10.1, 12.1, 15.3, 18.8, 22.8, 26.9, 28.1, 30.0])

garetz_time = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]
lut_garetz = TimeLUT(garetz_time, [0, 2, 5, 8, 11, 14, 16, 18, 19, 20, 21, 22, 23])

mosher_time = [5, 15, 30, 45, 60, 90]
lut_mosher_1030 = TimeLUT(mosher_time, [6.0, 15.0, 22.0, 26.0, 29.0, 35.0])
lut_mosher_1040 = TimeLUT(mosher_time, [6.0, 15.0, 21.0, 26.0, 28.0, 34.0])
lut_mosher_1050 = TimeLUT(mosher_time, [5.0, 14.0, 21.0, 25.0, 28.0, 33.0])
lut_mosher_1060 = TimeLUT(mosher_time, [5.0, 14.0, 21.0, 24.0, 27.0, 32.0])
lut_mosher_1070 = TimeLUT(mosher_time, [4.0, 13.0, 19.0, 23.0, 26.0, 31.0])
lut_mosher_1080 = TimeLUT(mosher_time, [4.0, 13.0, 18.0, 22.0, 25.0, 29.0])
lut_mosher_1090 = TimeLUT(mosher_time, [3.0, 11.0, 16.0, 21.0, 23.0, 27.0])
mosher_gravity = [1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090]
lut_mosher = GravityLUT(mosher_gravity, [lut_mosher_1030, lut_mosher_1040, lut_mosher_1050, lut_mosher_1060, lut_mosher_1070, lut_mosher_1080, lut_mosher_1090])

daniels_time = [9, 19, 29, 44, 49, 74, 75]
lut_daniels = TimeLUT(daniels_time, [6, 15, 19, 24, 27, 30, 34])

noonan_time = [4, 5, 15, 30, 60, 90]
lut_noonan_1032 = TimeLUT(noonan_time, [5, 6, 12, 18, 31, 33])
lut_noonan_1051 = TimeLUT(noonan_time, [5, 6, 12, 17, 28, 30])
lut_noonan_1066 = TimeLUT(noonan_time, [4, 5, 10, 16, 26, 28])
lut_noonan_1076 = TimeLUT(noonan_time, [4, 4, 9, 16, 25, 27])
lut_noonan_1086 = TimeLUT(noonan_time, [3, 4, 8, 15, 23, 25])
noonan_gravity = [1.032, 1.051, 1.066, 1.076, 1.086]
lut_noonan = GravityLUT(noonan_gravity, [lut_noonan_1032, lut_noonan_1051, lut_noonan_1066, lut_noonan_1076, lut_noonan_1086])

def calc_fga_rager(sg):
    gravity_adjustment = 0.0
    if (sg > 1.050):
        gravity_adjustment = (sg - 0.05) / 2.0
    return 1.0 / (1.0 + gravity_adjustment)

def calc_utilization_rager(boil_time, sg):
    utilization = lut_rager.lookup(boil_time)   
    return utilization * calc_fga_rager(sg)

calc_utilization_rager_vectorized = np.vectorize(calc_utilization_rager)

def calc_utilization_rager_function(boil_time, sg):
    utilization = 18.11 + (13.86 * np.tanh((boil_time - 31.32) / 18.27))
    return utilization * calc_fga_rager(sg) 

def calc_utilization_garetz(boil_time, sg):
    utilization = lut_garetz.lookup(boil_time)  
    return utilization  

calc_utilization_garetz_vectorized = np.vectorize(calc_utilization_garetz)

def calc_utilization_garetz_function(boil_time, sg):
    utilization = np.maximum(0.0, 7.2994 + (15.0746 * np.tanh((boil_time - 21.86) / 24.71))) 
    return utilization    

def calc_utilization_mosher(boil_time, sg):
    return lut_mosher.lookup(sg).lookup(boil_time)

calc_utilization_mosher_vectorized = np.vectorize(calc_utilization_mosher)

def calc_utilization_tinseth(boil_time, sg_mean):
    gravity_adjustment = 1.65 * np.power(0.000125, sg_mean - 1.0)
    time_adjustment = (1.0 - np.exp(-0.04 * boil_time)) / 4.15 * 100.0
    return gravity_adjustment * time_adjustment

def calc_utilization_daniels(boil_time, sg):
    utilization = lut_daniels.lookup(boil_time)   
    return utilization * calc_fga_rager(sg)

calc_utilization_daniels_vectorized = np.vectorize(calc_utilization_daniels)

def calc_utilization_noonan(boil_time, sg):
    return lut_noonan.lookup(sg).lookup(boil_time)

calc_utilization_noonan_vectorized = np.vectorize(calc_utilization_noonan)

def calc_ibu(alpha_acid_concentration, utilization):
    utilization_decimal = utilization / 100.0
    return alpha_acid_concentration * utilization_decimal

def calc_kfwp_smith(temp):
    return 2.39 * 10.0**11 * np.exp(-9773.0 / c_to_k(temp))

print(calc_kfwp_smith(90))

hop_weight = 30
alpha_acid_rating = 5.0
pre_boil_volume = 25
pre_boil_extract = 13.0
pre_boil_sg = p_to_sg(pre_boil_extract)
boil_time = 120.0
evaporation_rate = 2.5
cast_wort_volume = pre_boil_volume - (evaporation_rate * boil_time / 60.0)
oe = pre_boil_extract * pre_boil_volume / cast_wort_volume
og = p_to_sg(oe)
sg_mean = (og + pre_boil_sg) / 2.0
pre_boil_sg = sg_mean

time_scale = np.linspace(0, boil_time, dtype=int)
utilizations_tinset = calc_utilization_tinseth(time_scale, sg_mean)
utilizations_rager = calc_utilization_rager_vectorized(time_scale, pre_boil_sg)
utilizations_rager_function = calc_utilization_rager_function(time_scale, pre_boil_sg)
utilizations_garetz = calc_utilization_garetz_vectorized(time_scale, pre_boil_sg)
utilizations_garetz_function = calc_utilization_garetz_function(time_scale, pre_boil_sg)
utilizations_mosher = calc_utilization_mosher_vectorized(time_scale, pre_boil_sg)
utilizations_daniels = calc_utilization_daniels_vectorized(time_scale, pre_boil_sg)
utilizations_noonan = calc_utilization_noonan_vectorized(time_scale, pre_boil_sg)

plt.plot(time_scale, utilizations_tinset)
plt.plot(time_scale, utilizations_rager)
plt.plot(time_scale, utilizations_rager_function)
plt.plot(time_scale, utilizations_garetz)
plt.plot(time_scale, utilizations_garetz_function)
plt.plot(time_scale, utilizations_mosher)
plt.plot(time_scale, utilizations_daniels)
plt.plot(time_scale, utilizations_noonan)
plt.show()