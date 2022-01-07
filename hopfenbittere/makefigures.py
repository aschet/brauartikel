#!/usr/bin/env python3
# Copyright 2022 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

import math

class HopUtilizationLUT:
    


def calc_alpha_acid_concentration(hop_weight, alpha_acid_rating, cast_wort_volume):
    hop_weight_mg = hop_weight * 1000.0
    alpha_acid_rating_decimal = alpha_acid_rating / 100.0
    return alpha_acid_rating_decimal * hop_weight_mg / cast_wort_volume

def calc_utilization_rager(boil_time, sg):
    utilization = 0.0
    if boil_time >= 0 and boil_time <= 5:
        utilization = 5.0
    elif boil_time >= 6 and boil_time <= 10:
        utilization = 6
    elif boil_time >= 11 and boil_time <= 15:
        utilization = 8
    elif boil_time >= 16 and boil_time <= 20:
        utilization = 10.1
    elif boil_time >= 21 and boil_time <= 25:
        utilization = 12.1
    elif boil_time >= 26 and boil_time <= 30:
        utilization = 15.3
    elif boil_time >= 31 and boil_time <= 35:
        utilization = 18.8
    elif boil_time >= 36 and boil_time <= 40:
        utilization = 22.8
    elif boil_time >= 41 and boil_time <= 45:
        utilization = 26.9
    elif boil_time >= 46 and boil_time <= 50: 
        utilization = 28.1
    elif boil_time >= 51:
        utilization = 30.0

    gravity_adjustment = 0.0
    if (sg > 1.050):
        gravity_adjustment = (sg - 0.05) / 2.0    
    return utilization / (1.0 + gravity_adjustment)

def calc_utilization_tinseth(boil_time, sg_mean):
    gravity_adjustment = 1.65 * math.pow(0.000125, sg_mean - 1.0)
    time_adjustment = (1.0 - math.exp(-0.04 * boil_time)) / 4.15
    return gravity_adjustment * time_adjustment * 100.0

def calc_ibu(alpha_acid_concentration, utilization):
    utilization_decimal = utilization / 100.0
    return alpha_acid_concentration * utilization_decimal

hop_weight = 30
alpha_acid_rating = 5.0
cast_wort_volume = 22.73045
boil_time = 60.0
sg_mean = 1.050

alpha_acid_concentration = calc_alpha_acid_concentration(hop_weight, alpha_acid_rating, cast_wort_volume)
utilization = calc_utilization_tinseth(boil_time, sg_mean)
print(utilization)
print(calc_ibu(alpha_acid_concentration, utilization))
