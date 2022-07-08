#!/usr/bin/env python3
# Copyright 2022 Thomas Ascher
# SPDX-License-Identifier: GPL-3.0+

from cmath import nan
import numpy as np
import matplotlib.pyplot as plt

def ebc_to_l(ebc):
    return ebc / 2.65

def ebc_to_srm(ebc):
    return ebc / 1.97

def srm_to_ebc(srm):
    return srm * 1.97

def kg_to_lb(kg):
    return kg / 0.45359237

def l_to_gal_us(l):
    return l / 3.785

def calc_srm_morey(mcu):
    return 1.49 * np.power(mcu, 0.69)

def calc_srm_daniels_lin(mcu):
    return 0.2 * mcu + 8.4

def calc_srm_daniels_druey(mcu):
    return 1.73 * np.power(mcu, 0.64) - 0.267

def calc_srm_mosher_lin(mcu):
    return 0.3 * mcu + 4.7

def calc_srm_noonan_druey(mcu):
    return 15.03 * np.power(mcu, 0.27) - 15.53

def calc_cc_weyermann(oe):
    if oe <= 7.0:
        return 0.0
    elif oe <= 10.0:
        return 3.0
    elif oe <= 15.0:
        return 5.0
    elif oe <= 20.0:
        return 7.0
    else:
        return 10

def calc_ebc_weyermann(ebc, oe):
    return ebc * oe / 10.0 + calc_cc_weyermann(oe)

class Addition:
    def __init__(self, weight, ebc):
        self.weight = weight
        self.ebc = ebc

    def calc_mcu_imp(self, volume):
        return kg_to_lb(self.weight) * ebc_to_l(self.ebc) / l_to_gal_us(volume)

    def calc_mcu_met(self, weight):
        return self.weight * self.ebc / weight

class BrewData:
    def __init__(self, oe, volume, malt_additions):
        self.oe = oe
        self.volume = volume
        self.malt_additions = malt_additions

    def calc_ebc_imp(self, functor):
        srm_lin = sum(i.calc_mcu_imp(self.volume) for i in self.malt_additions)
        return srm_to_ebc(functor(srm_lin))

    def calc_ebc_met(self, functor):
        weight = sum(i.weight for i in self.malt_additions)
        ebc = sum(i.calc_mcu_met(weight) for i in self.malt_additions)
        return functor(ebc, self.oe)

mcu_scale = np.linspace(0, 100, dtype=int)

srm_morey = calc_srm_morey(mcu_scale)

fig_srm = plt.figure(constrained_layout=True, figsize=(12, 4))
axes = fig_srm.subplots(1, 3, sharex=True, sharey=True)

def plot_srm(ax, name, srm_funcs):
    ax.set_title(name)
    ax.set_ylim([0,40])
    ax.set_xlabel('MCU')
    ax.set_ylabel('SRM')
    ax.plot(mcu_scale, srm_morey, label='Morey')

    for srm_func in srm_funcs:
        srm_scale = srm_func[1](mcu_scale)
        ax.plot(mcu_scale, srm_scale, label=name + '-' + srm_func[0])

    ax.legend(loc='lower right')  

plot_srm(axes[0], 'Daniels', [('Linear', calc_srm_daniels_lin), ('Druey', calc_srm_daniels_druey)])
plot_srm(axes[1], 'Mosher', [('Linear', calc_srm_mosher_lin)])
plot_srm(axes[2], 'Noonan', [('Druey', calc_srm_noonan_druey)])

#fig_srm.savefig('graph_srm.pdf', format='pdf')
#plt.show()



pilsner = BrewData(11.7, 275.0, [Addition(42.8, 3.75), Addition(2.3, 4.5), Addition(0.9, 6), Addition(0.5, 195)])
print(pilsner.calc_ebc_met(calc_ebc_weyermann))