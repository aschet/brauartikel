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

def calc_ebc_weyermann(ebc, boil_time, oe):
    return ebc * oe / 10.0 + calc_cc_weyermann(oe)

def calc_ebc_krueger(ebc, boil_time, oe):
    return ebc * oe / 10.0 + (boil_time / 60.0 * 1.5) + 2.0

def calc_ebc_hanghofer(ebc, boil_time, oe):
    return ebc * oe / 9.0 + 2.0

class Addition:
    def __init__(self, weight, ebc):
        self.weight = weight
        self.ebc = ebc

    def calc_mcu_l(self, volume):
        return kg_to_lb(self.weight) * ebc_to_l(self.ebc) / l_to_gal_us(volume)

    def calc_mcu_srm(self, volume):
        return kg_to_lb(self.weight) * ebc_to_srm(self.ebc) / l_to_gal_us(volume)

    def calc_mcu_ebc(self, weight):
        return self.weight * self.ebc / weight

class BrewData:
    def __init__(self, ebc_ref, oe, boil_time, volume, malt_additions):
        self.ebc_ref = ebc_ref
        self.oe = oe
        self.boil_time = boil_time
        self.volume = volume
        self.malt_additions = malt_additions

    def calc_ebc_l(self, functor):
        mcu = sum(i.calc_mcu_l(self.volume) for i in self.malt_additions)
        return srm_to_ebc(functor(mcu))

    def calc_ebc_srm(self):
        return sum(i.calc_mcu_srm(self.volume) for i in self.malt_additions)

    def calc_ebc(self, functor):
        weight = sum(i.weight for i in self.malt_additions)
        ebc = sum(i.calc_mcu_ebc(weight) for i in self.malt_additions)
        return functor(ebc, self.boil_time, self.oe)

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
fig_srm.savefig('graph_srm.pdf', format='pdf')

pilsner = BrewData(10.0, 11.7, 75.0, 275.0, [Addition(42.8, 3.75), Addition(2.3, 4.5), Addition(0.9, 6), Addition(0.5, 195)])
amber = BrewData(40.0, 11.7, 75.0, 275.0, [Addition(41.4, 3.75), Addition(2.3, 4.5), Addition(0.9, 6), Addition(1.9, 400)])
dark = BrewData(82.0, 12.3, 75.0, 275.0, [Addition(38.9, 3.75), Addition(2.4, 4.5), Addition(2.4, 195), Addition(0.9, 6), Addition(1.9, 1400)])

def calc_stats(brew_data):
    stats = []
    stats.append(('Burch', brew_data.calc_ebc_srm()))
    stats.append(('Daniels-Druey', brew_data.calc_ebc_l(calc_srm_daniels_druey)))
    stats.append(('Daniels-Linear', brew_data.calc_ebc_l(calc_srm_daniels_lin)))
    stats.append(('Hanghofer', brew_data.calc_ebc(calc_ebc_hanghofer)))
    stats.append(('Krüger', brew_data.calc_ebc(calc_ebc_krueger)))
    stats.append(('Morey', brew_data.calc_ebc_l(calc_srm_morey)))    
    stats.append(('Mosher-Linear', brew_data.calc_ebc_l(calc_srm_mosher_lin)))
    stats.append(('Noonan-Druey', brew_data.calc_ebc_l(calc_srm_noonan_druey)))
    stats.append(('Weyermann', brew_data.calc_ebc(calc_ebc_weyermann)))
    return stats

def to_deviation(ebc_ref, stats):
    for i, val in enumerate(stats):
        stats[i] = (val[0], val[1] - ebc_ref)

def print_stats(stats):
    for i in stats:
        print(i[0] + ': ' + '{:.0f}'.format(i[1]))

pilsner_stats = calc_stats(pilsner)
print("Pilsner")
print_stats(pilsner_stats)
print("Dunkles")
dark_stats = calc_stats(dark)
print_stats(dark_stats)

to_deviation(pilsner.ebc_ref, pilsner_stats)
to_deviation(dark.ebc_ref, dark_stats)

labels = list(zip(*pilsner_stats))[0]
dev_pilsner = list(zip(*pilsner_stats))[1]
dev_dark = list(zip(*dark_stats))[1]

fig_ebc_dev = plt.figure(constrained_layout=True, figsize=(8, 4))
axes = fig_ebc_dev.subplots(1, 1)
x = np.arange(len(labels))
width = 0.35
rects1 = axes.bar(x - width/2, dev_pilsner, width, label='Pilsner')
rects2 = axes.bar(x + width/2, dev_dark, width, label='Dunkles')
axes.set_ylabel('Abweichung [EBC]')
axes.set_xticks(x, labels)
axes.tick_params(axis="x", rotation=50)
axes.legend()
fig_ebc_dev.savefig('graph_dev.pdf', format='pdf')


#plt.show()