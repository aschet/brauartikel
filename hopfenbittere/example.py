#!/usr/bin/env python3

import math

def p_to_sg(p):
    return p / (258.6 - (p / 258.2 * 227.1)) + 1.0

def ba_tinseth(kd, dkd):
    bakt = (1.0 - math.exp(-0.04 * kd)) / 4.15 * 100.0
    fd = 1.65 * math.pow(0.000125, dkd - 1.0)   
    return bakt * fd

def calc_ibu(hm, ha, ba, awv):
    return hm * 1000.0 * ha / 100.0 * ba / 100.0 / awv

vwk = 10.5
stw = 12.0
awv = 20.0
hm1 = 10.0
ha1 = 16.4
kd1 = 60.0
hm2 = 7.0
ha2 = 6.6
kd2 = 5.0

dvwk = p_to_sg(vwk)
print('dvwk=%.3f' % dvwk)
dstw = p_to_sg(stw)
print('dstw=%.3f' % dstw)
dkd = (dvwk + dstw) / 2
print('dkd=%.3f' % dkd)
ba1 = ba_tinseth(kd1, dkd) * 1.1
print('ba1=%.1f' % ba1)
ibu1 = calc_ibu(hm1, ha1, ba1, awv)
print('ibu1=%.1f' % ibu1)
ba2 = ba_tinseth(kd2, dkd) * 1.1
print('ba2=%.1f' % ba2)
ibu2 = calc_ibu(hm2, ha2, ba2, awv)
print('ibu2=%.1f' % ibu2)
ibu = ibu1 + ibu2
print('ibu=%.1f' % ibu)