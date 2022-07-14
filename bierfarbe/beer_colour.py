# Philip Lee's approximation from a color swatch and curve fitting.
# http://www.brewtarget.org/

EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

def ebc_to_rgb(ebc):
    srm = ebc / 1.97
    r = round(min(253, max(0, 0.5 + (272.098 - 5.80255 * srm))))
    if srm > 35.0:
        g = 0.0
    else:
        g = round(min(255, max(0, 0.5 + (2.41975e2 - 1.3314e1 * srm + 1.881895e-1 * srm * srm))))
    b = round(min(255, max(0, 0.5 + (179.3 - 28.7 * srm))))
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))