# (c) 2019 Thomas Mansencal
# https://stackoverflow.com/questions/58722583/how-do-i-convert-srm-to-lab-using-e-308-as-an-algorithm

EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 80.0]

def ebc_to_rgb(ebc):
    srm = ebc / 1.97
    r = round(min(255, max(0, 255 * 0.975**srm)))
    g = round(min(255, max(0, 245 * 0.88**srm)))
    b = round(min(255, max(0, 220 * 0.7**srm)))
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))