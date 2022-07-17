EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

def ebc_to_rgb(ebc):
    srm = ebc / 1.94
    # SRM to sRGB model for 7.5 cm glas diameter
    r=round(max(0.0, min(255.0, 255.0*(1.017e+00+srm*(-4.587e-02+srm*(2.848e-04+srm*(2.162e-05+srm*(-3.604e-07))))))))
    g=round(max(0.0, min(255.0, 255.0*(9.723e-01+srm*(-1.292e-01+srm*(6.489e-03+srm*(-1.477e-04+srm*(1.277e-06))))))))
    b=round(max(0.0, min(255.0, 255.0*(1.022e+00+srm*(-5.359e-01+srm*(8.222e-02+srm*(-5.791e-03+srm*(2.088e-04+srm*(-3.745e-06+srm*(2.650e-08))))))))))
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))