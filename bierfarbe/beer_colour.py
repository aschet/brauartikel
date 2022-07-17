EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

def ebc_to_rgb(ebc):
    # observer=CIE 1964 10 Degree Standard Observer, illuminant=D65, path=7.5cm
    R=round(max(0, min(255, 2.604e+02+ebc*(-6.233e+00+ebc*(2.938e-02+ebc*(5.671e-04+ebc*(-5.317e-06)))))))
    G=round(max(0, min(255, 2.465e+02+ebc*(-1.673e+01+ebc*(4.267e-01+ebc*(-4.914e-03+ebc*(2.145e-05)))))))
    B=round(max(0, min(255, 2.625e+02+ebc*(-7.102e+01+ebc*(5.612e+00+ebc*(-2.031e-01+ebc*(3.756e-03+ebc*(-3.452e-05+ebc*(1.250e-07)))))))))
    return '#%02x%02x%02x' % (int(R), int(G), int(B))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))