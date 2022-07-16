EBC_SCALE = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

def ebc_to_rgb(EBC):
    # observer=CIE 1964 10 Degree Standard Observer, illuminant=D65, path=7.5cm
    R=round(max(0, min(255, (-4.8043E-06*EBC**4)+(4.6721E-04*EBC**3)+(3.6423E-02*EBC**2)+(-6.4444E+00*EBC)+(2.6192E+02))))
    G=round(max(0, min(255, (2.1860E-05*EBC**4)+(-4.9979E-03*EBC**3)+(4.3213E-01*EBC**2)+(-1.6826E+01*EBC)+(2.4569E+02))))
    B=round(max(0, min(255, (1.1796E-07*EBC**6)+(-3.2688E-05*EBC**5)+(3.5723E-03*EBC**4)+(-1.9423E-01*EBC**3)+(5.4069E+00*EBC**2)+(-6.9262E+01*EBC)+(2.6361E+02))))
    return '#%02x%02x%02x' % (int(R), int(G), int(B))

for ebc in EBC_SCALE:
    print('{} -> {}'.format(ebc, ebc_to_rgb(ebc)))