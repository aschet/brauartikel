import numpy as np
import colour.plotting

def ebc_to_rgb(ebc):
    srm = ebc / 1.94
    # SRM to sRGB model for 7.5 cm glas diameter
    r=max(0.0, min(1.0, 1.017e+00+srm*(-4.587e-02+srm*(2.848e-04+srm*(2.162e-05+srm*(-3.604e-07))))))
    g=max(0.0, min(1.0, 9.723e-01+srm*(-1.292e-01+srm*(6.489e-03+srm*(-1.477e-04+srm*(1.277e-06))))))
    b=max(0.0, min(1.0, 1.022e+00+srm*(-5.359e-01+srm*(8.222e-02+srm*(-5.791e-03+srm*(2.088e-04+srm*(-3.745e-06+srm*(2.650e-08))))))))
    return [r, g, b]

ebc = np.arange(start=0, stop=80+1, dtype='int')

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=ebc_to_rgb(i)) for i in ebc], **{'standalone': False})
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
