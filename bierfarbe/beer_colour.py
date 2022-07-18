import numpy as np
import colour.plotting

def ebc_to_rgb(ebc):
    srm = ebc / 1.94
    # SRM to sRGB model for 7.5 cm glas diameter
    r=1.0022e+00+srm*(-3.1966e-02+srm*(-2.5115e-03+srm*(2.3949e-04+srm*(-8.1638e-06+srm*(1.2825e-07+srm*(-7.7209e-10))))))
    g=9.9473e-01+srm*(-1.5079e-01+srm*(1.0788e-02+srm*(-4.8003e-04+srm*(1.3086e-05+srm*(-1.9252e-07+srm*(1.1490e-09))))))
    b=1.0284e+00+srm*(-5.5003e-01+srm*(8.7968e-02+srm*(-6.6913e-03+srm*(2.7656e-04+srm*(-6.3720e-06+srm*(7.6924e-08+srm*(-3.7921e-10)))))))
    return [r, g, b]

ebc = np.arange(start=0, stop=80+1, dtype='int')

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(ebc_to_rgb(i), 0, 1)) for i in ebc], **{'standalone': False})
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
