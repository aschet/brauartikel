import numpy as np
import colour.plotting

def ebc_to_rgb(ebc):
    # EBC to sRGB model, multiply outputs by 255 and clip between 0 and 255
    # glas diameter: 7.5 cm
    # observer: CIE 1964 10 Degree Standard Observer
    # illuminant: D65
    # scale: 80 EBC
    r=1.0149e+00+ebc*(-2.0207e-02+ebc*(-3.1471e-04+ebc*(1.7334e-05+ebc*(-2.3906e-07+ebc*(1.1090e-09)))))
    g=9.9965e-01+ebc*(-8.3923e-02+ebc*(3.6936e-03+ebc*(-1.0839e-04+ebc*(1.9421e-06+ebc*(-1.8311e-08+ebc*(6.8745e-11))))))
    b=1.0531e+00+ebc*(-2.8811e-01+ebc*(2.4180e-02+ebc*(-9.7637e-04+ebc*(2.1588e-05+ebc*(-2.6749e-07+ebc*(1.7427e-09+ebc*(-4.6448e-12)))))))
    return [r, g, b]

ebc = np.arange(start=0, stop=80)

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(ebc_to_rgb(i), 0, 1)) for i in ebc],
    **{'standalone': False, 'tight_layout': True })
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
#fig_scale.savefig('colorscale.pdf')
