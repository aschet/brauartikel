import numpy as np
import colour.plotting

def ebc_to_rgb(ebc):
    srm = ebc / 1.94
    # SRM to sRGB model, multiply outputs by 255 and clip between 0 and 255
    # 5 cm transmission, CIE 1964 10 Degree Standard Observer, D65 illuminant
    r=1.0362e+00+srm*(-3.5446e-02+srm*(4.4920e-04+srm*(-1.8232e-06)))
    g=9.7869e-01+srm*(-9.3377e-02+srm*(3.4883e-03+srm*(-5.0097e-05)))
    b=1.0043e+00+srm*(-3.5269e-01+srm*(5.8755e-02+srm*(-5.6951e-03)))
    return [r, g, b]

ebc = np.arange(start=1, stop=81)

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(ebc_to_rgb(i), 0, 1)) for i in ebc],
    **{'standalone': False, 'tight_layout': True })
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
#fig_scale.savefig('colorscale.pdf')
