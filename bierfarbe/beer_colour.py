import numpy as np
import colour.plotting

def ebc_to_rgb(ebc):
    srm = ebc / 1.94
    # SRM to sRGB model, multiply outputs by 255 and clip between 0 and 255
    # 7.5 cm, CIE 1964 10 Degree Standard Observer, D65, 50 SRM
    r=1.0379e+00+srm*(-5.4150e-02+srm*(1.1040e-03+srm*(-8.3300e-06)))
    g=9.7813e-01+srm*(-1.3992e-01+srm*(7.8410e-03+srm*(-1.6906e-04)))
    b=1.0102e+00+srm*(-5.4806e-01+srm*(1.4730e-01+srm*(-2.2536e-02)))
    return [r, g, b]

ebc = np.arange(start=1, stop=81)

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches([colour.plotting.ColourSwatch(RGB=np.clip(ebc_to_rgb(i), 0, 1)) for i in ebc],
    **{'standalone': False, 'tight_layout': True })
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
#fig_scale.savefig('colorscale.pdf')
