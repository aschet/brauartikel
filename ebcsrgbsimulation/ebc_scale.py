import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import olfarve
import colour.plotting
import matplotlib.pyplot as plt

max_ebc = 60
ebc_colors = []
for l in range(1, 16):
    for ebc in range(1, max_ebc + 1):
        ebc_colors.append(colour.plotting.ColourSwatch(RGB=olfarve.ebc_to_srgb(ebc, l)))

fig_scale, ax_scale = colour.plotting.plot_multi_colour_swatches(colour_swatches=ebc_colors, columns=max_ebc, **{'show': False, 'tight_layout': False })
ax_scale.xaxis.set_label_text('EBC')
ax_scale.xaxis.set_ticks_position('bottom')
ax_scale.yaxis.set_label_text('l [cm]')
ax_scale.yaxis.set_ticks_position('left')
fig_scale.savefig('ebc_scale.pdf')

#plt.show()
