# Based on an implementation by Thomas Mansencal
# https://stackoverflow.com/questions/58722583/how-do-i-convert-srm-to-lab-using-e-308-as-an-algorithm

import colour
import colour.plotting
import numpy as np

colour.utilities.describe_environment()
colour.plotting.colour_style()

ASBC_SHAPE = colour.SpectralShape(380, 780, 5)
OBSERVER = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
ILLUMINANT = colour.SDS_ILLUMINANTS['C']
ILLUMINANT_XY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']

# Compute the beer transmission spectral distribution for given SRM and path length
def beer_transmission_sd(SRM, path=1, shape=ASBC_SHAPE):
    e = np.exp(1)
    wl = shape.range()
    values = np.exp(-(SRM / 12.7) * (0.018747 * e**(-(wl - 430) / 13.374) +
        0.98226 * e** (-(wl - 430) / 80.514)) * path)
    return colour.SpectralDistribution(
        values, wl, name='Beer - SRM {0} - Path {1}'.format(SRM, path))

PATHS = np.arange(1, 11, dtype='int')
EBC = np.arange(0, 65, step=5, dtype='int')
EBC[0] = 1
XYZ = []

# Convert the spectral distribution to CIE XYZ tristimulus values using the
# integration method for the CIE 1964 10 Degree Standard Observer and Illuminant C
for i in PATHS:
    for j in EBC:
        XYZ.append(colour.sd_to_XYZ(beer_transmission_sd(SRM=j / 1.94, path=i),
            cmfs=OBSERVER, illuminant=ILLUMINANT) / 100.0)

# Convert the CIE XYZ tristimulus values to sRGB via CIE Lab
RGB = colour.XYZ_to_sRGB(XYZ, illuminant=ILLUMINANT_XY)

figure, axes = colour.plotting.plot_multi_colour_swatches(
    [colour.plotting.ColourSwatch(RGB=np.clip(i, 0, 1)) for i in RGB],
    columns=len(EBC),
    **{
        'standalone': False,
    })

axes.yaxis.set_label_text('Pfad (cm)')
axes.yaxis.set_ticks(PATHS)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_label_text('EBC')
axes.xaxis.set_ticks_position('bottom')
axes.xaxis.set_ticks(np.arange(1, len(EBC) + 1, dtype='int'), EBC)

colour.plotting.render(standalone=True)
