import numpy as np
import matplotlib.pyplot as plt

ebc_values = np.array(range(0, 60))
absorbance_10mm = ebc_values / 25.0

def calc_transmission(absorbance):
    return np.power(np.full(shape=absorbance.shape, fill_value=10.0), absorbance * -1.0)

transmission_10mm = calc_transmission(absorbance_10mm)

fig = plt.figure(constrained_layout=True, figsize=(4, 4))
axes = fig.subplots(1, 1)
axes.set_xlabel('EBC')
axes.set_ylabel('T')
axes.plot(ebc_values, transmission_10mm, label='1.0 cm path length ')
axes.set_xticks(ticks=np.arange(0, len(transmission_10mm), 5), dtype=int)
axes.set_yticks(ticks=np.arange(0, 1.1, 0.1))
fig.savefig('transmission.pdf', format='pdf')

#plt.show()