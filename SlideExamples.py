# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

# Slide 9
mp = 0.115
vp = 35.0
mo = 0.265
vo = 0.0
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 10
mp = 0.115
vp = 114.13
mo = 0.265
vo = 0.0
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 11
mp = 0.115
vp = 114.13 * 0.44704  # mph to m/s
mo = 0.265
vo = 0.0
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 12
mp = 0.115  # kg
vp = 114.13 * 0.44704  # mph to m/s
mo = 0.265  # kg
vo = 0.0  # m/s
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 15
import quantities as pq
mp = 0.115 * pq.kg
vp = 35.0 * (pq.m / pq.s)
mo = 0.265 * pq.kg
vo = 0.0 * (pq.m / pq.s)
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 16
import quantities as pq
mp = 0.115 * pq.kg
vp = 114.13 * (pq.mi / pq.h)
mo = 0.265 * pq.kg
vo = 0.0 * (pq.m / pq.s)
v = (mp * vp + mo * vo) / (mp + mo)
print v

# <codecell>

# Slide 17
import quantities as pq
mp = 0.115 * pq.kg
vp = 114.13 * (pq.mi / pq.h)
mo = 0.265 * pq.kg
vo = 0.0 * (pq.m / pq.s)
v = (mp * vp + mo * vo) / (mp + mo)
print v.rescale(pq.km / pq.h)

# <codecell>

# Slide 18
import quantities as pq
mp = 6.0 * pq.ounce
vp = 114.13 * (pq.mi / pq.h)
mo = 0.265 * pq.kg
vo = 0.0 * (pq.m / pq.s)
v = (mp * vp + mo * vo) / (mp + mo)
print v.rescale(pq.km / pq.h)

# <codecell>

# Slide 20
import quantities as pq
mp = 6.0 * pq.ounce
vp = 114.13 * (pq.mi / pq.h)
mo = 0.265 * pq.kg
vo = 0.0
v = (mp * vp + mo * vo) / (mp + mo)

# <codecell>

# Slides 24-25
import quantities as pq
bags = pq.UnitQuantity('milk bags', pq.L * (4. / 3.))
person = pq.UnitQuantity('person', pq.dimensionless * 1)
need = 200 * (pq.mL / person)
need *= 250 * person
print need
print need.rescale(pq.L)
print need.rescale(bags)
import numpy as np
print np.ceil(need.rescale(bags))

# <codecell>

# Slide 34
import numpy as np
import quantities as pq

def load_spikes(filename):
    # Dummy implementation: just make a spike train
    spikes = np.random.uniform(0, 20 * pq.s, size=20 * pq.s * 10 * pq.Hz)
    spikes = np.append(spikes, np.random.normal(11.0 * pq.s, size=10 * pq.Hz * 5))
    return np.sort(spikes)

def time_slice(spikes, tstart, tend):
    return spikes[np.logical_and(tstart <= spikes, spikes < tend)]

def raster_plot(spikes):
    plt.scatter(spikes, np.zeros(spikes.shape), color='k', marker='|')
    plt.xlim(10.5 * pq.s, 13.0 * pq.s)
    plt.xlabel("Time in " + spikes.dimensionality.string)
    plt.tight_layout()

spikes = load_spikes('spikes.csv') * pq.s
event = 11 * pq.s
window = (-0.5, 2) * pq.s
perievent = time_slice(spikes, *(event + window))
raster_plot(perievent)

# <codecell>

# Slide 35
import numpy as np
import quantities as pq

def load_spikes(filename):
    # Dummy implementation: just make a spike train
    spikes = np.random.uniform(0, 20 * pq.s, size=20 * pq.s * 10 * pq.Hz)
    spikes = np.append(spikes, np.random.normal(11.0 * pq.s, size=10 * pq.Hz * 5))
    return np.sort(spikes)

def time_slice(spikes, tstart, tend):
    return spikes[np.logical_and(tstart <= spikes, spikes < tend)]

def raster_plot(spikes):
    plt.scatter(spikes, np.zeros(spikes.shape), color='k', marker='|')
    plt.xlim(10500 * pq.ms, 13000 * pq.ms)  # This also needs changing
    plt.xlabel("Time in " + spikes.dimensionality.string)
    plt.tight_layout()

spikes = (load_spikes('spikes.csv') * pq.s).rescale(pq.ms)
event = 11 * pq.s
window = (-0.5, 2) * pq.s
perievent = time_slice(spikes, *(event + window))
raster_plot(perievent)

# <codecell>

# Slide 36
import numpy as np
import quantities as pq

def load_spikes(filename):
    # Dummy implementation: just make a spike train
    spikes = np.random.uniform(0, 20 * pq.s, size=20 * pq.s * 10 * pq.Hz)
    spikes = np.append(spikes, np.random.normal(11.0 * pq.s, size=10 * pq.Hz * 5))
    return np.sort(spikes)

def bin_spikes(spikes, binsize, tstart, tend):
    binsize.units = tstart.units
    bins = np.arange(
      tstart, tend + binsize, binsize)
    return np.histogram(spikes, bins=bins)[0]

def binned_plot(bins):
    plt.bar(range(len(bins)), bins)
    plt.xlabel("Time bin")
    plt.xlim(0, bins.shape[0])

spikes = load_spikes('spikes.csv') * pq.s
event = 11 * pq.s
window = (-0.5, 2) * pq.s
binned = bin_spikes(spikes, 20 * pq.ms, *(event + window))
binned_plot(binned)

# <codecell>


