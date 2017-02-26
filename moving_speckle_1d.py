import numpy as np
import matplotlib.pyplot as plt
#
# This script shows how the amplitude of an 1D laser antenna with ISI-type
# speckle looks like in a small simulation box. The random phase is modeled
# by an autoregressive order 1 process
# ==============================================================================


# number of speckle modes
n = 31
# size of the antenna
ly = 1600
# time step of the simulation
dt = 0.14
# bandwidth of phase modulation
pm_bw = 0.0002
# amplitude of phase modulation
pm_am = np.pi
# time interval between speckle pattern updates
tu = 0.01 / (pm_bw * pm_am)
# maximum time to show in the plot
tmax = 4000.0
# ------------------------------------------------------------------------------

dk = 2.0*np.pi/ly
rph = np.random.uniform(-np.pi, np.pi, n+1)
y = np.arange(0, ly)


def ar1(b, sigma, phase):
    return b * phase + sigma * np.random.normal(size=n+1)


tnmax = int(np.floor(tmax/tu))
n_update = round(tu/dt)
b = np.exp(- n_update * dt * pm_bw)
sigma = np.sqrt(1 - b * b) * pm_am
efld = np.zeros((tnmax, np.size(y, 0)))
for ti in range(1, tnmax):
    rph = ar1(b, sigma, rph)
    for i in range(n/2-n+1, n/2):
        efld[ti, :] = efld[ti, :] + np.sin(i*dk*y+rph[i+n-n/2])
im0 = plt.imshow(abs(efld), extent=[0, ly, 0, dt*n_update*tnmax])

plt.title('envelope of E field')
plt.xlabel('$x (c/\omega_0)$')
plt.ylabel('$t (1/\omega_0)$')
plt.axis('auto')
plt.show()


