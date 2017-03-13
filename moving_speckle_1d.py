import numpy as np
import matplotlib.pyplot as plt
#
# This script shows how the amplitude of an 1D laser antenna with ISI-type
# and FM SSD speckle looks like in a small simulation box.
#
# The random phases of ISI-type beamlets are modeled by an autoregressive
# order 1 process.
#
# The distribution of amplitude of the sinusoidal phase modulation is
# assumed to be uniform
# ==============================================================================


# number of speckle modes in simulation box
n = 32
# total number of speckle modes
m = 256
# size of the antenna
ly = 1600
# time step of the simulation
dt = 0.14
# bandwidth of phase modulation
pm_bw = 0.002
# amplitude of phase modulation
pm_am = np.pi
# time interval between speckle pattern updates
tu = 0.01 / (pm_bw * pm_am)
# maximum time to show in the plot
tmax = 4000.0
# number of color cycle
ncc = 1.0
# number of fm (sinusoidal phase) modulation frequency
nfm = 1
# ------------------------------------------------------------------------------

dk = 2.0*np.pi*float(n)/ly
y = np.arange(0, ly)
xk = np.arange(-0.5, 0.5, 1.0/m)
tnmax = int(np.floor(tmax/tu))
n_update = round(tu/dt)
efld = np.zeros((tnmax, np.size(y, 0)))


# ISI (order 1 autoregressive)
def ar1(b, sigma, phase):
    return b * phase + sigma * np.random.normal(size=m)

arcoeff1 = np.exp(- n_update * dt * pm_bw)
arcoeff2 = np.sqrt(1 - arcoeff1 * arcoeff1) * pm_am
rph = np.random.uniform(-np.pi, np.pi, m)
rph_t = np.zeros(tnmax)
for ti in range(0, tnmax-1):
    rph = ar1(arcoeff1, arcoeff2, rph)
    rph_t[ti] = rph[0]
    for i in range(0, m-1):
        efld[ti, :] = efld[ti, :] + np.sin(xk[i]*dk*y+rph[i])
im0 = plt.imshow(abs(efld), extent=[0, ly, 0, dt*n_update*tnmax])
plt.title('envelope of E field')
plt.xlabel('$x (c/\omega_0)$')
plt.ylabel('$t (1/\omega_0)$')
plt.axis('auto')
plt.figure()
plt.plot(np.real(np.fft.fftshift(np.fft.fft(rph_t))))
plt.title('FT of the phase of one beamlet')

# ISI (linear interpolation of white noise, i.e. MA(0) process)
rph = np.random.uniform(-np.pi, np.pi, m)
ntu = int(1.0 / (tu * pm_bw))
rph_t = np.zeros(tnmax)
efld.fill(0)
for ti in range(0, tnmax-1):
    if ti % ntu == 0:
        rph_next = np.random.uniform(-pm_am, pm_am, m)
        dwdt = rph_next / ntu
    rph += dwdt
    rph_t[ti] = rph[0]
    for i in range(0, m-1):
        efld[ti, :] = efld[ti, :] + np.sin(xk[i]*dk*y+rph[i])
plt.figure()
im0 = plt.imshow(abs(efld), extent=[0, ly, 0, dt*n_update*tnmax])
plt.title('envelope of E field')
plt.xlabel('$x (c/\omega_0)$')
plt.ylabel('$t (1/\omega_0)$')
plt.axis('auto')
plt.figure()
plt.plot(np.real(np.fft.fftshift(np.fft.fft(rph_t))))
plt.title('FT of the phase of one beamlet')

# FM SSD
rpp_rph = np.random.uniform(-np.pi, np.pi, m)
s = 2 * ncc * np.pi / pm_bw
# fmBand = np.random.uniform(0, 2*pm_bw, nfm-1)
fmph = np.random.uniform(-np.pi, np.pi, nfm-1)
efld.fill(0)
dph = 2.0 / nfm
for ti in range(1, tnmax):
    rph = pm_am * np.sin(2 * pm_bw * (ti - s * xk))
    for ib in range(1, nfm - 1):
        rph += pm_am * np.sin(ib * dph * pm_bw * (ti - s * xk) + fmph[ib])
    rph = rph / np.sqrt(nfm) + rpp_rph
    rph_t[ti] = rph[0]
    for i in range(0, m-1):
        efld[ti, :] = efld[ti, :] + np.sin(xk[i]*dk*y+rph[i])
plt.figure()
im1 = plt.imshow(abs(efld), extent=[0, ly, 0, dt*n_update*tnmax])
plt.title('envelope of E field')
plt.xlabel('$x (c/\omega_0)$')
plt.ylabel('$t (1/\omega_0)$')
plt.axis('auto')
plt.figure()
plt.plot(np.real(np.fft.fftshift(np.fft.fft(rph_t))))
plt.title('FT of the phase of one beamlet')

plt.show()

