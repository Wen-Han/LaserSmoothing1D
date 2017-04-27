import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# input starts
# total number of beamlets
m = 100
# laser bandwidth due to phase modulation, defined as HWHM, in laser frequency
lsr_bw = 0.00117
# amplitude of phase modulation
pm_am = np.pi
# number of color cycle
ncc = 1.0
# number of fm frequencies
multi_fm = 2
# max time of the time series
tmax = float(550000)
# if True then plot the laser bandwidth instead of phase modulation bandwidth
plot_laser_bandwidth = True
# if True then do SMA on the AR(1) time series
# to approximate Gaussian PSD (experimental)
if_sma = True
# how many ensemble average to carry out
n_ens = 100
# what time series to generate. any of the following (can be multiple):
# 'sin, ar1, gaussian, reference'
time_series = 'sin, ar1, gaussian, reference'
# type of plot semilogy or linear plot
plot = plt.plot  # or plt.semilogy
# debug - plot phase/amplitude and autocorrelation function
# it's up to the user where to put the plot_autocorr function
if_debug = False
# ------------------------------------------------------------------------------
# input ends

tu = 0.01 / lsr_bw
tn = int(tmax / tu)


# ISI (order 1 autoregressive)
def ar1(b, sigma, pha, num=m):
    return b * pha + sigma * np.random.normal(size=num)
    # return b * pha + np.random.normal(scale=sigma, size=num)


# SMA
def sma1d(pha, num):
    # n = np.size(pha, axis=-1)
    ret = np.cumsum(pha)
    ret[num:] = ret[num:] - ret[:-num]
    return ret / num


# autocorrelation
def autocorr(x):
    """ For debugging. autocorrelation of 1d array x
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]/result[result.size/2]


def plot_autocorr(axes, pha):
    """ For debugging. plot 1d series and its autocorrelation
    
    :param axes: plot axes handle
    :param pha: 1d series
    """
    plt.sca(axes[0])
    plot(time, pha)
    plt.sca(axes[1])
    # plt.semilogy(time, autocorr(pha))
    plot(time, autocorr(pha))


rph = np.random.uniform(-np.pi, np.pi, m)
rph_t = np.zeros((m, tn))
ft_co = np.zeros((m, tn))
s = 2 * ncc * np.pi
sma_tag = ''
# plot the spectrum of the phase modulation
omega = np.fft.fftshift(np.fft.fftfreq(tn, d=tu))
time = tu * np.arange(tn) * 1.86e-4   # in picoseconds


sinw_ens = 0
ar_ens = 0
gs_ens = 0
ar_ens_rms = 0
gs_ens_rms = 0
ar_ens_mean = 0
gs_ens_mean = 0
time_series = time_series.upper()
if if_debug:
    acf = np.zeros(tn)
    f, ax = plt.subplots(2, 1)
for ens in range(0, n_ens):
    # # sinusoidal
    # bandwidth of phase modulation, in \omega_0
    if 'SIN' in time_series:
        pm_bw = 0.5 * lsr_bw / pm_am / multi_fm
        fm_am = pm_am
        sinw = fm_am * np.sin(pm_bw * np.arange(0, tn) * tu)
        for bwi in range(2, multi_fm + 1):
            sinw += fm_am * np.sin(pm_bw * bwi * np.arange(0, tn) * tu)
        if plot_laser_bandwidth:
            sinw = np.sin(sinw)
        sinw_ens += np.square(np.abs(np.fft.fftshift(np.fft.fft(sinw)))) / n_ens

    # # # AR(1) + (optional) Simple moving average
    if 'AR1' in time_series:
        tn_ar = tn
        if if_sma:
            # generate a longer series to avoid edge effect of SMA
            n = int(1 / (lsr_bw / pm_am ** 2) / tu)
            tn_ar = tn + 2 * n
        pm_bw = 0.5 * lsr_bw / (pm_am * pm_am)
        arcoeff1 = np.exp(- tu * pm_bw)
        arcoeff2 = np.sqrt(1 - arcoeff1 * arcoeff1) * pm_am
        phase = np.zeros(tn_ar)
        # Discard the first part of the random sequence
        for ti in range(256):
            phase[0] = ar1(arcoeff1, arcoeff2, phase[0], num=1)
        for zi in range(1, tn_ar):
            phase[zi] = ar1(arcoeff1, arcoeff2, phase[zi - 1], num=1)
        # SMA
        if if_sma:
            denor = (arcoeff1 * arcoeff1 * (n * arcoeff1 - n + 2) -
                     2 * np.power(arcoeff1, n + 1) * (
                         arcoeff1 - 1) + n - arcoeff1 * (
                         n + 2))
            nor = ((1 - arcoeff1 * arcoeff1) * np.square(1 - arcoeff1) * n * n)
            var = np.sqrt(nor / denor)
            phase = sma1d(phase, n) * var
            sma_tag = 'SMA'
            phase = phase[n:tn + n]
        # plot_autocorr(ax, phase)
        if plot_laser_bandwidth:
            phase = np.sin(phase)
        # plot_autocorr(ax, phase)
        # # ============= debug only ===============
        # acf += autocorr(phase)
        # # ============= debug only ===============
        ar_ens += np.square(np.abs(np.fft.fftshift(
            np.fft.fft(phase)))) / n_ens
        ar_ens_mean += np.mean(phase)
        ar_ens_rms += np.mean(np.square(phase))

    # gaussian
    if 'GAUSSIAN' in time_series:
        pm_bw = 0.5 * lsr_bw / pm_am
        rand_ph = np.random.normal(scale=np.pi, size=tn)
        psd = np.exp(-np.log(2) * 0.5 * np.square(omega / pm_bw * 2 * np.pi))
        psd *= np.sqrt(2 * tn) / np.sqrt(np.mean(np.square(psd))) * pm_am
        phase3 = np.array(psd) * (np.cos(rand_ph) + 1j * np.sin(rand_ph))
        phase3 = np.real(np.fft.ifft(np.fft.ifftshift(phase3)))
        # plot_autocorr(ax, phase3)
        if plot_laser_bandwidth:
            phase3 = np.sin(phase3)
            psd = np.exp(-np.log(2) * np.square(omega / lsr_bw * 4 * np.pi))
        # # ============= debug only ===============
        # acf += autocorr(phase3)
        # # ============= debug only ===============
        gs_ens += np.square(np.abs(np.fft.fftshift(
            np.fft.fft(phase3)))) / n_ens
        gs_ens_mean += np.mean(phase3)
        gs_ens_rms += np.mean(np.square(phase3))

plt.figure()
if 'SIN' in time_series:
    plot(omega, sinw_ens, label='Sin')
if 'AR1' in time_series:
    plot(omega, ar_ens, label='AR(1)' + sma_tag)
    am_bw_sq = np.square(lsr_bw * 0.25 / np.pi)
    theo = am_bw_sq / (np.square(omega) + am_bw_sq)
    plot(omega, np.mean(ar_ens) / np.mean(theo) * theo, label='Lorentzian')
    print 'AR: RMS=', np.sqrt(ar_ens_rms) / np.sqrt(n_ens)
    print 'AR: mean=', np.mean(ar_ens_mean) / n_ens
if 'GAUSSIAN' in time_series:
    plot(omega, gs_ens, label='Gaussian sim.')
    plot(omega, np.mean(gs_ens) / np.mean(psd) * psd,
             label='Gaussian theo.')
    print 'Gaussian: RMS=', np.sqrt(gs_ens_rms) / np.sqrt(n_ens)
    print 'Gaussian: mean=', np.mean(gs_ens_mean) / n_ens
if 'REFERENCE' in time_series:
    sinbench = np.sin(0.5 * lsr_bw * np.arange(0, tn) * tu)
    sin_fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(sinbench))))
    plot(omega, sin_fft, label='reference')


plt_range = 0.005
plt.xlim([-plt_range, plt_range])
plt.legend()
plt.xlabel('$f (\omega_0)$')
plt.ylabel('PSD (A.U.)')

if if_debug:
    plt.figure()
    plot(time, np.abs(acf) / n_ens, linewidth=2)
    plt.sca(ax[0])
    plt.axis('tight')
    plt.sca(ax[1])
    plt.axis('tight')
    plt.xlabel('time (ps)')

plt.show()
