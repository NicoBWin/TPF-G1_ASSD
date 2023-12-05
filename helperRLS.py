from re import T
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
from IPython.display import Audio, update_display, display
from ipywidgets import IntProgress
import pyroomacoustics as pra
import numpy as np
import scipy.linalg as lin
from numpy.fft import fft, rfft
from numpy.fft import fftshift, fftfreq, rfftfreq


def play(signal, fs):
    audio = ipd.Audio(signal, rate=fs, autoplay=True)
    return audio


def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = sp.spectrogram(w, fs=fs, nperseg=256, nfft=576)
    fig, ax = plt.subplots()
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r',
                  shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)


def getNextPowerOfTwo(len):
    return 2**(len*2).bit_length()


def get_optimal_params(x, y, M):

    N = len(x)
    r = sp.correlate(x, x)/N
    p = sp.correlate(x, y)/N
    r = r[N-1:N-1 + M]
    # Correlate calcula la cross-corr r(-k), y necesitamos r(k), y esto no es par como la autocorrelacion
    p = p[N-1:N-1-(M):-1]
    wo = lin.solve_toeplitz(r, p)

    jo = np.var(y) - np.dot(p, wo)

    NMSE = jo/np.var(y)

    return wo, jo, NMSE


def periodogram_averaging(data, fs, L, padding_multiplier, window):
    wind = window(L)
    # Normalizamos la ventana para que sea asintoticamente libre de bias

    def getChuncks(lst, K): return [lst[i:i + K]
                                    for i in range(0, len(lst), K)][:-1]
    corrFact = np.sqrt(L/np.square(wind).sum())
    wind = wind*corrFact
    dataChunks = getChuncks(data, L)*wind
    fftwindowSize = L*padding_multiplier
    freqs = rfftfreq(fftwindowSize, 1/fs)
    periodogram = np.zeros(len(freqs))
    for i in range(len(dataChunks)):
        # Se van agregando al promediado los periodogramas de cada bloque calculado a partir de la FFT del señal en el tiempo
        periodogram = periodogram + \
            np.abs(rfft(dataChunks[i], fftwindowSize))**2/(L*len(dataChunks))

    return freqs, periodogram, len(dataChunks)


windows = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
           'blackmanharris', 'flattop', 'bartlett', 'barthann',
           'hamming', ('kaiser', 10), ('tukey', 0.25)]


def fxnlms_sim(w0, mu, P, S, S_hat, xgen, sound, orden_filtro, N=10000):
    w = w0
    J = np.zeros(N)
    e = np.zeros(N)
    d_hat = np.zeros(N)
    n = np.arange(0, N, 1, dtype=int)

    x = xgen(n)
    d = sp.lfilter(P[0], P[1], x)
    xf = sp.lfilter(S_hat[0], S_hat[1], x)

    xf = np.concatenate([np.zeros(orden_filtro-1), xf])
    zis = np.zeros(np.max([len(S[0]), len(S[1])])-1)
    ziw = np.zeros(orden_filtro-1)

    f = IntProgress(min=0, max=N)
    display(f)

    i = 0
    display(J[0], display_id='J')
    for n in range(N):

        y, ziw = sp.lfilter(w, [1], [x[n]], zi=ziw)

        y = y[0] + sound(n)

        d_hat_aux, zis = sp.lfilter(S[0], S[1], [y], zi=zis)
        d_hat[n] = d_hat_aux[0]
        e[n] = d[n] + d_hat_aux[0]
        J[n] = e[n] * e[n]
        w = w - mu * xf[n:n + orden_filtro][::-1] * e[n] / \
            (np.linalg.norm(xf[n:n + orden_filtro][::-1])**2 + 1e-6)
        if n//(N//100) > i:
            update_display(J[n], display_id='J')
            f.value = n
            i += 1

    return w, J, e, d, d_hat

def get_model_output(d, W):
    h = calc_h(W)
    
    u = signal.lfilter(h, [1], d)
    v = np.random.normal(0, np.sqrt(sigma2v), size=len(u))
    
    return u + v

def fxrls_sim(lamda, delta, P, S, S_hat, xgen, sound, orden_filtro,  Nsamp, N=10000):

    """
    w0: valor inicial del filtro adaptativo
    lambda: factor de olvido de RLS
    delta: parámetro de regularización de RLS
    M: orden del filtro RLS
    N: iteraciones
    """
    
    w = np.zeros(orden_filtro)
    J = np.zeros(N)
    e = np.zeros(N)
    d_hat = np.zeros(N)
    n = np.arange(0, N, 1, dtype=int)
    Pmat = np.identity(orden_filtro) * (1/delta)

    x = xgen(n)
    d = sp.lfilter(P[0], P[1], x)
    xf = sp.lfilter(S_hat[0], S_hat[1], x)

    zis = np.zeros(np.max([len(S[0]), len(S[1])])-1)
    ziw = np.zeros(orden_filtro-1)

    f = IntProgress(min=0, max=N)
    display(f)

    i = 0
    display(J[0], display_id='J')
    for n in range(N):

        y, ziw = sp.lfilter(w, [1], [x[n]], zi=ziw)

        y = y[0] + sound(n)

        d_hat_aux, zis = sp.lfilter(S[0], S[1], [y], zi=zis)
        d_hat[n] = d_hat_aux[0]
        e[n] = d[n] + d_hat_aux[0]
        J[n] = e[n] * e[n]
        if n >= orden_filtro:
            gbar = 1/lamda * np.matmul(Pmat, np.flip(xf[n - orden_filtro + 1: n + 1]))
            alphabar = 1 + np.dot(gbar, np.flip(xf[n - orden_filtro + 1: n + 1]))
            g = gbar / alphabar
            Pmat = Pmat/lamda - np.outer(g, gbar)
            Pmat = (Pmat + Pmat.T)/2
            w = w - g * np.conjugate(e[n])
            
        if n//(N//100) > i:
            update_display(J[n], display_id='J')
            f.value = n
            i += 1

    return w, J, e, d, d_hat


def plot_results(results, P, S, S_hat, xgen, soundgen, compare_S_Shat=False):
    w, J, e, d, d_hat = results
    ran = [x / 48000 for x in range(len(e))]
    plt.figure(figsize=(10, 5))

    plt.grid()
    plt.title('SE vs n')
    plt.xlabel('n')
    plt.ylabel('Square error')
    plt.semilogy(ran, J)

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title('Señal error vs t')
    plt.xlabel('t [s]')
    plt.ylabel('modulo del error [dB]')
    plt.plot(ran, 20*np.log(np.abs(e)))
    plt.ylim([-100,80])

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title('Señal deseada negada y deseada estimada')
    plt.xlabel('n')
    plt.ylabel('Amplitud')
    plt.plot(d_hat, label='Señal deseada estimada')
    plt.plot(-d, label='Señal deseada negada')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.grid()
    freqs, s = sp.freqz(P[0], S[0], fs=48000, worN=10000)
    freqw, wps = sp.freqz(w, [1], fs=48000, worN=10000)
    plt.title('Comparativa entre módulo de W(z) y P(z)/S(z)')
    plt.semilogx(freqs, 20*np.log10(np.abs(s)), label='P(z)/S(z)')
    plt.semilogx(freqw, 20*np.log10(np.abs(wps)), label='W(z)')
    plt.xlabel('f [Hz]')
    plt.ylabel('Modulo [dB]')
    plt.xlim([0,20000])
    plt.legend()

    if compare_S_Shat:
        plt.figure(figsize=(10, 5))
        plt.grid()
        plt.subplot(211)
        freqs, s = sp.freqz(S[0], S[1], fs=48000, worN=10000)
        freqw, s_hat = sp.freqz(S_hat[0], S_hat[1], fs=48000, worN=10000)
        plt.title('Comparativa entre modulo de S(z) y S_hat(z)')
        plt.semilogx(freqw, 20*np.log10(np.abs(s)), label='S(z)')
        plt.semilogx(freqs, 20*np.log10(np.abs(s_hat)), label='S_hat(z)')
        plt.legend()

        plt.subplot(212)
        plt.title('Comparativa entre modulo de S(n) y S_hat(n)')
        plt.plot(S[0], label='S(n)')
        plt.plot(S_hat[0], label='S_hat(n)')
        plt.legend()

    n = np.arange(0, len(e))
    x = xgen(n)
    sound = soundgen(n)
    max = np.max([np.max(np.abs(e)), np.max(np.abs(sound)),
                 np.max(np.abs(x)), np.max(np.abs(x + sound))])
    display('Señal error:', Audio(e/max, rate=48000, normalize=False))
    display('Señal interferencia:', Audio(
        x/max, rate=48000, normalize=False))
    display('Señal interferencia + sonido:',
            Audio((x + sound)/max, rate=48000, normalize=False))

def plot_compare(resultsLMS, resultsRLS, P, S, xgen, soundgen):
    w_lms, J_lms, e_lms, d_lms, d_hat_lms = resultsLMS
    w_rls, J_rls, e_rls, d_rls, d_hat_rls = resultsRLS
    ran = [x / 48000 for x in range(len(e_rls))]
    plt.figure(figsize=(10, 5))

    plt.grid()
    plt.title('SE vs n')
    plt.xlabel('t [s]')
    plt.ylabel('Square error')
    plt.semilogy(ran, J_lms, alpha=0.45, label='J LMS')
    plt.semilogy(ran, J_rls, alpha=0.45, label='J RLS')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.title('Señal error vs t')
    plt.xlabel('t [s]')
    plt.ylabel('modulo del error [dB]')
    plt.plot(ran, 20*np.log(np.abs(e_lms)), alpha=0.45, label='error LMS')
    plt.plot(ran, 20*np.log(np.abs(e_rls)), alpha=0.45, label='error RLS')
    plt.ylim([-100,80])
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.grid()
    freqs, s = sp.freqz(P[0], S[0], fs=48000, worN=10000)
    freqw_lms, wps_lms = sp.freqz(w_lms, [1], fs=48000, worN=10000)
    freqw_rls, wps_rls = sp.freqz(w_rls, [1], fs=48000, worN=10000)
    plt.title('Comparativa entre módulo de W(z) y P(z)/S(z)')
    plt.semilogx(freqs, 20*np.log10(np.abs(s)), label='P(z)/S(z)')
    plt.semilogx(freqw_lms, 20*np.log10(np.abs(wps_lms)), label='$W_{LMS}(z)$', alpha = 0.7)
    plt.semilogx(freqw_rls, 20*np.log10(np.abs(wps_rls)), label='$W_{RLS}(z)$', alpha = 0.7)
    plt.xlabel('f [Hz]')
    plt.ylabel('Modulo [dB]')
    plt.xlim([20,20000])
    plt.legend()

    n = np.arange(0, len(e_lms))
    x = xgen(n)
    sound = soundgen(n)
    display('Señal error LMS:', Audio(e_lms, rate=48000))
    display('Señal error RLS:', Audio(e_rls, rate=48000))
    display('Señal interferencia:', Audio(
        x, rate=48000))
    display('Señal interferencia + sonido:',
            Audio((x + sound), rate=48000))


def createRoom(print):

    fs = 48000

    x = np.random.randn(fs*5)  # Generamos ruido gaussiano
    y = np.random.randn(fs*5)  # Mic de cancelacion
    # Seteamos los materiales de la habitacion
    m = pra.Material('rough_concrete')

    height = 0.1
    roomCorners = np.array([[-1.0, -1.0, 0.4, 0.2, 0.2+np.sqrt(2)/20, 0.4+np.sqrt(2)/10, 0.6, 0.6],
                            [0.1, 0.2, 0.2, 0.4, 0.4+np.sqrt(2)/20, 0.2, 0.2, 0.1]])
    room = pra.Room.from_corners(
        roomCorners, fs=fs, materials=m, max_order=3, ray_tracing=True, air_absorption=True)
    room.extrude(height, materials=m)

    # Agregamos la fuente y el microfono
    micError = np.array([0.6, 0.15, height/2])
    micRef = np.array([-0.6, 0.15, height/2])
    micArray = pra.beamforming.MicrophoneArray(
        R=np.array([micRef, micError]).T, fs=fs)
    room.add_microphone_array(micArray)

    room.add_source([-1.0, 0.15, height/2], signal=x)
    room.add_source([0.2+np.sqrt(2)/40, 0.4+np.sqrt(2)/40, height/2], signal=y)

    if (print):
        # Mostramos la habitacion
        fig, ax = room.plot(mic_marker_size=50)
        ax.set_box_aspect((np.ptp([0,1]), np.ptp([0,0.5]), np.ptp([0,0.5])))
        ax.set_xlim([-1, 0.6])
        ax.set_ylim([-0, 0.6])
        ax.set_zlim([0, 0.2])
        plt.show()

    return room


def getProomImpulse():

    fs = 48000
    m = pra.Material('rough_concrete')

    height = 0.1
    roomCorners = np.array([[-1.0, -1.0, 0.4, 0.2, 0.2+np.sqrt(2)/20, 0.4+np.sqrt(2)/10, 0.6, 0.6],
                            [0.1, 0.2, 0.2, 0.4, 0.4+np.sqrt(2)/20, 0.2, 0.2, 0.1]])
    room = pra.Room.from_corners(
        roomCorners, fs=fs, materials=m, max_order=3, ray_tracing=True, air_absorption=True)
    room.extrude(height, materials=m)

    # Agregamos la fuente y el microfono
    micError = np.array([0.6, 0.15, height/2])
    sourceRef = np.array([-0.6, 0.15, height/2])
    room.add_microphone(micError)

    room.add_source(sourceRef,)

    # Computamos y mostramos la respuesta al impulso
    room.compute_rir()

    return room.rir
