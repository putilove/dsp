import matplotlib.pyplot as plt
import numpy as np
import pyreaper
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
import statsmodels.tsa.stattools as st
from collections.abc import Iterable


def my_acf(x, m):
    n = x.size
    mu = (1/n)*sum(x)
    r = 0
    for i in range(0, n - m - 1):
        r += (x[i]-mu) * (x[i+m]-mu)
    r = r*(1/(n-m))
    return r


def test_acf(x):
    m_acf = np.array([my_acf(x, i) for i in range(0, 11)])
    m_acf /= m_acf[0]
    acf = st.acf(x, adjusted=True, nlags=10)

    if np.allclose(acf, m_acf, atol=1e-06):
        return True
    return False


def draw_acf(x, m):
    acf = st.acf(x, adjusted=True, nlags=m)
    plt.figure('acf')
    plt.plot(acf)
    plt.show()


def my_dtft(x, fs, f):
    n = np.arange(len(x))
    omega = (2 * np.pi * f) / fs
    if isinstance(f, Iterable):
        res = np.array([abs(np.dot(np.exp(-1.j * omega[i] * n), x)) for i in range(len(omega))])
    else:
        res = abs(np.dot(np.exp(-1.j * omega * n), x))
    return res


def draw_dftf(x, fs):
    arg = np.arange(40, 500)
    a = my_dtft(x, fs, arg)
    m = np.arange(40, 500, 1)
    plt.figure('dft')
    plt.plot(m, a)
    plt.show()


def psola(x, fs, k, hilbert_transform=True):
    int16_info = np.iinfo(np.int16)
    tmp = x * min(int16_info.min, int16_info.max)
    tmp = tmp.astype(np.int16)

    pm_times, pm, f_times, f, _ = pyreaper.reaper(tmp, fs, do_hilbert_transform=hilbert_transform)
    peaks = pm_times[pm == 1]
    peaks *= fs

    res = np.zeros(round(len(x) * k * 1.1))
    dist = peaks[1] - peaks[0]
    res[:round(peaks[0]-dist)] += x[:round(peaks[0]-dist)]

    s = 0
    T = (1 / np.mean(f[f != -1])) * fs
    first_peak = round(peaks[0])
    for i in range(1, len(peaks)):
        dist = peaks[i] - peaks[i-1]
        if dist > T * 1.5:
            first = first_peak + round(s)
            pie = x[round(peaks[i-1]):round(peaks[i])]
            res[first:round(first + dist)] += pie
            s += dist
        else:
            shift = 2 * (dist * k - dist)
            first = first_peak + round(shift + s)
            pie = x[round(peaks[i] - dist):round(peaks[i]+dist)]
            pie *= np.concatenate((np.linspace(0, 1, round(dist)), np.linspace(1, 0, round(dist))))
            res[first:round(first+2*dist)] += pie

            s += round(shift + dist)

    res /= max(abs(res.min()), abs(res.max()))
    return res


def google_reaper(x, fs):
    t = np.linspace(0, (len(x) - 1) / fs, len(x))
    int16_info = np.iinfo(np.int16)
    x = x * min(int16_info.min, int16_info.max)
    x = x.astype(np.int16)
    pm_times, pm, f_times, f, _ = pyreaper.reaper(x, fs)
    plt.figure('[Reaper] Pitch Marks')
    plt.plot(t, x)
    plt.scatter(pm_times[pm == 1], x[(pm_times * fs).astype(int)][pm == 1], marker='x', color='red')
    plt.figure('[Reaper] Fundamental Frequency')
    plt.plot(f_times, f)
    print('Average fundamental frequency:', np.mean(f[f != -1]))
    plt.show()


def main():
    fs, x = wavfile.read('input.wav')
    x = x.astype(np.float32)
    x /= max(abs(x.min()), abs(x.max()))

    #print(test_acf(x))
    #draw_acf(x, 2000)
    #draw_dftf(x, fs)
    #google_reaper(x, fs)

    k = 0.5
    res = psola(x, fs, k)
    write(f'output_{k}.wav', fs, res)


if __name__ == '__main__':
    main()

