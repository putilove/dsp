import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.interpolate import interp1d as interpolation
from math import ceil
from math import floor
import matplotlib.pyplot as plt


def signal(n, s, dt):
    np.random.seed(s)
    arr = np.array([np.random.random()*np.exp(1.j*np.random.uniform(0, 2*np.pi)) for _ in range(0, 50)])
    zero = np.zeros(1, dtype=complex)
    zero[0] = dt*100*np.random.random() + 0.j
    reverse = np.array([np.conjugate(arr[len(arr)-i-1]) for i in range(0, len(arr))])
    empty_size = floor((n - 1 - 100)/2)
    empty = np.zeros(empty_size, dtype=complex)
    if(n-1) % 2 == 0:
        sp = np.concatenate((zero, empty, reverse, arr, empty))
    else:
        sp = np.concatenate((zero, empty, reverse, zero, arr, empty))
    print(sp[floor(len(sp)/2)])
    sig = np.fft.ifft(sp)
    sig = np.real(sig)
    return sig


def sincinterpolation(f, x, need_time, index):
    res = 0
    for i in range(-round(6/2), round(6/2)):
        if i >= len(x) - 1:
            break
        if i < 0:
            continue
        res += x[index+i] * np.sinc((need_time-f[index+i]) / (f[index+i+1] - f[index+i]))
    return res


def shift_v3(x, fs, dt, s):
    sig = signal(len(x), s, dt)
    shifted_time = np.array([i / fs + dt + sig[round(i)] for i in range(len(x))])
    shifted_time[0] = dt
    if np.all(np.diff(shifted_time) < 0):
        print('some troubles')
    shifted_x = np.zeros(len(x))
    shifted_x[ceil(dt * fs):] = [sincinterpolation(shifted_time, x, i / fs + dt, i) for i in range(len(x) - ceil(dt * fs))]
    return shifted_x, sig


def shift_v2(x, fs, dt, s):
    sig = signal(len(x), s, dt)
    shifted_time = np.array([i / fs + dt + sig[round(i)] for i in range(len(x))])
    shifted_time[0] = dt
    if np.all(np.diff(shifted_time) < 0):
        print('some troubles')
    f = interpolation(shifted_time, x)
    shifted_x = np.zeros(len(x))
    shifted_x[ceil(dt * fs):] = [f(i / fs) for i in range(ceil(dt * fs), len(x))]
    return shifted_x


def shift(x, fs, dt, at, f):
    shifted_time = np.array([i/fs + dt + at*np.sin(2*np.pi*f*i/fs) for i in range(len(x))])
    if np.all(np.diff(shifted_time) < 0):
        print('some troubles')
    f = interpolation(shifted_time, x)
    shifted_x = np.zeros(len(x))
    shifted_x[ceil(dt*fs):] = [f(i/fs) for i in range(ceil(dt * fs), len(x))]
    return shifted_x


def main():
    fs, x = wavfile.read('input.wav')
    x = x.astype(np.float32)
    x /= max(abs(x.min()), abs(x.max()))
    k = 0.4

    shifted, sig = shift_v3(x, fs, 0.02, 0)
    x += k*shifted
    plt.figure('signal 1')
    plt.plot(sig)

    shifted, sig = shift_v3(x, fs, 0.1, 1)
    x += k*shifted
    plt.figure('signal 2')
    plt.plot(sig)

    shifted, sig = shift_v3(x, fs, 0.2, 2)
    x += k * shifted
    plt.figure('signal 3')
    plt.plot(sig)

    x /= max(abs(x.max()), abs(x.min()))
    write(f'output_test.wav', fs, x)
    plt.show()


if __name__ == '__main__':
    main()
