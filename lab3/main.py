import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.interpolate import interp1d as interpolation
from math import ceil
from math import floor


'''y = np.array([i/fs for i in range(len(x))])
    for i in range(1, len(tmp)):
        min_idx = i
        while y[i] < z[min_idx]:
            if min_idx <= 0:
                min_idx += 1
                break
            min_idx -= 1
        max_idx = i
        while y[i] > z[max_idx]:
            if max_idx >= len(tmp):
                max_idx -= 1
                break
            max_idx += 1
        if min_idx >= max_idx:
            continue
        tmp[i] += x[min_idx]+(x[max_idx]-x[min_idx])/(z[max_idx]-z[min_idx]) * (i/fs - z[min_idx])'''


def signal(n, s):
    np.random.seed(s)
    arr = np.array([np.random.random()*np.exp(1.j*np.random.uniform(0, 2*np.pi)) for _ in range(0, 50)])
    zero = np.zeros(1, dtype=complex)
    zero[0] = 10*np.random.random() + 0.j
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
    sig = signal(len(x), s)
    shifted_time = np.array([i / fs + dt + sig[round(i)] for i in range(len(x))])
    shifted_time[0] = dt
    if np.all(np.diff(shifted_time) < 0):
        print('some troubles')
    shifted_x = np.zeros(len(x))
    shifted_x[ceil(dt * fs):] = [sincinterpolation(shifted_time, x, i / fs + dt, i) for i in range(len(x) - ceil(dt * fs))]
    return shifted_x


def shift_v2(x, fs, dt, s):
    sig = signal(len(x), s)
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
    x_copy = x
    k = 0.4
    #x += k*shift_v2(x, fs, 0.02, 0)
    #x += k*shift_v2(x, fs, 0.1, 1)
    #x += k*shift_v2(x, fs, 0.2, 2)
    #x /= max(abs(x.max()), abs(x.min()))
    #write(f'output.wav', fs, x)

    #x_copy += k*shift(x_copy, fs, 0.02, 0.01, 3)
    #x_copy += k * shift(x_copy, fs, 0.08, 0.01, 4)
    #x_copy += k * shift(x_copy, fs, 0.12, 0.01, 5)
    #x_copy /= max(abs(x_copy.max()), abs(x_copy.min()))
    #write(f'output2.wav', fs, x_copy)

    x += k*shift_v3(x, fs, 0.02, 0)
    x += k*shift_v3(x, fs, 0.1, 1)
    x += k*shift_v3(x, fs, 0.2, 2)
    x /= max(abs(x.max()), abs(x.min()))
    write(f'output_test.wav', fs, x)
    #write(f'test.wav', fs, shift_v3(x, fs, 0.2, 2))


if __name__ == '__main__':
    main()
