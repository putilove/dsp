from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.signal import stft
from scipy.signal import istft
import numpy as np
import matplotlib.pyplot as plt


def robotization(x, fs, time, name_window, overlap_time):
    size = round(time * fs)
    noverlap = round(overlap_time * size)
    window = get_window(name_window, size)
    f, t, zxx = stft(x, fs, window=window, nperseg=size, noverlap=noverlap)
    res = np.abs(zxx)
    _, res = istft(res, fs, window=window, nperseg=size, noverlap=noverlap)
    return f, t, zxx, res


def main():
    fs, x = wavfile.read('input.wav')
    x = x.astype(np.float32)
    x /= max(abs(x.min()), abs(x.max()))

    time = 0.07
    name = 'triangle'
    overlap_time = 0.3

    f, t, zxx, res = robotization(x, fs, time, name, overlap_time)

    write(f'output.wav', fs, res)

    plt.figure('spectorgram')
    plt.pcolormesh(t, f, np.abs(zxx)**2)
    plt.xlabel('sec')
    plt.ylabel('Hz')
    plt.show()


if __name__ == '__main__':
    main()
