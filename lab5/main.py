from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window


def main():
    fs, x = wavfile.read('lab5_4.wav')
    x = x[:, 0]
    x = x.astype(np.float32)
    x /= max(abs(x.min()), abs(x.max()))

    time = 0.1
    name = 'triangle'
    size = round(time * fs)
    window = get_window(name, size)
    f, t, zxx = stft(x, fs, window=window, nperseg=size)

    plt.figure('spectorgram')
    plt.pcolormesh(t, f, np.abs(zxx)**2)
    plt.xlabel('sec')
    plt.ylabel('Hz')
    plt.show()


if __name__ == '__main__':
    main()

