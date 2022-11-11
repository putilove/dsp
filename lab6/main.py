import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import signal


def main():
    x = np.load("4.npy")

    plt.figure('input')
    plt.plot(x)

    spectre = fft.fft(x)

    plt.figure('input spectre')
    plt.plot(np.abs(spectre[:round(len(spectre)/2)]))

    f = 241
    omega = f * np.pi/len(x)
    samples = np.pi / (2 * omega)
    print(omega)
    print(samples)
    m = np.ceil(samples)
    freq = 1 / m
    print(freq)

    h = signal.firls(m, [0, freq, freq, 1], [1, 1, 0, 0])
    h_padded = np.pad(h, (0, 9 * len(h)))
    spectre_h = np.abs(fft.fft(h_padded))
    spectre_h = spectre_h[:len(spectre_h) // 2]

    plt.figure('frequency response')
    plt.plot(spectre_h)

    result = signal.convolve(x, h)

    plt.figure("reconstructed")
    plt.plot(result)

    plt.show()


if __name__ == '__main__':
    main()

