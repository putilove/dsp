import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


def create_signal(mes):
    res = np.zeros(1000)
    dot = np.ones(20)
    dash = np.ones(60)
    space_word = np.zeros(120)
    space_char = np.zeros(40)
    empty = np.zeros(20)

    for c in mes:
        if c == '.':
            res = np.concatenate((res, dot, empty))
        if c == '-':
            res = np.concatenate((res, dash, empty))
        if c == ' ':
            res = np.concatenate((res, space_char))
        if c == '_':
            res = np.concatenate((res, space_word))
    return res


def mse(lhs, rhs):
    return np.mean(np.array([(lhs[i]-rhs[i])**2 for i in range(len(lhs))]))


def main():
    mes = '.-_.--- --- ..- .-. -. . -.--_--- ..-._- .... --- ..- ... .- -. -.._' \
          '-- .. .-.. . ..._-... . --. .. -. ..._' \
          '.-- .. - ...._.-_... .. -. --. .-.. ._... - . .--.'

    clear_signal = create_signal(mes)

    plt.figure('Reference')
    plt.plot(clear_signal)

    first_res = np.load('res.npy')

    plt.figure('BandpassReconstructed')
    plt.plot(first_res)

    first_res = np.concatenate((first_res[10:], np.zeros(10)))
    print('6lab: ', mse(clear_signal, first_res))

    signal = np.load('4.npy')

    plt.figure('Input')
    plt.plot(signal)

    m = 20
    r_y = acf(signal[1000:], adjusted=True, nlags=m)
    d = np.var(signal[:1000])

    r_xy = r_y.copy()
    r_xy[0] -= d

    A = np.fromfunction(lambda i, j: r_y[np.abs(i.astype(int)-j.astype(int))], (len(r_y), len(r_y)), dtype=float)
    b = np.array([r_xy[i] for i in range(len(r_xy))])
    x = np.linalg.solve(A, b)

    plt.figure('x')
    plt.plot(x)

    result = sp.signal.convolve(signal, x)

    print('7lab: ', mse(clear_signal, result))

    plt.figure('SuboptimalReconstructed')
    plt.plot(result)
    plt.show()


if __name__ == '__main__':
    main()

