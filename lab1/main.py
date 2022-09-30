import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from waveformenum import WaveformEnum
from numpy import pi
from key_number import KeyNumber
import matplotlib.pyplot as plt



def note_to_frequency(note):
    return 2**((note.value-49)/12)*440


def get_harmonic_coefficient():
    k = np.zeros(20000)
    for i in range(20000):
        if i < 100:
            k[i] = 1
        elif i < 200:
            k[i] = 1
        elif i < 350:
            k[i] = 1
        elif i < 400:
            k[i] = 0.9
        elif i < 450:
            k[i] = 0.8
        elif i < 500:
            k[i] = 0.9
        elif i < 550:
            k[i] = 0.8
        elif i < 600:
            k[i] = 0.7
        elif i < 650:
            k[i] = 0.7
        elif i < 700:
            k[i] = 0.8
        elif i < 750:
            k[i] = 0.9
        elif i < 800:
            k[i] = 1
        elif i < 850:
            k[i] = 0.8
        elif i < 900:
            k[i] = 0.6
        elif i < 950:
            k[i] = 0.3
        elif i < 1000:
            k[i] = 0.2
        elif i < 1100:
            k[i] = 0.1
        elif i < 1200:
            k[i] = 0.2
        elif i < 1300:
            k[i] = 0.3
        elif i < 1400:
            k[i] = 0.5
        elif i < 1500:
            k[i] = 0.6
        elif i < 1600:
            k[i] = 0.3
        elif i < 1700:
            k[i] = 0.4
        elif i < 1800:
            k[i] = 0.3
        elif i < 1900:
            k[i] = 0.2
        else:
            k[i] = 0.05

    return k


def tone(f, t, waveform=WaveformEnum.harmonic, fs=44100):
    time = np.linspace(0, 1, fs)
    omega = 2 * pi * f
    x = np.array([])
    if waveform == WaveformEnum.harmonic:
        x = np.sin(omega * time)
    elif waveform == WaveformEnum.meander:
        x = signal.square(omega * time)
    elif waveform == WaveformEnum.triangle:
        x = signal.sawtooth(omega * time, 0.5)
    elif waveform == WaveformEnum.saw:
        x = signal.sawtooth(omega * time)

    integer = int(t)
    if integer == 0:
        result = x[:round(fs*t)]
        return result
    result = x
    for i in range(integer-1):
        result = np.concatenate((result, x))
    if t-integer == 0:
        return result
    result = np.concatenate((result, x[:round(fs*(t-integer))]))
    return result


def attenuation(fs, t, db):
    times = 10 ** (db / 20)
    a = times ** (1 / (t * fs))
    result = np.zeros(round(t * fs))
    k = 1
    for i in range(round(t * fs)):
        result[i] = k
        k *= a
    return result


def musical_tone(f, t, waveform=WaveformEnum.harmonic, fs=44100, db=-100, db_attack=-60, k_time_attack=0.4,
                 harmonic_coefficient=np.ones(20000)):
    main_tone = tone(f, t, waveform, fs)

    f_harmonic = 2 * f
    while f_harmonic < 14000:
        harmonic = tone(f_harmonic, t, waveform, fs)
        main_tone += harmonic*harmonic_coefficient[round(f_harmonic)]
        f_harmonic += f

    if db != 0:
        main_tone *= attenuation(fs, t, db)

    attack_time = k_time_attack * t
    attack_attenuation = np.flip(attenuation(fs, attack_time, db_attack))
    main_tone[:round(attack_time * fs)] *= attack_attenuation

    maximum = max(main_tone.max(), abs(main_tone.min()))
    main_tone /= maximum
    return main_tone


def sum_note(lhs, rhs):
    result = lhs + rhs
    maximum = max(result.max(), abs(result.min()))
    result /= maximum
    return result


def main():

    form = WaveformEnum.harmonic
    fs = 44100

    c604 = musical_tone(note_to_frequency(KeyNumber.C6), 0.4, form, harmonic_coefficient=get_harmonic_coefficient())
    d606 = musical_tone(note_to_frequency(KeyNumber.D6), 0.6, form, harmonic_coefficient=get_harmonic_coefficient())
    a506 = musical_tone(note_to_frequency(KeyNumber.A5), 0.6, form, harmonic_coefficient=get_harmonic_coefficient())
    c606 = musical_tone(note_to_frequency(KeyNumber.C6), 0.6, form, harmonic_coefficient=get_harmonic_coefficient())
    b504 = musical_tone(note_to_frequency(KeyNumber.B5), 0.4, form, harmonic_coefficient=get_harmonic_coefficient())
    g504 = musical_tone(note_to_frequency(KeyNumber.G5), 0.4, form, harmonic_coefficient=get_harmonic_coefficient())
    g506 = musical_tone(note_to_frequency(KeyNumber.G5), 0.6, form, harmonic_coefficient=get_harmonic_coefficient())
    f504 = musical_tone(note_to_frequency(KeyNumber.F5), 0.4, form, harmonic_coefficient=get_harmonic_coefficient())
    f506 = musical_tone(note_to_frequency(KeyNumber.F5), 0.6, form, harmonic_coefficient=get_harmonic_coefficient())
    e51 = musical_tone(note_to_frequency(KeyNumber.E5), 1, form, db=0, harmonic_coefficient=get_harmonic_coefficient())

    empty = np.zeros(round(fs * 0.15))

    x = np.concatenate((c604, d606, empty, c606, empty, b504, a506, empty,  d606,
                        empty, c606, empty, b504, a506, empty, g504, g506, empty, f504, f506, empty, g504, e51))

    write(f'Unravel.wav', fs, x)

    b404 = musical_tone(note_to_frequency(KeyNumber.B4), 0.35, form, harmonic_coefficient=get_harmonic_coefficient())
    c504 = musical_tone(note_to_frequency(KeyNumber.C5), 0.35, form, harmonic_coefficient=get_harmonic_coefficient())
    g4b = musical_tone(note_to_frequency(KeyNumber.G4), 0.35, form, harmonic_coefficient=get_harmonic_coefficient())
    e412 = musical_tone(note_to_frequency(KeyNumber.E4), 0.8, form, harmonic_coefficient=get_harmonic_coefficient())
    e504 = musical_tone(note_to_frequency(KeyNumber.E5), 0.35, form, harmonic_coefficient=get_harmonic_coefficient())
    d504 = musical_tone(note_to_frequency(KeyNumber.D5), 0.35, form, harmonic_coefficient=get_harmonic_coefficient())
    b412 = musical_tone(note_to_frequency(KeyNumber.B4), 0.8, form, harmonic_coefficient=get_harmonic_coefficient())

    x = np.concatenate((b404, c504, b404, g4b, b404, c504, b404, g4b, e412, empty, empty))
    x = np.concatenate((x, x))
    x = np.concatenate((x, e504, d504, c504, empty, e504, d504, c504, empty, e504, e504, b412, empty, empty,
                        e504, d504, c504, empty, d504, c504, empty, e504, e504, b412, empty, empty))
    x = np.concatenate((x, x))

    write(f'Heathens.wav', fs, x)

    b3b = musical_tone(note_to_frequency(KeyNumber.B3), 0.5, form, db=-10,
                       harmonic_coefficient=get_harmonic_coefficient())
    fd4s = musical_tone(note_to_frequency(KeyNumber.Fd4), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    fd4b = musical_tone(note_to_frequency(KeyNumber.Fd4), 0.5, form, db=-10,
                        harmonic_coefficient=get_harmonic_coefficient())
    b3s = musical_tone(note_to_frequency(KeyNumber.B3), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    g4b = musical_tone(note_to_frequency(KeyNumber.G4), 0.5, form, db=-10,
                       harmonic_coefficient=get_harmonic_coefficient())
    e4s = musical_tone(note_to_frequency(KeyNumber.E4), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    g4s = musical_tone(note_to_frequency(KeyNumber.G4), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    d4s = musical_tone(note_to_frequency(KeyNumber.D4), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    cd4s = musical_tone(note_to_frequency(KeyNumber.Cd4), 0.27, form, harmonic_coefficient=get_harmonic_coefficient())
    d4b = musical_tone(note_to_frequency(KeyNumber.D4), 0.5, form, db=-10,
                       harmonic_coefficient=get_harmonic_coefficient())
    b3l = musical_tone(note_to_frequency(KeyNumber.B3), 0.5, form, db=-3,
                       harmonic_coefficient=get_harmonic_coefficient())

    x = np.concatenate((b3b, fd4s, b3s, g4b, fd4s, e4s, fd4b, e4s, fd4s, g4s, g4s, fd4s, e4s,
                        b3b, fd4s, b3s, g4b, fd4s, e4s, d4b, e4s, d4s, cd4s, cd4s, d4s, cd4s, b3l))

    write(f'Shut your mouth.wav', fs, x)


if __name__ == '__main__':
    main()
