#!/usr/bin/env python3
# -*- mode: python; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-

## playbacktest.py
##
## This is an example of a simple sound playback script.
##
## The script opens an ALSA pcm for sound playback. Set
## various attributes of the device. It then reads data
## from stdin and writes it to the device.
##
## To test it out do the following:
## python recordtest.py out.raw # talk to the microphone
## python playbacktest.py out.raw


from __future__ import print_function

import sys
import time
import getopt
from typing import List
import alsaaudio
from math import sin, pi, ceil
import numpy as np
from Crypto.Hash import SHAKE256
from scipy import signal
import matplotlib.pyplot as plt
import os
import numba

def normalize(waveform):
    lo, hi = minmax(waveform)
    waveform = ((waveform - lo)/(hi-lo)) * 2
    mu = np.mean(waveform)
    waveform -= mu
    waveform /= max(abs(-mu), abs(2-mu))
    return waveform

def byte2bin(byte: int) -> List[int]:
    assert byte < 256
    l = []
    while byte > 0:
        l.append(byte & 1)
        byte >>= 1
    while len(l) < 8:
        l.append(0)
    return l

def bytes2bin(b: bytes) -> List[int]:
    l = []
    for x in b:
        l.extend(byte2bin(x))
    return l

def bin2byte(b: List[int]) -> int:
    n = 0
    for x in reversed(b):
        n *= 2
        n += x
    return n

def bin2bytes(b: List[int]) -> bytes:
    bs = []
    for i in range(0, len(b), 8):
        bs.append(bin2byte(b[i:i+8]))
    return bytes(bs)


def bandpass(fs, order=5):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [1000/nyq, 20000/nyq], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(data, fs, order=5):
    b, a = bandpass(fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

@numba.njit
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)

def decode(data, spread_per_bit, spread_repeat, rate, fig=None, axs=None):
    shake = SHAKE256.new()
    shake.update(b'Your Secret Key Here')

    information = b'KNOWN'

    decoded_data = []
    for i in range(0, len(data), 2):
        d = int.from_bytes(data[i:i+2], 'little', signed=True)
        d /= pow(2, 15) - 1
        decoded_data.append(d)
    
    print(f"Amplitude: {np.mean(np.abs(decoded_data))}")
    # decoded_data = normalize(bandpass_filter(decoded_data, rate))
    decoded_data = normalize(np.array(decoded_data))

    dsss = bytes2bin(shake.read(ceil(len(decoded_data)/8)))

    i = 0
    waveform = []
    for x in information:
        for _ in range(spread_per_bit):
            v = pow(-1, x) * pow(-1, dsss[i])
            for _ in range(spread_repeat):
                waveform.append(v)
            i += 1
        
    waveform = normalize(bandpass_filter(waveform, rate))
    # plt.show()
    print(f"{len(waveform)=} {len(decoded_data)=}")
    # plot_fft(decoded_data)
    sync = np.convolve(decoded_data, np.flip(waveform), 'valid')
    signal_start = np.argmax(sync)
    # print(f"{signal_start=}")
    if axs is None:
        fig, axs = plt.subplots(3)
    decoded_data = normalize(decoded_data[signal_start:])
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    axs[0].plot(sync[signal_start:signal_start+200] / len(waveform), label="Sync")
    axs[0].plot(waveform[:1000], label="Known")
    axs[0].plot(decoded_data[:1000], label="Decoded")
    axs[0].legend()

    despreaded = []
    for i in range(min(spread_repeat*len(dsss), len(decoded_data))):
        s = dsss[i // spread_repeat]
        d = decoded_data[i]
        despreaded.append(pow(-1, s) * d)
    step = spread_per_bit * spread_repeat
    hstep = step//2
    d = normalize(moving_average(despreaded, hstep))
    # axs[1].scatter(list(range(200)), d[:200])
    # axs[1].vlines(list(range(hstep, 200, step)), -1, 1)
    axs[1].scatter(list(range(1000)), despreaded[:1000])
    # plt.scatter(list(range(1000)), d[:1000])
    plot_fft(decoded_data, axs[2])
    plt.show(block=False)
    plt.pause(0.08)
    bits = []
    for i in range(hstep, len(d), step):
        bits.append(d[i] < 0)
    # print(bits[:1000//step])
    print(bin2bytes(bits))

def plot_fft(x, ax=plt, fs=1/44100):
    ps = np.abs(np.fft.fft(x))**2 / len(x)

    freqs = np.fft.fftfreq(len(x), fs)
    idx = np.argsort(freqs)

    ax.plot(freqs[idx], ps[idx])


if __name__ == '__main__':

    device = 'pulse'

    # Open the device in playback mode in Mono, 44100 Hz, 16 bit little endian frames
    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.

    rate = 44100
    periodsize = rate
    out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, channels=1, rate=rate, format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=periodsize, device=device)
    t = 0
    spread_per_bit = 128
    spread_repeat = 1

    binary_data_buffer = []
    while True:
        while len(binary_data_buffer) < 2*periodsize:
            # waveform = [ (sin(440*2*pi*(t+dt))+1) * .5 * amplitude for dt in np.arange(0, delta, 1/rate)] # between 0 and 1
            shake = SHAKE256.new()
            shake.update(b'Your Secret Key Here')

            information = bytes2bin(b"KNOWNPREFIX Hello World! ")
            # print(information[:10])
            dsss = bytes2bin(shake.read(ceil(len(information)/8) * spread_per_bit))
            assert len(dsss) >= len(information) * spread_per_bit
            # print(len(dsss))
            i = 0
            waveform = []
            for x in information:
                for _ in range(spread_per_bit):
                    v = pow(-1, x) * pow(-1, dsss[i])
                    for _ in range(spread_repeat):
                        waveform.append(v)
                    i += 1
            # plot_fft(waveform)
            waveform = bandpass_filter(waveform, rate)
            waveform = normalize(waveform)
            # plot_fft(waveform)
            # plt.hlines(np.mean(waveform), -10000, 10000)
            # plt.show()
            # plt.plot(waveform)
            # plt.show()
            
            # plt.plot(waveform[:40])
            # plt.show()
            for v in waveform:
                x = int((pow(2, 15)-1) * v)
                binary_data_buffer.extend(x.to_bytes(2, 'little', signed=True))
        bdata = bytes(binary_data_buffer[:2*periodsize])
        binary_data_buffer = binary_data_buffer[2*periodsize:]
        out.write(bdata)

        # decode(os.urandom(2*25000) + bdata, spread_per_bit, spread_repeat, rate)
        # decode(bdata, spread_per_bit, spread_repeat, rate)