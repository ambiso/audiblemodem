#!/usr/bin/env python3
# -*- mode: python; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-

## recordtest.py
##
## This is an example of a simple sound capture script.
##
## The script opens an ALSA pcm device for sound capture, sets
## various attributes of the capture, and reads in a loop,
## writing the data to standard out.
##
## To test it out do the following:
## python recordtest.py out.raw # talk to the microphone
## aplay -r 8000 -f S16_LE -c 1 out.raw

#!/usr/bin/env python

from __future__ import print_function

import time
import alsaaudio
import matplotlib.pyplot as plt

from playback import decode, plot_fft

if __name__ == '__main__':

    device = 'pulse'

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, 
        channels=1, rate=44100, format=alsaaudio.PCM_FORMAT_S16_LE, 
        periodsize=160, device=device)

    rate = 44100

    fig, axs = plt.subplots(3)
    while True:
        buf = []
        while len(buf) < 4*rate:
            l, data = inp.read()
            buf.extend(data)
            assert l is not None or len(data) == 0

        decode(buf, 128, 1, rate, fig, axs)
