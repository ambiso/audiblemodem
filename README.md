# Audible Modem

Trying to create a modem that transmits information via sound.

## Tech

The information is prefixed with a string (binary data) known to both 
receiver and sender.
The receiver can search for this pattern in the recorded audio stream.
This is achieved by convolving the recorded audio signal with the expected signal.
The position of maximum amplitude in the convolution yields the start of the transmission.

The information is encoded using Direct-Sequence Spread Spectrum (DSSS).
The randomness is derived from SHAKE256, an eXtendible Output Function (XOF).
This essentially yields an insecure (read: unauthenticated) and inefficient (lots of randomness has to be generated for each bit) cipher.
However, it also yields some interesting properties (that likely have been studied before): if 
the amplitude of the signal is low enough an adversary could likely not distinguish the signal from random noise (but this likely also depends on how powerful the adversary is; i.e. how many antennas and locations they have access to).
