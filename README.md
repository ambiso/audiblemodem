# Audible Modem

Trying to create a modem that transmits information via sound.

## Tech

The information is prefixed with a string (binary data) known to both 
receiver and sender.
The receiver can search for this pattern in the recorded audio stream.
This is achieved by convolving the recorded audio signal with the expected signal.
The position of maximum amplitude in the convolution yields the start of the transmission.

The information is encoded using Direct-Sequence Spread Spectrum (DSSS).
DSSS encodes information by taking a low frequency information signal, and multiplying it with a high frequency (pseudo-)random signal.
Both signals are binary signals encoded as $-1^b$ where $b$ is the bit. This results in encoding $0$ as $1$ and $1$ as $-1$.

The randomness is derived using SHAKE256, an eXtendable Output Function (XOF).
An XOF is essentially a hash function with an arbitrary output length.

This essentially yields an insecure (read: unauthenticated) and inefficient (lots of randomness has to be generated for each bit) cipher.
However, it also yields some interesting properties (that have very likely been studied before): if 
the amplitude of the signal is low enough an adversary could likely not distinguish the signal from random noise, unless they have access to the secret key that the XOF is initialized with (but this likely also depends on how powerful the adversary is; i.e. how many antennas and locations they have access to).
