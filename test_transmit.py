from playback import *
import os

def test_bin2byte():
    for i in range(256):
        assert bin2byte(byte2bin(i)) == i


def test_bytes2bin():
    b = os.urandom(100)
    assert bin2bytes(bytes2bin(b)) == b