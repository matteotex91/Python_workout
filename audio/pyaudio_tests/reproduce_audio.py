import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from time import sleep
import wave

CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 1


class AudioFile:
    def __init__(self, file):
        """Init audio stream"""
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )
        chunks = 100
        samples = CHUNK * chunks
        final_time = chunks * CHUNK / RATE

        self.wave = AMP * np.sin(np.linspace(0, final_time, samples))
        self.stream.write()

    def play(self):
        """Play entire file"""
        data = self.wf.readframes(CHUNK)
        while data != b"":
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """Graceful shutdown"""
        self.stream.close()
        self.p.terminate()


# Usage example for pyaudio
a = AudioFile("1.wav")
a.play()
a.close()
