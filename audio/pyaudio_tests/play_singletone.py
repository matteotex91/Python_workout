import numpy as np
import pyaudio

CHUNK = 1024
CHUNKS = 100
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 2000
FREQUENCY = 440  # Hz


class AudioFile:
    def __init__(self):
        """Init audio stream"""
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )
        samples = CHUNK * CHUNKS
        final_time = CHUNKS * CHUNK / RATE
        omega = 2 * np.pi * FREQUENCY
        self.wave = np.int16(
            AMP * (np.sin(omega * np.linspace(0, final_time, samples)))
        )
        self.stream.write(self.wave, samples)
        self.stream.close()


a = AudioFile()
