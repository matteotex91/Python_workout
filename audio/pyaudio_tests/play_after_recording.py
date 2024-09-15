import pyaudio
import numpy as np
from time import sleep

CHUNK = 1024
CHUNKS = 50
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 1000
FREQUENCY = 440  # Hz
FORMAT = pyaudio.paInt16
audio = pyaudio.PyAudio()

acquired_wave = np.array([], dtype=np.int16)
chunk_count = CHUNKS


def acquire():
    global chunk_count, acquired_wave

    def callback(in_data, frame_count, time_info, flag):
        global chunk_count, acquired_wave
        audio_data = np.frombuffer(buffer=in_data, count=frame_count, dtype=np.int16)
        acquired_wave = np.append(acquired_wave, audio_data)
        chunk_count -= 1
        return (
            in_data,
            pyaudio.paContinue if chunk_count >= 0 else pyaudio.paAbort,
        )

    audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
    )
    while chunk_count > 0:
        sleep(0.1)
    stream.write(acquired_wave, acquired_wave.shape[0])
    stream.close()


if __name__ == "__main__":
    acquire()
