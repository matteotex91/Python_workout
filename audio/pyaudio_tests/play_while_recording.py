import pyaudio
import numpy as np
from time import sleep

CHUNK = 1024
CHUNKS = 50
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = np.int16(1000)
FREQUENCY = 440  # Hz
FORMAT = pyaudio.paInt16
audio = pyaudio.PyAudio()

chunk_count = CHUNKS

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
)


def acquire():
    global chunk_count, stream

    def callback(in_data, frame_count, time_info, flag):
        global chunk_count, stream, AMP
        audio_data = np.frombuffer(buffer=in_data, count=frame_count, dtype=np.int16)
        audio_data = AMP / np.max(audio_data)
        stream.write(audio_data, len(audio_data))
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
    stream.close()


if __name__ == "__main__":
    acquire()
