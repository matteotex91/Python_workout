import pyaudio
import numpy as np
from time import sleep

CHUNK = 1024
CHUNKS = 200
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = np.int16(100)
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
        norm_audio_data = audio_data
        stream.write(norm_audio_data, len(norm_audio_data))
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
    sleep(5)
    stream.close()


if __name__ == "__main__":
    acquire()
