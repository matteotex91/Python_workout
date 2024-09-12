import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from time import sleep

"""This script acquires 100 callbacks performing immediately the fft of the chunk
Then the timetrace of the rfft is plotted as colormesh only in the audible range
"""


fft_vectors = []
freq_vectors = []
time_axis = []
count = 100

if __name__ == "__main__":
    # AUDIO INPUT
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, flag):
        global fft_vectors, freq_vectors, count
        audio_data = np.frombuffer(buffer=in_data, count=frame_count, dtype=np.int16)
        fft = np.fft.rfft(audio_data)
        fft_vectors.append(fft)
        N = len(fft)
        n = np.arange(N)
        T = N / RATE
        freq = n / T
        freq_vectors.append(freq)
        time_axis.append(time_info["current_time"])
        count -= 1
        return (audio_data, pyaudio.paContinue if count > 0 else pyaudio.paAbort)

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )
    sleep(2)
    while count > 0:
        sleep(0.01)

    time_axis = np.array(time_axis) - min(time_axis)
    freq = freq_vectors[0]
    min_freq_ind = np.argmin(np.abs(freq - 20))
    max_freq_ind = np.argmin(np.abs(freq - 20000))
    fft_matrix = np.abs(np.array(fft_vectors))
    plt.pcolormesh(
        freq[min_freq_ind:max_freq_ind],
        time_axis,
        fft_matrix[:, min_freq_ind:max_freq_ind],
    )
    plt.xlabel("frequency [Hz]")
    plt.ylabel("time [s]")
    plt.show()

    print("stop")
