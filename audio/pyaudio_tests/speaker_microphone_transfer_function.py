import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from audio.datasets.data_manager import load_transferfunction, save_transferfunction


CHUNK = 1024
CHUNKS = 10
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 1000
FREQUENCY = 440  # Hz
FORMAT = pyaudio.paInt16
audio = pyaudio.PyAudio()

fft_sum_0 = None
fft_sum_1 = None
fft_sum_2 = None
fft_fr_arr = None
running = True


freq_index = 0
acquiring = False
fft_amp = []
fft_std = []


def acquire():
    global fr_arr, fft_amp, fft_std, fft_fr_arr, fft_sum_0, fft_sum_1, fft_sum_2, acquiring, running

    def callback(in_data, frame_count, time_info, flag):
        global acquiring, fft_sum_0, fft_sum_1, fft_sum_2, running, fft_fr_arr, running
        if acquiring:
            audio_data = np.frombuffer(
                buffer=in_data, count=frame_count, dtype=np.int16
            )
            fft = np.fft.rfft(audio_data)
            freq = np.fft.fftfreq(audio_data.size, 1 / RATE)
            min_freq_ind = np.argmin(np.abs(freq - 20))
            max_freq_ind = np.argmin(np.abs(freq - 20000))
            if fft_fr_arr is None:
                fft_fr_arr = freq[min_freq_ind:max_freq_ind]
            clipped_fft = np.abs(fft[min_freq_ind:max_freq_ind])
            if fft_sum_0 is None:
                fft_sum_0 = np.ones_like(clipped_fft)
                fft_sum_1 = clipped_fft
                fft_sum_2 = np.power(clipped_fft, 2)
            else:
                fft_sum_0 += np.ones_like(clipped_fft)
                fft_sum_1 += clipped_fft
                fft_sum_2 += np.power(clipped_fft, 2)

        return (
            in_data,
            pyaudio.paContinue if running else pyaudio.paAbort,
        )

    audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )
    samples = CHUNK * CHUNKS
    final_time = CHUNKS * CHUNK / RATE
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
    )
    for index, freq in enumerate(fr_arr):

        freq_index = index
        print(freq_index)
        omega = 2 * np.pi * freq
        wave = np.int16(AMP * (np.sin(omega * np.linspace(0, final_time, samples))))
        fft_sum_0 = None
        fft_sum_1 = None
        fft_sum_2 = None
        acquiring = True
        stream.write(wave, samples)
        acquiring = False
        fft_amp.append(fft_sum_1 / fft_sum_0)
        fft_std.append(
            np.sqrt(fft_sum_2 / fft_sum_0 - np.power(fft_sum_1 / fft_sum_0, 2))
        )
    stream.close()
    running = False


if __name__ == "__main__":
    fr_arr = np.linspace(20, 20000, 200)
    flag_acquire = True
    load_run_id = 4

    if flag_acquire:
        acquire()
        save_transferfunction(fr_arr, fft_fr_arr, fft_amp, fft_std)
    else:
        fr_arr, fft_fr_arr, fft_amp, fft_std = load_transferfunction(load_run_id)

    plt.pcolormesh(np.array(fft_amp))
    plt.show()

    peak_intensities = np.array(
        [row[np.argmax(np.abs(fft_fr_arr - f0))] for row, f0 in zip(fft_amp, fr_arr)]
    )
    peak_sigmas = np.array(
        [row[np.argmax(np.abs(fft_fr_arr - f0))] for row, f0 in zip(fft_std, fr_arr)]
    )
    plt.fill_between(
        fr_arr,
        peak_intensities - peak_sigmas,
        peak_intensities + peak_sigmas,
        color="orange",
    )
    plt.plot(fr_arr, peak_intensities)

    plt.show()
