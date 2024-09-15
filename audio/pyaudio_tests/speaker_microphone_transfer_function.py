import pyaudio
import numpy as np
import matplotlib.pyplot as plt


CHUNK = 1024
CHUNKS = 10
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 1000
FREQUENCY = 440  # Hz

fft_sum_0 = None
fft_sum_1 = None
fft_sum_2 = None
fft_freq_vector = None
running = True

frequencies = np.linspace(700, 1000, 200)
freq_index = 0
acquiring = False
fft_responses = []
fft_sigmas = [fft_responses]

if __name__ == "__main__":
    FORMAT = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, flag):
        global acquiring, fft_sum_0, fft_sum_1, fft_sum_2, running, fft_freq_vector
        if acquiring:
            audio_data = np.frombuffer(
                buffer=in_data, count=frame_count, dtype=np.int16
            )
            fft = np.fft.rfft(audio_data)
            freq = np.fft.fftfreq(audio_data.size, 1 / RATE)
            min_freq_ind = np.argmin(np.abs(freq - 20))
            max_freq_ind = np.argmin(np.abs(freq - 20000))
            if fft_freq_vector is None:
                fft_freq_vector = freq[min_freq_ind:max_freq_ind]
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

    stream_in = audio.open(
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
    for index, freq in enumerate(frequencies):

        freq_index = index
        omega = 2 * np.pi * freq
        wave = np.int16(AMP * (np.sin(omega * np.linspace(0, final_time, samples))))
        fft_sum_0 = None
        fft_sum_1 = None
        fft_sum_2 = None
        acquiring = True
        stream.write(wave, samples)
        acquiring = False
        fft_responses.append(fft_sum_1 / fft_sum_0)
        fft_sigmas.append(
            np.sqrt(fft_sum_2 / fft_sum_0 - np.power(fft_sum_1 / fft_sum_0, 2))
        )
    stream.close()
    running = False

    plt.pcolormesh(np.array(fft_responses))
    plt.show()

    peak_intensities = [
        row[np.argmax(np.abs(fft_freq_vector - f0))]
        for row, f0 in zip(fft_responses, frequencies)
    ]
    plt.plot(frequencies, peak_intensities)
    plt.show()

    print("stop here")
