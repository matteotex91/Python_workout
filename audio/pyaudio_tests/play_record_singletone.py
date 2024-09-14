import pyqtgraph as pg
import pyaudio
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSignal, QEvent, QThread, QObject, QMutex
from PyQt5.QtGui import QMouseEvent
import sys
import numpy as np
from time import sleep

CHUNK = 1024
CHUNKS = 100
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
AMP = 2000
FREQUENCY = 440  # Hz


class ListenerTask(QObject):
    fft_data_signal = pyqtSignal(object)

    def __init__(self):
        QObject.__init__(self)
        self.running = True
        self.runningLock = QMutex()

    def acquire(self):
        FORMAT = pyaudio.paInt16
        audio = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, flag):
            audio_data = np.frombuffer(
                buffer=in_data, count=frame_count, dtype=np.int16
            )
            fft = np.fft.rfft(audio_data)
            N = len(fft)
            n = np.arange(N)
            T = N / RATE
            freq = n / T
            min_freq_ind = np.argmin(np.abs(freq - 20))
            max_freq_ind = np.argmin(np.abs(freq - 20000))
            self.fft_data_signal.emit(
                [
                    freq[min_freq_ind:max_freq_ind],
                    np.abs(fft[min_freq_ind:max_freq_ind]),
                ]
            )
            return (
                audio_data,
                pyaudio.paContinue if self.running > 0 else pyaudio.paAbort,
            )

        stream_in = audio.open(
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

        samples = CHUNK * CHUNKS
        final_time = CHUNKS * CHUNK / RATE
        omega = 2 * np.pi * FREQUENCY
        self.wave = np.int16(
            AMP * (np.sin(omega * np.linspace(0, final_time, samples)))
        )
        stream.write(self.wave, samples)
        stream.close()

        while self.running:
            sleep(0.1)

    def stop(self):
        self.runningLock.lock()
        self.running = False
        self.runningLock.unlock()


class AudioListenerThread(QThread):
    def __init__(self, callback_function):
        QThread.__init__(self)
        self.task = ListenerTask()
        self.task.moveToThread(self)
        self.started.connect(self.task.acquire)
        self.task.fft_data_signal.connect(callback_function)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.acquire_thread = AudioListenerThread(self.replot_fft)

    def replot_fft(self, fft_signal):
        self.plot_graph.clear()

        self.plot_graph.plot(fft_signal[0], fft_signal[1] / max(fft_signal[1]))

    def closeEvent(self, evt):
        self.acquire_thread.task.stop()


app = QApplication(sys.argv)
main = MainWindow()
main.show()
main.acquire_thread.start()
app.exec()
