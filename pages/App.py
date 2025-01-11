import pyaudio
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


SAMPLE_RATE = 16000
CHUNK_SIZE = 16000
MAX_AMPL = 8000
MAX_FREQ = 4000


def plot_spectrogram(data, placeholder):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.specgram(data, Fs=SAMPLE_RATE)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(0, MAX_FREQ)
    
    placeholder.pyplot(fig)
    plt.close(fig)


def plot_amplitude(data, placeholder):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(np.abs(data))
    ax.set_title("Amplitude Plot")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, MAX_AMPL)
    ax.grid()

    placeholder.pyplot(fig)
    plt.close(fig)


audio_interface = pyaudio.PyAudio()

stream = audio_interface.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

placeholder_spectrogram = st.empty()
placeholder_amplitude = st.empty()

while stream.is_active():
    data = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)

    plot_spectrogram(data, placeholder_spectrogram)
    plot_amplitude(data, placeholder_amplitude)

stream.stop_stream()
stream.close()
audio_interface.terminate()
    