import torch
import pyaudio
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from io import BytesIO
from scipy import signal
from torchvision import transforms


CLASSES = ["go", "stop", "bed", "other"]
SAMPLE_RATE = 16000
CHUNK_SIZE = 16000
MAX_AMPL = 8000
MAX_FREQ = 4000


def plot_spectrogram(data, placeholder):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.specgram(data, Fs=SAMPLE_RATE)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(0, MAX_FREQ)

    placeholder.pyplot(fig)
    plt.close(fig)


def plot_amplitude(data, placeholder):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(np.abs(data))
    ax.set_title("Amplitude Plot")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, MAX_AMPL)
    ax.grid()

    placeholder.pyplot(fig)
    plt.close(fig)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.batch1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act4 = nn.GELU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc5 = nn.Linear(2048, 1024)
        self.act5 = nn.GELU()

        self.fc6 = nn.Linear(1024, 512)
        self.act6 = nn.GELU()

        self.fc7 = nn.Linear(512, 4)
        self.act7 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)

        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        x = self.act3(self.conv3(x))
        x = self.pool3(x)

        x = self.act4(self.conv4(x))
        x = self.pool4(x)

        x = self.flat(x)

        x = self.act5(self.fc5(x))
        x = self.act6(self.fc6(x))
        x = self.act7(self.fc7(x))

        return x


model = ConvNet()
weights = torch.load("model", map_location=torch.device("cpu"))
model.load_state_dict(weights)

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

audio_interface = pyaudio.PyAudio()

stream = audio_interface.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

placeholder_label = st.empty()
placeholder_spectrogram = st.empty()
placeholder_amplitude = st.empty()

while stream.is_active():
    data = np.frombuffer(
        stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16
    )

    frequencies, times, Sxx = signal.spectrogram(data, fs=SAMPLE_RATE)
    Sxx = np.clip(Sxx, 1e-10, None)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=64)
    ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading="auto")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    data_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(data_tensor)
        predictions = torch.argmax(outputs, dim=1)
        print(outputs)
        print(predictions)

    placeholder_label.write(CLASSES[predictions])
    plot_spectrogram(data, placeholder_spectrogram)
    plot_amplitude(data, placeholder_amplitude)

stream.stop_stream()
stream.close()
audio_interface.terminate()
