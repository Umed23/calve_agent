import pyaudio
import numpy as np

# Audio constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def create_audio_stream(p):
    """Creates a PyAudio input stream."""
    return p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

def is_silence(data, threshold=500):
    """Simple energy-based silence detection (backup to VAD)."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() < threshold
