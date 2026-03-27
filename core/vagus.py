import webrtcvad
import pyaudio
from utils.audio import RATE, CHUNK

class VagusNerve:
    """
    Handles interruption detection (VAD).
    'Vagus' because it controls the heart (core loop) and voice.
    """
    def __init__(self, sensitivity=3):
        self.vad = webrtcvad.Vad(sensitivity)
        self.interrupted = False

    def is_speech(self, audio_frame):
        """Returns True if the frame contains speech."""
        try:
            return self.vad.is_speech(audio_frame, RATE)
        except Exception:
            return False

    def check_for_interruption(self, stream):
        """
        Reads a chunk from the stream and checks for speech.
        Returns True if speech is detected.
        """
        if stream.is_active():
            data = stream.read(CHUNK, exception_on_overflow=False)
            if self.is_speech(data):
                self.interrupted = True
                return True
        return False

    def reset(self):
        self.interrupted = False
