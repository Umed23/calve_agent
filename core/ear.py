import pyaudio
import numpy as np
import webrtcvad
import queue
import threading
import collections
import torch
import soundfile as sf
import torchaudio
import logging

logger = logging.getLogger(__name__)

# Monkey-patch BEFORE transformers import
def safe_load(filepath, **kwargs):
    audio_np, sr = sf.read(filepath, dtype='float32')
    if audio_np.ndim > 1:
        audio = torch.from_numpy(audio_np).T
    else:
        audio = torch.from_numpy(audio_np).unsqueeze(0)
    return audio, sr

torchaudio.load = safe_load
torchaudio.info = lambda x: None

from transformers import pipeline

class Ear:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", language="hi"):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        model_id = "ARTPARK-IISc/whisper-small-vaani-hindi"
        logger.info(f"[Ear] Loading ARTPARK Vaani STT ({model_id}) on {self.device}...")
        
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=self.device,
            # Removed chunk_length_s=30 — unnecessary for short phrases, adds overhead
            torch_dtype=torch.float32,
        )
        
        self.vad = webrtcvad.Vad(3)
        self.audio_queue = queue.Queue()
        self.rate = 16000
        self.chunk_duration_ms = 30
        self.chunk_size = int(self.rate * self.chunk_duration_ms / 1000)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.listening = False
        self.capture_thread = None
        self.language = language

        # Pre-roll buffer: 500ms (was 1000ms — less audio to process)
        self.preroll_duration_ms = 500
        self.preroll_chunks = int(self.preroll_duration_ms / self.chunk_duration_ms)
        self.preroll_buffer = collections.deque(maxlen=self.preroll_chunks)

        # Result queue for non-blocking transcription
        self._result_queue = queue.Queue()

    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.capture_thread = threading.Thread(target=self._capture_audio)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("[Ear] Started listening to microphone.")

    def stop_listening(self):
        self.listening = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        logger.info("[Ear] Stopped listening.")

    def _capture_audio(self):
        while self.listening:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                logger.error(f"[Ear] Audio Capture Error: {e}", exc_info=True)
                break

    def _transcribe_async(self, audio_np):
        """Runs transcription in a background thread, puts result in _result_queue."""
        try:
            result = self.transcriber(
                {"sampling_rate": self.rate, "raw": audio_np},
                generate_kwargs={"language": "hindi", "task": "transcribe"}
            )
            text = result["text"].strip()
            if text and not self._is_hallucination(text):
                logger.info(f"[Ear] Fully Transcribed Speech: '{text}'")
                self._result_queue.put(text)
            else:
                if text:
                    logger.debug(f"[Ear] Ignored STT hallucination: '{text}'")
        except Exception as e:
            logger.error(f"[Ear] Transcription background error: {e}", exc_info=True)

    def listen(self):
        """Generator that yields transcribed text segments."""
        frames = []
        silence_duration = 0
        speech_detected = False
        transcription_thread = None

        # Tuned parameters
        SILENCE_THRESHOLD_MS = 800   # was 1500 — faster end-of-speech detection
        MIN_PHRASE_DURATION_MS = 300
        RMS_THRESHOLD = 300

        while self.listening:
            # Yield any completed transcriptions first (non-blocking)
            try:
                while True:
                    text = self._result_queue.get_nowait()
                    yield text
            except queue.Empty:
                pass

            # Get next audio chunk
            try:
                data = self.audio_queue.get(timeout=0.1)  # was 1 — much more responsive
            except queue.Empty:
                continue

            rms = self._calculate_rms(data)

            try:
                is_speech = self.vad.is_speech(data, self.rate) if rms > RMS_THRESHOLD else False
            except Exception:
                is_speech = False

            if is_speech:
                if not speech_detected:
                    logger.debug(f"[Ear] Speech activity detected (RMS: {int(rms)})...")
                    frames.extend(self.preroll_buffer)
                    self.preroll_buffer.clear()
                speech_detected = True
                silence_duration = 0
                frames.append(data)
            else:
                if speech_detected:
                    silence_duration += self.chunk_duration_ms
                    frames.append(data)

                    if silence_duration > SILENCE_THRESHOLD_MS:
                        total_duration_ms = len(frames) * self.chunk_duration_ms

                        if total_duration_ms < MIN_PHRASE_DURATION_MS:
                            pass # discard silently to reduce log spam
                            # logger.debug(f"[Ear] Discarding short noise ({total_duration_ms}ms)")
                        else:
                            logger.info(f"[Ear] Processing spoken phrase block ({len(frames)} frames)...")
                            audio_data = b''.join(frames)
                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                            # Fire transcription in background — Ear keeps listening immediately
                            transcription_thread = threading.Thread(
                                target=self._transcribe_async,
                                args=(audio_np,),
                                daemon=True
                            )
                            transcription_thread.start()

                        frames = []
                        speech_detected = False
                        silence_duration = 0
                else:
                    self.preroll_buffer.append(data)

    def _calculate_rms(self, data):
        shorts = np.frombuffer(data, dtype=np.int16)
        if len(shorts) == 0:
            return 0
        return np.sqrt(np.sum(shorts.astype(np.float32)**2) / len(shorts))

    def _is_hallucination(self, text):
        hallucinations = ["thanks for watching", "thank you", "subtitles by", "watching"]
        text_lower = text.lower()
        return any(h in text_lower for h in hallucinations) or len(text) < 2







# # MUST be first — before transformers, torchaudio, etc.
# import torch
# import numpy as np
# import soundfile as sf
# import torchaudio

# def safe_load(filepath, **kwargs):
#     audio_np, sr = sf.read(filepath, dtype='float32')
#     if audio_np.ndim > 1:
#         audio = torch.from_numpy(audio_np).T
#     else:
#         audio = torch.from_numpy(audio_np).unsqueeze(0)
#     return audio, sr

# torchaudio.load = safe_load
# torchaudio.info = lambda x: None


# import pyaudio
# import numpy as np
# import webrtcvad
# import queue
# import threading
# import collections
# import torch
# from transformers import pipeline


# class Ear:
#     """
#     Handles listening to microphone input and transcribing speech using Faster Whisper.
#     Uses a separate thread for audio capture and a Circular Buffer for pre-roll.
#     """
#     def __init__(self, model_size="base", device="cpu", compute_type="int8", language="hi"):
#         self.device = 0 if torch.cuda.is_available() else "cpu"
        
#         if model_size == "base": model_size = "small"
#         model_id = f"ARTPARK-IISc/whisper-{model_size}-vaani-hindi"
#         print(f"Loading ARTPARK Vaani STT variant ({model_id}) on {self.device}...")
        
#         self.transcriber = pipeline(
#             "automatic-speech-recognition",
#             model=model_id,
#             device=self.device,
#             chunk_length_s=30, # Allow pipeline chunking if audio gets long
#         )
        
#         self.vad = webrtcvad.Vad(3) # Aggressive VAD (0-3)
#         self.audio_queue = queue.Queue()
#         self.rate = 16000
#         self.chunk_duration_ms = 30 # WebRTC VAD requires 10, 20, or 30ms
#         self.chunk_size = int(self.rate * self.chunk_duration_ms / 1000)
#         self.p = pyaudio.PyAudio()
#         self.stream = None
#         self.listening = False
#         self.capture_thread = None
#         self.language = language 
        
#         # Pre-roll buffer: 1 second of audio
#         self.preroll_duration_ms = 1000
#         self.preroll_chunks = int(self.preroll_duration_ms / self.chunk_duration_ms)
#         self.preroll_buffer = collections.deque(maxlen=self.preroll_chunks)

#     def start_listening(self):
#         """Starts the microphone stream in a background thread."""
#         if self.listening:
#             return
            
#         self.listening = True
#         self.stream = self.p.open(format=pyaudio.paInt16,
#                                   channels=1,
#                                   rate=self.rate,
#                                   input=True,
#                                   frames_per_buffer=self.chunk_size)
        
#         self.capture_thread = threading.Thread(target=self._capture_audio)
#         self.capture_thread.daemon = True
#         self.capture_thread.start()
#         print("Ear: Started listening.")

#     def stop_listening(self):
#         """Stops the microphone stream."""
#         self.listening = False
#         if self.capture_thread:
#             self.capture_thread.join()
        
#         if self.stream:
#             self.stream.stop_stream()
#             self.stream.close()
#             self.p.terminate()
#         print("Ear: Stopped listening.")

#     def _capture_audio(self):
#         """Loop to read audio chunks and put them in the queue."""
#         while self.listening:
#             try:
#                 data = self.stream.read(self.chunk_size, exception_on_overflow=False)
#                 self.audio_queue.put(data)
#             except Exception as e:
#                 print(f"Ear Error: {e}")
#                 break

#     def listen(self):
#         """
#         Generator that yields transcribed text segments.
#         """
#         frames = []
#         silence_duration = 0
#         speech_detected = False
        
#         # VAD Parameters
#         # Increase SILENCE_THRESHOLD_MS to 1500 or 2000 so the user can pause without the agent interrupting
#         SILENCE_THRESHOLD_MS = 1500 
#         MIN_PHRASE_DURATION_MS = 300 # Lowered slightly to capture short commands
#         RMS_THRESHOLD = 300 # Lowered significantly so normal speech gets detected
        
#         while self.listening:
#             try:
#                 data = self.audio_queue.get(timeout=1)
#             except queue.Empty:
#                 continue
                
#             # 1. Check volume (RMS) first to filter background noise
#             rms = self._calculate_rms(data)
            
#             # Debug noise levels occasionally or if speech is detected but rejected?
#             # For now, let's trust the threshold.
            
#             # 2. Check VAD
#             try:
#                 # Only trust VAD if volume is above threshold
#                 if rms > RMS_THRESHOLD:
#                     is_speech = self.vad.is_speech(data, self.rate)
#                 else:
#                     # Optional: Print low RMS values to help debug? 
#                     # print(f"Noise ignored: {int(rms)}") 
#                     is_speech = False
#             except Exception:
#                 is_speech = False

#             if is_speech:
#                 if not speech_detected:
#                     print(f"Ear: Speech detected (RMS: {int(rms)})...")
#                     frames.extend(self.preroll_buffer)
#                     self.preroll_buffer.clear()
                    
#                 speech_detected = True
#                 silence_duration = 0
#                 frames.append(data)
#             else:
#                 if speech_detected:
#                     silence_duration += self.chunk_duration_ms
#                     frames.append(data)
                    
#                     if silence_duration > SILENCE_THRESHOLD_MS:
#                         # Check total duration of the phrase
#                         total_duration_ms = len(frames) * self.chunk_duration_ms
                        
#                         if total_duration_ms < MIN_PHRASE_DURATION_MS:
#                             print(f"Ear: Discarding short noise ({total_duration_ms}ms)")
#                             frames = []
#                             speech_detected = False
#                             silence_duration = 0
#                             continue
                            
#                         print(f"Ear: Processing phrase ({len(frames)} frames)...")
#                         audio_data = b''.join(frames)
                        
#                         # Convert int16 to float32
#                         audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
#                         try:
#                             # HF Pipeline expects the raw float array and sampling rate
#                             result = self.transcriber({"sampling_rate": self.rate, "raw": audio_np})
#                             text = result["text"].strip()
                            
#                             if text:
#                                 if self._is_hallucination(text):
#                                     print(f"Ear ignored hallucination: '{text}'")
#                                 else:
#                                     print(f"Ear heard: '{text}'")
#                                     yield text
#                         except Exception as e:
#                             print(f"Transcription error: {e}")
                        
#                         frames = []
#                         speech_detected = False
#                         silence_duration = 0
#                 else:
#                     self.preroll_buffer.append(data)

#     def _calculate_rms(self, data):
#         """Calculates RMS amplitude of the audio chunk."""
#         # Convert raw bytes to integers
#         shorts = np.frombuffer(data, dtype=np.int16)
#         # Avoid calculation on empty/error
#         if len(shorts) == 0:
#             return 0
#         # RMS = sqrt(mean(square(samples)))
#         sum_squares = np.sum(shorts.astype(np.float32)**2)
#         mean_squares = sum_squares / len(shorts)
#         return np.sqrt(mean_squares)

#     def _is_hallucination(self, text):
#         hallucinations = ["thanks for watching", "thank you", "subtitles by", "watching"]
#         text_lower = text.lower()
#         return any(h in text_lower for h in hallucinations) or len(text) < 2
