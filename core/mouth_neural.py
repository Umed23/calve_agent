from transformers import AutoTokenizer, VitsModel
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import os
import torch
import re
import logging

logger = logging.getLogger(__name__)

class NeuralMouth:
    """
    A neural TTS engine using MMS-TTS-Hin for high-quality Hindi.
    """
    
    # ============================================================
    # TEXT PREPROCESSOR — fixes domain terms for natural speech
    # ============================================================
    REPLACEMENTS = {
        # Greetings
        "hello": "हेलो", "hi": "हाय", "bye": "बाय",
        "ok": "ओके", "okay": "ओके",

        # Customer care English terms → Hindi phonetic
        "sorry": "सॉरी", "thank you": "धन्यवाद", "thanks": "शुक्रिया",
        "order": "ऑर्डर", "refund": "रिफंड", "complaint": "शिकायत",
        "status": "स्थिति", "cancel": "रद्द", "payment": "भुगतान",
        "account": "खाता", "customer": "ग्राहक", "support": "सहायता",
        "issue": "समस्या", "problem": "समस्या", "resolve": "हल",
        "callback": "वापस कॉल", "email": "ईमेल", "mobile": "मोबाइल",
        "number": "नंबर", "otp": "ओ टी पी", "pin": "पिन",
        "password": "पासवर्ड", "login": "लॉगिन", "logout": "लॉगआउट",
        "app": "ऐप", "update": "अपडेट", "download": "डाउनलोड",
        "website": "वेबसाइट", "link": "लिंक", "call": "कॉल",
        "chat": "चैट", "ticket": "टिकट", "booking": "बुकिंग",
        "delivery": "डिलीवरी", "address": "पता", "manager": "मैनेजर",
        "department": "विभाग", "service": "सेवा", "feedback": "प्रतिक्रिया",
        "rating": "रेटिंग", "offer": "ऑफर", "discount": "छूट",
        "coupon": "कूपन", "bill": "बिल", "invoice": "चालान",
        "report": "रिपोर्ट", "id": "आई डी", "verify": "सत्यापित",
        "verified": "सत्यापित",

        # Numbers → Hindi words
        "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन",
        "4": "चार", "5": "पांच", "6": "छह", "7": "सात",
        "8": "आठ", "9": "नौ", "10": "दस",

        # Punctuation
        "...": "। ", " - ": " ",
    }
    
    @classmethod
    def preprocess_text(cls, text):
        sorted_replacements = sorted(
            cls.REPLACEMENTS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for eng, hindi in sorted_replacements:
            pattern = r'\b' + re.escape(eng) + r'\b'
            text = re.sub(pattern, hindi, text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()
        
        # We don't artificially add "।" here because we are streaming tokens and 
        # want to speak partial clauses without forcing sentence ends abruptly.
        return text

    def __init__(self, use_gpu=False, language='h', play_audio=True, enhance_audio=False):
        logger.info(f"[NeuralMouth] Initializing MMS-TTS-Hin (enhance_audio={enhance_audio})...")
        try:
            self.play_audio = play_audio
            self.accumulated_audio = []
            self.device = 'cuda' if use_gpu else 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
                
            self.model_name = "facebook/mms-tts-hin"
            
            # Disable symlink warnings (especially on Windows)
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            import warnings
            warnings.filterwarnings("ignore")
            
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use model's expected sampling rate
            self.sample_rate = self.model.config.sampling_rate
            self.language = language 
            
            # VITS does not use a voice prompt
            self.voice_prompt = ""
            
            self.enhance_audio = enhance_audio
            self.df_model, self.df_state = None, None
            if self.enhance_audio:
                try:
                    from df.enhance import init_df
                    logger.info("[NeuralMouth] Initializing DeepFilterNet enhancer...")
                    self.df_model, self.df_state, _ = init_df()
                except ImportError:
                    logger.warning("[NeuralMouth] Warning: deepfilternet package not installed. Setting enhance_audio=False.")
                    self.enhance_audio = False
            
            # Queue for audio playback
            self.audio_queue = queue.Queue()
            self.is_speaking = False
            self.stop_event = threading.Event()
            
            # Start background playback thread
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            logger.info(f"[NeuralMouth] Ready ({self.device}). Language: {self.language}")
            
        except Exception as e:
            logger.error(f"[NeuralMouth] Error: Failed to load Indic Parler-TTS. {e}", exc_info=True)
            self.model = None

    def set_language(self, lang_code):
        # Indic Parler-TTS handles multiple Indian languages directly in the text payload.
        # No need to swap models.
        self.language = lang_code

    def _playback_loop(self):
        """Background loop to play audio chunks."""
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if self.play_audio:
                    self.is_speaking = True
                    sd.play(audio_chunk, self.sample_rate)
                    sd.wait()
                    self.is_speaking = False
                else:
                    self.accumulated_audio.append(audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[NeuralMouth] Playback Error: {e}", exc_info=True)
                self.is_speaking = False

    def speak_stream(self, token_iterator, check_interrupt_func=None):
        """
        Consumes a stream of tokens, buffers them into sentences, and speaks them.
        """
        if not self.model:
            logger.warning("[NeuralMouth] TTS Unavailable")
            return

        sentence_buffer = ""
        # Simple sentence delimiters (added comma and Hindi danda for lower latency)
        delimiters = {'.', '?', '!', '\n', '।', '|', ','}
        
        self.stop_event.clear()
        
        if check_interrupt_func and check_interrupt_func():
            logger.info("[NeuralMouth] Stream interrupted before start.")
            return

        for token in token_iterator:
            if self.stop_event.is_set():
                logger.info("[NeuralMouth] Stopping stream due to interruption signal.")
                break
                
            if check_interrupt_func and check_interrupt_func():
                logger.info("[NeuralMouth] Stream interrupted during generation.")
                self.stop_immediately()
                break
                
            sentence_buffer += str(token)
            
            # Check if token ends with a delimiter or contains one
            if any(d in token for d in delimiters) or len(sentence_buffer) > 100: # fallback length
                 # Synthesize this chunk
                 text_to_speak = sentence_buffer.strip()
                 if text_to_speak:
                    self._synthesize_and_enqueue(text_to_speak)
                    
                 sentence_buffer = ""

        # Speak remaining buffer
        if sentence_buffer.strip() and not self.stop_event.is_set():
             self._synthesize_and_enqueue(sentence_buffer.strip())

    def _synthesize_and_enqueue(self, text):
        if not self.model: return
        try:
            clean_text = self.preprocess_text(text)
            logger.info(f"[NeuralMouth] Synthesizing segment: '{clean_text[:50]}...'")
            
            inputs = self.tokenizer(clean_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model(**inputs).waveform
                
            audio_np = output.squeeze().cpu().float().numpy()
            raw_sr = self.model.config.sampling_rate
            
            if self.enhance_audio and self.df_model and self.df_state:
                from df.enhance import enhance
                import soundfile as sf
                
                # df.enhance expects a specific torchaudio tensor shape and rate
                # We save temp file to properly reload it with df's load_audio,
                # as the user script does, because resamplers and channel formats vary.
                sf.write("_temp_mouth.wav", audio_np, raw_sr)
                from df.enhance import load_audio
                audio_tensor, _ = load_audio("_temp_mouth.wav", sr=self.df_state.sr())
                try:
                    os.remove("_temp_mouth.wav")
                except OSError:
                    pass
                
                enhanced = enhance(self.df_model, self.df_state, audio_tensor)
                
                # Enhanced is a tensor, we need to convert it back to numpy for sounddevice to play
                # Assuming shape is [channels, frames]
                audio_np = enhanced.squeeze().cpu().numpy()
                self.sample_rate = self.df_state.sr() # Update playback sr to df sr!
            else:
                self.sample_rate = raw_sr
            
            if self.stop_event.is_set(): 
                return
                
            self.audio_queue.put(audio_np)
        except Exception as e:
            logger.error(f"[NeuralMouth] Generation Error: {e}", exc_info=True)

    def stop_immediately(self):
        """Cancels all playback and clears queues."""
        logger.warning("[NeuralMouth] EMERGENCY STOP invoked")
        self.stop_event.set()
        
        # Clear the queue
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        # Stop sounddevice immediately
        sd.stop()
    def save_audio(self, filename):
        import scipy.io.wavfile as wavfile
        
        if not hasattr(self, 'audio_queue') or self.model is None:
            print("Mouth: TTS model is not loaded. Cannot save audio.")
            return

        # Wait for all generated audio to be processed by _playback_loop
        self.audio_queue.join() 
        
        if not getattr(self, 'accumulated_audio', []):
            print("Mouth: No audio to save.")
            return
            
        combined = np.concatenate(self.accumulated_audio)
        wavfile.write(filename, self.sample_rate, combined)
        self.accumulated_audio = [] # Clear after saving
        print(f"Mouth: Saved combined audio to {filename}")

    def stop(self):
        """Stops the mouth gracefully."""
        self.stop_event.set()
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)

    def _contains_hindi(self, text):
        return any('\u0900' <= char <= '\u097f' for char in text)
        
    def _is_pure_ascii(self, text):
        return all(ord(char) < 128 for char in text)

