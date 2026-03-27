"""
local_agent.py — CALVE Local Voice Agent (Desktop)

Run this locally for the microphone-driven Hindi voice agent.
This was the original main.py before the Render web service restructure.

Requirements: requirements-local.txt
Usage: python local_agent.py
"""

import os
import time
import threading
from dotenv import load_dotenv
import torch
import torchaudio
import soundfile as sf

# Monkey-patch torchaudio.load globally
def safe_load(filepath, **kwargs):
    audio_np, sr = sf.read(filepath, dtype='float32')
    if audio_np.ndim > 1:
        audio = torch.from_numpy(audio_np).T
    else:
        audio = torch.from_numpy(audio_np).unsqueeze(0)
    return audio, sr

torchaudio.load = safe_load
torchaudio.info = lambda x: None

from core.ear import Ear
from core.brain import Brain
from core.mouth_neural import NeuralMouth
from core.vagus import VagusNerve
from core.state import AgentState

# Load environment variables
load_dotenv()


def main():
    print("Initializing CALVE Voice Agent (Streaming Architecture)...")

    # Initialize components
    try:
        stt_model = os.getenv("STT_MODEL_SIZE", "small")
        stt_lang = os.getenv("STT_LANGUAGE", None)  # None = Auto-detect

        ear = Ear(model_size=stt_model, device="cpu", compute_type="int8", language=stt_lang)
        brain = Brain()
        mouth = NeuralMouth(use_gpu=False)
        vagus = VagusNerve(sensitivity=3)

        state = AgentState.IDLE

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("Agent is ready. Speak into the microphone.")
    print("Press Ctrl+C to exit.")

    # Start listening background thread
    ear.start_listening()

    try:
        text_generator = ear.listen()

        def check_interruption():
            if vagus.check_for_interruption(ear.stream):
                return True
            return False

        # Agent introduction
        intro_text = """नमस्ते! मैं काल्वे की ओर से शर्मा क्लिनिक के लिए आपसे संपर्क कर रहा हूँ।
        मैं आपकी अपॉइंटमेंट के संबंध में सहायता करने के लिए यहाँ हूँ।

        क्या आप नई अपॉइंटमेंट लेना चाहते हैं या अपनी मौजूदा अपॉइंटमेंट में कोई बदलाव करना चाहते हैं?
        कृपया बताइए, मैं आपकी सहायता करने की पूरी कोशिश करूँगा।
        """
        print(f"\nAgent: {intro_text}")
        current_state = AgentState.SPEAKING
        mouth.speak_stream([intro_text], check_interrupt_func=check_interruption)

        current_state = AgentState.LISTENING
        print("\nEar: Waiting for you to speak...")

        for user_text in text_generator:
            print(f"\nUser: {user_text}")

            if "exit" in user_text.lower() or "quit" in user_text.lower():
                print("Exiting...")
                break

            current_state = AgentState.THINKING

            if user_text:
                token_stream = brain.think_stream(user_text)
                current_state = AgentState.SPEAKING
                mouth.speak_stream(token_stream, check_interrupt_func=check_interruption)
                current_state = AgentState.LISTENING

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ear.stop_listening()
        mouth.stop()
        print("Agent Stopped.")


if __name__ == "__main__":
    main()
