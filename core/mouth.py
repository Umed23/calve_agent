import pyttsx3
import threading

class Mouth:
    """
    Handles Text-to-Speech (TTS).
    Currently uses pyttsx3 for offline TTS, but can be swapped for ElevenLabs/OpenAI.
    """
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text, check_interrupt_func=None):
        """
        Speaks the given text.
        Args:
            text: string to speak
            check_interrupt_func: function that returns True if we should stop speaking
        """
        if not text:
            return

        print(f"Agent: {text}")
        
        if check_interrupt_func and check_interrupt_func():
            print("[Interrupted]")
            return

        try:
             # Re-initialize engine each time to avoid loop issues
             engine = pyttsx3.init()
             engine.setProperty('rate', 150)
             engine.setProperty('volume', 1.0)
             
             engine.say(text)
             engine.runAndWait()
             engine.stop()
             del engine
        except Exception as e:
            print(f"TTS Error: {e}")

    def stop(self):
        pass
