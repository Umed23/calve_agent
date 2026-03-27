
import sys
import os
import time

# Add current directory to path so we can import core
sys.path.append(os.getcwd())

try:
    from core.mouth_neural import NeuralMouth  # type: ignore
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test():
    print("Testing NeuralMouth...")
    try:
        mouth = NeuralMouth(use_gpu=False)
        if mouth.model is None:
            print("FAILURE: Pipeline failed to initialize.")
            return

        text = "Hello! I am Kokoro, your new neural voice. I hope you like how I sound."
        print(f"Speaking: '{text}'")
        mouth.speak_stream([text])
        
        # Wait for playback (approx duration of text)
        time.sleep(5)
        
        mouth.stop()
        print("SUCCESS: Test completed without errors.")
        
    except Exception as e:
        print(f"FAILURE: Exception during test: {e}")

if __name__ == "__main__":
    test()
