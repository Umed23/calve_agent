
import sys
import os
import time

sys.path.append(os.getcwd())

try:
    from core.mouth_neural import NeuralMouth  # type: ignore
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test():
    print("Testing NeuralMouth (Multi-Language)...")
    try:
        mouth = NeuralMouth(use_gpu=False)
        
        # Test 1: English (Default)
        text_en = "Hello! I am speaking in English."
        print(f"1. English: '{text_en}'")
        mouth.speak_stream([text_en])
        time.sleep(4)
        
        # Test 2: Hindi (Auto-switch)
        # "Namaste! Main Hindi bol sakta hoon." (In Devanagari)
        text_hi = "नमस्ते! मैं हिंदी बोल सकता हूँ।"
        print(f"2. Hindi: '{text_hi}'")
        mouth.speak_stream([text_hi])
        time.sleep(5)
        
        # Test 3: Switch back to English
        text_en_2 = "Now I am back to English."
        print(f"3. English: '{text_en_2}'")
        mouth.speak_stream([text_en_2])
        time.sleep(4)
        
        mouth.stop()
        print("SUCCESS: Multi-language test completed.")
        
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test()
