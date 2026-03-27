import os
import sys
import time

sys.path.append(os.getcwd())

try:
    from core.mouth_neural import NeuralMouth
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_preprocessing():
    print("Testing Text Preprocessor...")
    text = "Hello! Your Order status is cancelled. The Refund will come soon."
    clean = NeuralMouth.preprocess_text(text)
    print(f"Original: {text}")
    print(f"Cleaned:  {clean}")
    
    # Assert conditions for phonetics
    assert "हेलो" in clean, "Preprocessing failed on 'Hello'"
    assert "ऑर्डर" in clean, "Preprocessing failed on 'Order'"
    assert "रिफंड" in clean, "Preprocessing failed on 'Refund'"
    
def test_tts(enhance):
    print(f"\nTesting Speech Pipeline (enhance_audio={enhance})...")
    
    # Make sure we don't open Playback threading queue, we just want to save
    mouth = NeuralMouth(use_gpu=False, play_audio=False, enhance_audio=enhance)
    tokens = ["Hello! ", "Your refund ", "is processing."]
    
    start_time = time.time()
    mouth.speak_stream(tokens)
    end_time = time.time()
    print(f"Time to synthesize frames: {end_time - start_time:.4f} sec")
    
    while not mouth.audio_queue.empty():
        time.sleep(0.1)
    time.sleep(1)
    
    output_name = f"test_en_to_hin_{'enhanced' if enhance else 'normal'}.wav"
    mouth.save_audio(output_name)
    mouth.stop()
    
    if os.path.exists(output_name):
        print(f"SUCCESS: Saved to {output_name}")
    else:
        print(f"FAILED: Did not save {output_name}")

if __name__ == "__main__":
    test_preprocessing()
    
    # Test Normal Pipeline (fast)
    test_tts(enhance=False)
    
    try:
        import df
        # Test Enhanced Pipeline (DeepFilterNet)
        test_tts(enhance=True)
    except ImportError:
        print("\nDeepFilterNet not installed. Skipping enhanced test.")
