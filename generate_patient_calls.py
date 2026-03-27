import os
import time
from core.brain import Brain
from core.mouth_neural import NeuralMouth

def main():
    print("Initializing Patient Calls Generator...")
    
    # Initialize components
    brain = Brain()
    
    # Initialize NeuralMouth with play_audio=False so it saves to array rather than playing to default audio output map
    mouth = NeuralMouth(use_gpu=False, play_audio=False) 
    
    # Mock patient phrases to simulate incoming transcribed queries
    dialogues = [
        "मुझे अपॉइंटमेंट चाहिए", # I want an appointment
        "कल सुबह का टाइम मिलेगा क्या?", # Is tomorrow morning available?
        "धन्यवाद, मैं कल आऊंगा।" # Thank you, I will come tomorrow
    ]
    
    print("\nStarting generation process...")
    
    # Give the NeuralMouth TTS engine time to warm up if necessary
    time.sleep(2)

    for i, user_text in enumerate(dialogues):
        print(f"\n[{i+1}/{len(dialogues)}] Simulated User: {user_text}")
        
        # Brain processes the incoming query and returns a token generator
        token_stream = brain.think_stream(user_text)
        
        print("Agent speaking...")
        
        # Pass tokens to NeuralMouth.
        # It generates audio chunks asynchronously and queues them up.
        mouth.speak_stream(token_stream)
        
        # mouth.save_audio() will safely block until all enqueued chunks are fully processed into `accumulated_audio`
        filename = f"demo_call_{i+1}.wav"
        mouth.save_audio(filename)
        
    mouth.stop()
    print("\nAll calls generated successfully.")

if __name__ == "__main__":
    main()
