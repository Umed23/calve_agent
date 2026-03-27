# `generate_patient_calls.py` Explanation

## Overview
`generate_patient_calls.py` is a standalone utility script. Rather than serving as an interactive virtual agent, it operates in batch mode to simulate a dialogue sequence and export the AI's generated voice responses as `.wav` files.

## Key Components

1. **Silent Initialization**:
   - The script initializes `Brain` and `NeuralMouth`, but specifically instantiates the mouth with `play_audio=False`. This tells the TTS callback thread to push generated NumPy audio frames into an internal `accumulated_audio` list instead of routing them to the speaker (`sounddevice`).

2. **Mock Simulation (`dialogues`)**:
   - It iterates through a hardcoded list of common patient Hindi phrases (e.g., "I want an appointment", "Thank you").
   - For each phrase, it feeds the text to the `Brain` model, which streams back response tokens.
   
3. **Audio File Exporting (`save_audio`)**:
   - The script streams the AI's tokens into the TTS engine. Once the engine is done synthesizing the final audio chunk for that phrase, the script calls `mouth.save_audio(filename)`. 
   - This cleanly concatenates the buffered audio chunks using `numpy` and exports them to disk rapidly (`demo_call_1.wav`, `demo_call_2.wav`, etc.) for offline evaluation of the LLM responses and TTS quality.
