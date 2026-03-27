# Test Scripts Explanation

## Overview
The project contains three main functional test scripts in the root directory used to validate changes to the Text-to-Speech sub-system independently of the core agent loops.

## Scripts Analyzed

1. **`test_enhancements.py`**:
   - Tests if the `NeuralMouth.preprocess_text` regex safely converts target English domain terms to Hindi phonetics.
   - Verifies if the TTS can synthesize audio successfully locally.
   - Tests conditional integration with the optional `DeepFilterNet` library (`enhance_audio=True`) to ensure the noise reduction models execute without crashing and save output `.wav` files.

2. **`test_tts.py`**:
   - A straightforward check to instantiate `NeuralMouth` and stream a single English sentence to the audio output, bypassing the LLM. 

3. **`test_tts_hindi.py`**:
   - Evaluates the ability of `NeuralMouth` to fluidly switch between synthesizing Latin characters (English) and Devanagari characters (Hindi) consecutively.
   
## Fixes Implemented
- The agent's `NeuralMouth` component was recently refactored to a chunk-based `speak_stream` mechanism, rendering old synchronous `speak()` methods obsolete. The `test_tts.py` and `test_tts_hindi.py` scripts were still calling `mouth.pipeline` and `mouth.speak(text)`. These were patched to `mouth.model` and `mouth.speak_stream([text])` to restore full testing functionality.
