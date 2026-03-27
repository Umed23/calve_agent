# `main.py` Explanation

## Overview
`main.py` serves as the primary entry point for the CALVE Voice Agent streaming application. It coordinates the different sub-systems (Ear, Brain, Mouth, and Vagus Nerve) to establish a continuous, real-time voice interaction loop.

## Key Components

1. **Monkey-Patching `torchaudio.load`**:
   At the beginning of the file, `torchaudio.load` is overridden with a custom `safe_load` function that uses the `soundfile` library. This is a robust way to ensure audio files are loaded correctly (as float32 tensors) regardless of the underlying OS audio backends, solving potential `torchaudio` backend issues.

2. **Component Initialization**:
   The `main()` function initializes the four core agent components:
   - **`Ear`**: The Speech-to-Text (STT) component that continuously listens on a background thread.
   - **`Brain`**: The Large Language Model (LLM) component that processes the recognized text and generates responses streamingly.
   - **`NeuralMouth`**: The Text-to-Speech (TTS) component responsible for converting text into audio output.
   - **`VagusNerve`**: A background monitor evaluating the microphone's audio stream to detect user interruptions.

3. **Execution Flow**:
   - The system starts by playing a predefined introductory greeting (`intro_text`) in Hindi.
   - It runs an infinite loop capturing the user's spoken input seamlessly through `ear.listen()`.
   - When text is detected, the `Brain` evaluates it and streams tokens to the `NeuralMouth` via `mouth.speak_stream()`.
   - The `check_interruption` local function utilizes the `VagusNerve` to halt the agent's speech instantly if the user attempts to interject.
   - State variables (`AgentState.IDLE`, `LISTENING`, `THINKING`, `SPEAKING`) are maintained for tracking the current operation.

4. **Graceful Shutdown**:
   The `KeyboardInterrupt` (`Ctrl+C`) block cleanly stops the background listening threads and TTS components before closing the application.
