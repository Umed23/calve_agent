# `core/mouth.py` Explanation

## Overview
`core/mouth.py` provides a basic Text-to-Speech (TTS) module using the lightweight, offline `pyttsx3` library. This acts as a fallback or fast-testing 'Mouth' for the agent, prioritizing ease of use over voice quality.

## Key Components

1. **Synchronous Speech Execution (`speak`)**:
   - The method accepts a string of text and an optional `check_interrupt_func`.
   - Before synthesizing speech, it actively checks if the user has interrupted. If an interruption is detected, the function aborts before starting audio playback.
   
2. **Engine Initialization**:
   - The `pyttsx3` engine is instantiated, used, and forcefully deleted within a `try/except` block on *every* call. This deliberate design pattern circumvents standard `pyttsx3` event loop deadlocks that frequently occur when running multiple TTS calls across different threads sequentially.
   - Playback speed (rate) is constrained to `150` for clear intelligibility.

3. **Interruption Limitations**:
   - Because `engine.runAndWait()` blocks the thread completely until the entire string finishes playing, this specific module cannot halt audio *mid-sentence*. It can only skip consecutive text blocks if streaming is implemented elsewhere to chunk the audio. (For a more advanced interruptible architecture, `mouth_neural.py` is utilized).
