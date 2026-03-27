# `utils/audio.py` Explanation

## Overview
`utils/audio.py` stores reusable, global audio configuration constants and helper functions to prevent magic numbers from being scattered across the codebase.

## Key Components

1. **Audio Constants**:
   - `CHUNK (1024)`: The default buffer block size for reading PyAudio streams.
   - `FORMAT (paInt16)`: 16-bit integer format, standard for PCM voice data.
   - `CHANNELS (1)`: Mono audio (essential for speech recognition models which usually expect mono).
   - `RATE (16000)`: 16kHz sampling rate, the standard frequency required by Whisper and most other STT/TTS architectures.

2. **Helper Functions**:
   - `create_audio_stream(p)`: A convenience function that returns a pre-configured PyAudio stream using the standard constants. (Though `Ear` defines its own stream locally, this is useful globally).
   - `is_silence(data, threshold)`: A programmatic volume monitor that evaluates the mean absolute amplitude of an audio frame array relative to a baseline to determine if a chunk contains dead silence.
