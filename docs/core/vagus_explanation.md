# `core/vagus.py` Explanation

## Overview
`core/vagus.py` defines the `VagusNerve` class, a specialized component designed exclusively for fast, low-latency barge-in (interruption) detection while the agent is actively talking. 

## Key Components

1. **Voice Activity Detection (`webrtcvad`)**:
   - Initializes an extremely aggressive VAD instance (`sensitivity=3`) explicitly tuned to catch human vocal frequencies quickly.

2. **`check_for_interruption` Method**:
   - Exposed to be called synchronously by the `Mouth` module inside its text-synthesis loop (via the `check_interrupt_func` delegate).
   - It directly probes the PyAudio `stream` for a raw `CHUNK` of audio data. If it detects speech in that discrete slice, it flips `self.interrupted` to `True` and aborts the ongoing TTS generation.

## Architectural Notes
Currently, `vagus` probes the PyAudio stream concurrently with `Ear`. PyAudio C-bindings generally allow this, but it results in `VagusNerve` effectively "stealing" an audio frame from the `Ear`'s continuous buffer. This is acceptable for a prototype because if the user provides a barge-in command (e.g. speaking for >300ms), both the stolen `vagus` frames and the remaining `ear` frames will contain the speech, ensuring the interruption triggers and the message is still roughly transcribed by Whisper.
