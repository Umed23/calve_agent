# `core/mouth_neural.py` Explanation

## Overview
`core/mouth_neural.py` provides an advanced, streaming Text-to-Speech component (`NeuralMouth`) using Facebook's `MMS-TTS-Hin` (VITS-based model). It generates high-fidelity, natural-sounding Hindi speech on-device, processing text chunk-by-chunk dynamically to eliminate rigid wait times.

## Key Components

1. **Text Preprocessor (`REPLACEMENTS`)**:
   - Because patients often use domain-specific English words in Hindi ("appointment", "booking", "OTP", "cancel"), this module implements a regex-based `preprocess_text` substitution map.
   - It intercepts English characters and replaces them with phonetically equivalent Devanagari text (e.g. "cancel" -> "रद्द" or "otp" -> "ओ टी पी") before sending it to the TTS engine. This ensures the Hindi-only TTS model does not stumble or skip over English vocabulary.

2. **Asynchronous Audio Queue (`_playback_loop`)**:
   - Audio generation and audio playback are decoupled.
   - Synthesized audio chunks (`numpy` arrays) are pushed to an `audio_queue`. A dedicated daemon thread perpetually pulls from this queue and feeds it to `sounddevice` (`sd.play`), allowing continuous uninterrupted speech playback.

3. **Stream Chunking (`speak_stream`)**:
   - It iterates over a generator of LLM text tokens (from the Brain).
   - Once the accumulated tokens reach a natural stop marker (punctuation like `.`, `?`, `\n`, `,`, or `।`) or a max length, the buffer is flushed to synthesis. This allows the first half of a sentence to be spoken while the second half is still being evaluated by the AI.

4. **DeepFilterNet Environment Enhancement**:
   - Optional `enhance_audio` flag integration. If enabled, the synthesized output is routed through `DeepFilterNet` (`df.enhance`). This denoises and boosts the clarity of the audio output to sound more studio-quality.
   
5. **Instant Interruption (`stop_immediately`)**:
   - Checks `check_interrupt_func` and sets `self.stop_event`. This immediately halts `sounddevice` playback, clears the pending queue, and breaks the token loop — giving the agent immediate responsiveness when users interrupt.

## Fixes Implemented
- **Temporary file cleanup**: When `DeepFilterNet` creates `_temp_mouth.wav` out of necessity to shape tensors correctly, it was previously abandoning the file on disk. Added a surgical fix to silently delete the temporary file after it is successfully read back, preventing disk clutter.
