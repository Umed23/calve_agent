# `core/ear.py` Explanation

## Overview
`core/ear.py` is responsible for capturing the user's speech via the microphone (Speech-to-Text), ignoring background noise, and accurately transcribing Hindi audio into text using an AI model.

## Key Components

1. **Audio Capture Thread**:
   - The application opens a continuous audio stream using `pyaudio` at a 16kHz sampling rate.
   - A background daemon thread (`_capture_audio`) continuously reads audio frames and places them into an in-memory `queue`. This ensures that capturing audio does not block the main application thread.

2. **Voice Activity Detection (VAD)**:
   - Uses `webrtcvad` configured with maximum aggressiveness (level 3) to precisely identify human speech.
   - Added a supplemental RMS volume calculation (`_calculate_rms`) to act as a noise gate. VAD is only executed if audio volume exceeds a baseline threshold (`RMS_THRESHOLD = 300`), eliminating false positives from low-volume background static.

3. **Pre-roll Buffer & Silence Detection**:
   - It maintains a sliding window of the last 500ms of audio (the "pre-roll buffer"). When speech is finally detected, this window is instantaneously prepended to the capture so the very start of words (plosives, consonants) aren't clipped off.
   - Once the user stops speaking (indicated by 800ms of consecutive silence), the system assumes a full phrase has been spoken and initiates transcription. Miniscules noises (< 300ms) are explicitly tossed out.

4. **Background Transcription (`ARTPARK Vaani`)**:
   - Transcription relies on the HuggingFace `pipeline` invoking `ARTPARK-IISc/whisper-small-vaani-hindi`, a specialized ASR model fine-tuned for Indian languages.
   - Crucially, transcription happens completely asynchronously in *another* background thread (`_transcribe_async`). This ensures the "Ear" can instantly pivot back to listening without waiting for inference processing.

5. **Hallucination Filtering (`_is_hallucination`)**:
   - Whisper models tend to hallucinate common training data watermarks (e.g. "thanks for watching", "subtitles by") during total silence. This script robustly catches and discards those common occurrences.
