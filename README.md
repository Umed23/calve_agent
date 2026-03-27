# CALVE Voice Agent

A modular, async voice agent implementing the "Listen - Think - Speak" loop with interruption handling.

## Components

- **Ear (`core/ear.py`)**: Listens to microphone using `pyaudio` and captures audio in a background thread. Uses `webrtcvad` for silence detection and `faster-whisper` for speech-to-text.
- **Brain (`core/brain.py`)**: Uses `LangGraph` and `Claude 3.5 Sonnet` via `langchain-anthropic` to generate intelligent responses.
- **Mouth (`core/mouth.py`)**: Uses `pyttsx3` for text-to-speech synthesis.
- **Vagus (`core/vagus.py`)**: Monitors for interruptions (Voice Activity Detection).
- **Main (`main.py`)**: Orchestrates the async loop: Listen -> Think -> Speak.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - Create a `.env` file in the root directory.
    - Add your API key:
      ```
      ANTHROPIC_API_KEY=your_api_key_here
      ```
    - (Optional) Add `OPENAI_API_KEY` if you plan to switch TTS/STT providers.

3.  **Run**:
    ```bash
    python main.py
    ```

## Notes

- **Whisper Model**: Defaults to `tiny` model on `cpu` for speed. To change this, modify `model_size` in `main.py` (options: `tiny`, `base`, `small`, `medium`, `large`). If you have a CUDA-enabled GPU, change `device="cpu"` to `device="cuda"`.
- **Interruption**: The `VagusNerve` class provides the logic for detecting speech during agent output. In a full implementation, this signal would be used to immediately stop audio playback.
- **Microphone**: Ensure your default microphone is set correctly in your OS settings.

## Troubleshooting

- **PyAudio Issues**: If `pip install pyaudio` fails, try installing `portaudio` first (e.g., `brew install portaudio` on Mac) or download a pre-compiled `.whl` file for your Python version on Windows.
- **WebRTCVAD**: Requires a C++ compiler. If installation fails, ensure you have build tools installed.
