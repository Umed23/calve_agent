from enum import Enum

class AgentState(Enum):
    IDLE = "idle"              # Waiting for wake word or VAD
    LISTENING = "listening"    # VAD triggered, accumulating audio
    THINKING = "thinking"      # STT complete, LLM processing
    SPEAKING = "speaking"      # TTS playing audio
    INTERRUPTED = "interrupted"  # Barge-in detected during SPEAKING
