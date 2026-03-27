# `core/state.py` Explanation

## Overview
`core/state.py` defines a straightforward enumeration mapping (`AgentState`) to globally manage the high-level operational status of the Voice Agent across all of its asynchronous subsystems.

## Key Components

1. **`AgentState` Enum**:
   - `IDLE`: The agent is completely inactive, potentially waiting for an initial trigger or wake-word.
   - `LISTENING`: The system has actively engaged its Voice Activity Detection (VAD) and is actively buffering audio from the microphone to understand the user.
   - `THINKING`: The agent has shipped the recognized text off to the LLM (Brain) and is actively streaming inference tokens.
   - `SPEAKING`: AI inference has produced language and the TTS pipeline (Mouth) is actively occupying the speaker output.
   - `INTERRUPTED`: A special state triggered when the `VagusNerve` detects loud user audio while the agent is currently `SPEAKING`, causing an immediate barge-in.

This enum ensures type-safety, clean state transitions, and high readability in `main.py` when passing context between modules.
