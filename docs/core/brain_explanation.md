# `core/brain.py` Explanation

## Overview
`core/brain.py` houses the `Brain` component, which acts as the central logic and decision-making unit for the virtual receptionist. It uses LangChain and LangGraph to manage the dialogue flow securely and conditionally invoke tools when needed.

## Key Components

1. **Agent State Management**:
   The `AgentState` extends LangGraph's state schema, holding a `messages` sequence. This allows the graph to accumulate conversational context chronologically.

2. **Tool Integration (`check_appointment_availability`)**:
   A mock tool is defined and bound to the language model. When a user asks about appointment dates, the LLM intercepts the intent and executes this Python function to "check the DB" before delivering a confirmation in natural language.

3. **`Brain` Class Initialization**:
   - The constructor initializes Anthropic's `Claude 3.5 Sonnet` model if an API key is available. Otherwise, it falls back to a simulated echo response mechanism.
   - It strictly configures a detailed `system_prompt` to constrain the agent's identity to a Hindi-speaking clinic assistant operating for "Sharma Clinic". The system prompt enforces politeness and forbids the use of English script.
   - Using LangGraph, it compiles a workflow (`StateGraph`) consisting of evaluating the model (`_call_model`) and executing tool nodes conditionally (`tools_condition`).

4. **Inference / Invocation**:
   - `think(user_input)`: A standard synchronous method that appends the user context, triggers the whole workflow execution, updates local message history with the final outcome, and returns the spoken text.
   - `think_stream(user_input)`: Operates the model in streaming mode to reduce wait latency. Using LangGraph's token-by-token emission (`stream_mode="messages"`), it proactively yields chunks to the `NeuralMouth` directly so that audio synthesis can begin almost immediately before the model finishes its total response.

## Fixes Implemented
- **Added `_call_model`**: Previously missing, this function was added to successfully bind the `agent` node in to the LangGraph `_build_graph` definition. This ensures the model correctly evaluates the accumulated `messages` list and routes back the generated state.
