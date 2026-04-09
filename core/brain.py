from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import os
import time
import requests
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- CLINIC ASSISTANT TOOLS ---
@tool
def check_appointment_availability(date: str) -> str:
    """Checks if the doctor is available for an appointment on a given date. 
    Pass the date in natural language, e.g., 'tomorrow' or 'next week'."""
    logger.info(f"[TOOL: check_appointment_availability] Querying DB for appointments on: {date}")
    # Mock database lookup
    time.sleep(1) 
    return f"Yes, there are open slots in the morning on {date}."

tools = [check_appointment_availability]

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

class Brain:
    """
    The 'Brain' of the agent, using LangGraph and Claude 3.5 Sonnet as the master model.
    """
    def __init__(self):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.simulated = False
        
        # Hardcode System Prompt for Clinic Assistant in Hindi
        self.system_prompt = """You are 'CALVE', an empathetic, helpful, and professional virtual receptionist for a healthcare clinic in India (Sharma Clinic).
        Your goal is to answer patient inquiries, book appointments, and provide basic clinic information.

        STRICT RULES:
        1. ALWAYS respond only in Hindi using Devanagari script.
        2. DO NOT use any English words, English letters, or Roman script in responses.
        3. Use simple, natural, spoken-style Hindi suitable for conversation with patients in India.
        4. DO NOT use markdown, emojis, bullet points, or long paragraphs, as the response will be spoken aloud by a TTS engine.
        5. Keep answers under 2 sentences unless something requires a slightly longer explanation.
        6. Be exceptionally polite and respectful. If the conversation is already ongoing, DO NOT repeat the "welcome" greeting—just answer their question directly.
        7. If the user asks for an appointment, ALWAYS use the `check_appointment_availability` tool to check the calendar before confirming."""
                
        if anthropic_key:
            logger.info("[Brain] Initializing Anthropic (Claude 3.5 Sonnet) with Tool Calling")
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                api_key=anthropic_key
            ).bind_tools(tools)
        else:
            logger.warning("[Brain] WARNING: No ANTHROPIC_API_KEY found. Falling back to simulated text.")
            self.simulated = True

        if not self.simulated:
            self.workflow = self._build_graph()
            self.app = self.workflow.compile()
            
        self.history = [SystemMessage(content=self.system_prompt)]

    def _call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(tools))

        # Define edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition, 
        )
        workflow.add_edge("tools", "agent")

        return workflow

    def _is_ollama_running(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            return response.status_code == 200
        except:
            return False

    def think(self, user_input: str):
        """
        Processes user input and returns the agent's response.
        """
        logger.info(f"[Brain] Thinking about user input: '{user_input}'...")
        
        if self.simulated:
            time.sleep(1) # simulate thinking
            return f"I heard you say: {user_input}. (Simulated Response)"
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # We can pass history here if we want to maintain conversation context
        # For this simple loop, we just pass the new message, 
        # but in a real app we'd append to self.history
        self.history.append(HumanMessage(content=user_input))
        
        config = {"recursion_limit": 50}
        try:
            result = self.app.invoke({"messages": self.history}, config=config)
            response_message = result["messages"][-1]
            response_text = response_message.content
            self.history.append(response_message)
            return response_text
        except Exception as e:
            logger.error(f"[Brain] Error during thinking: {e}", exc_info=True)
            return "I am having trouble connecting to my brain. Please check your API key and quota."

    def think_stream(self, user_input: str):
        """
        Generator that yields tokens from the LLM.
        """
        logger.info(f"[Brain] Streaming thought process for: '{user_input}'...")
        
        if self.simulated:
            time.sleep(0.5)
            yield "I "
            time.sleep(0.1)
            yield "heard "
            time.sleep(0.1)
            yield f"you: {user_input}."
            return

        self.history.append(HumanMessage(content=user_input))
        
        try:
            # We must use app.stream with stream_mode="messages" to get token-by-token output
            # from the Anthropic LLM, while still allowing the graph to route to tools.
            full_content = ""
            for event, chunk in self.app.stream({"messages": self.history}, stream_mode="messages"):
                # If the agent node is generating a message chunk, yield it
                if event == "agent" and isinstance(chunk, AIMessage) and chunk.content:
                    # Sometimes chunk returns a dict or list for tool calls. 
                    # We only yield text if it's a string.
                    if isinstance(chunk.content, str):
                        full_content += chunk.content
                        yield chunk.content
                        
            # After the stream finishes, we retrieve the final state to append to history
            # The final state will include both the LLM's message AND any ToolMessages if it called tools
            final_state = self.app.invoke({"messages": self.history})
            # LangGraph returns all messages; update our local history completely
            self.history = final_state["messages"]
            logger.info("[Brain] Finished streaming response.")
            
        except Exception as e:
            logger.error(f"[Brain] Streaming Error: {e}", exc_info=True)
            yield "I am having trouble connecting. "
