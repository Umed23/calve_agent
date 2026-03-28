"""
CALVE Voice Agent — FastAPI Web Service for Twilio
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse, Gather

from config.settings import settings
from api.booking_brain import BookingBrain
from api.models import TriggerCallRequest, BookingResponse

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate env
settings.validate()

# Global
booking_brain: BookingBrain | None = None

# ----------------------------------------------------------------------
# TwiML Helpers
# ----------------------------------------------------------------------
def build_greeting_twiml() -> str:
    response = VoiceResponse()
    try:
        response.say(
            "नमस्ते। आप शर्मा क्लिनिक में बुला रहे हैं। कृपया अपनी समस्या बताएं।",
            language="hi-IN",
            voice="Polly.Aditi"
        )
        gather = Gather(
            input="speech",
            language="hi-IN",
            action="/process-speech",
            method="POST",
            speech_timeout="auto",
            timeout=5,
        )
        response.append(gather)
        logger.info("✅ Greeting TwiML generated successfully")
    except Exception as e:
        logger.error(f"❌ Error in greeting TwiML: {e}")
        response.say("Hello. Please try again later.")
        response.hangup()
    return str(response)


async def build_response_twiml(speech_result: str, call_sid: str) -> str:
    response = VoiceResponse()
    try:
        if not speech_result or not speech_result.strip():
            logger.warning(f"[{call_sid}] Empty speech result - reprompting")
            gather = Gather(
                input="speech",
                language="hi-IN",
                action="/process-speech",
                method="POST",
                speech_timeout="auto",
                timeout=5,
            )
            gather.say(
                "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया दोबारा बोलें।",
                language="hi-IN",
                voice="Polly.Aditi"
            )
            response.append(gather)
            return str(response)

        if booking_brain is None:
            raise RuntimeError("BookingBrain not initialized")

        logger.info(f"[{call_sid}] Processing: {speech_result[:100]}...")
        ai_response = await booking_brain.process_patient_speech(speech_result)

        if not ai_response:
            ai_response = "क्षमा करें, मुझे समझ नहीं आया। कृपया दोबारा बोलें।"

        response.say(
            ai_response,
            language="hi-IN",
            voice="Polly.Aditi"
        )

        # Continue conversation
        gather = Gather(
            input="speech",
            language="hi-IN",
            action="/process-speech",
            method="POST",
            speech_timeout="auto",
            timeout=5,
        )
        response.append(gather)

        logger.info(f"[{call_sid}] Response sent: {ai_response[:80]}...")
        return str(response)

    except Exception as e:
        logger.error(f"[{call_sid}] Error in build_response_twiml: {e}", exc_info=True)
        response.say("माफ कीजिए, कोई समस्या हुई। कृपया बाद में कोशिश करें।")
        response.hangup()
        return str(response)


# ----------------------------------------------------------------------
# Lifespan
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global booking_brain
    logger.info("🚀 Starting Voice Agent API …")
    try:
        booking_brain = BookingBrain(use_twilio=settings.USE_TWILIO)
        logger.info("✅ BookingBrain initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BookingBrain: {e}", exc_info=True)
    yield
    logger.info("🛑 Voice Agent API shutting down.")


# ----------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------
app = FastAPI(
    title="Voice Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes...
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "brain_initialized": booking_brain is not None,
    }


@app.post("/incoming-call")
async def incoming_call():
    twiml = build_greeting_twiml()
    return Response(content=twiml, media_type="application/xml")


@app.post("/process-speech")
async def process_speech(
    SpeechResult: str = Form(default=""),
    CallSid: str = Form(default=""),
):
    twiml = await build_response_twiml(SpeechResult, CallSid)
    return Response(content=twiml, media_type="application/xml")


# ... keep your /trigger-call as before with proper try/except ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)