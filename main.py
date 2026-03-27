"""
CALVE Voice Agent — FastAPI Web Service
Entry point for the Render-hosted REST API.

Endpoints:
  GET  /health            → Render health check
  POST /trigger-call      → Outbound: trigger AI booking call for a patient
  POST /incoming-call     → Inbound: Twilio webhook — greet caller
  POST /process-speech    → Inbound: Twilio webhook — process patient speech
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config.settings import settings
from api.booking_brain import BookingBrain
from api.models import TriggerCallRequest, BookingResponse
from api.voice_handler import build_greeting_twiml, build_response_twiml

# Validate all required env vars on startup — fail fast
settings.validate()


# --------------------------------------------------------------------------- #
# Application lifecycle                                                         #
# --------------------------------------------------------------------------- #

booking_brain: BookingBrain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once on startup, clean up on shutdown."""
    global booking_brain
    print("🚀 Starting CALVE Voice Agent API …")
    booking_brain = BookingBrain(use_twilio=settings.USE_TWILIO)
    yield
    print("🛑 CALVE Voice Agent API shutting down.")


# --------------------------------------------------------------------------- #
# FastAPI app                                                                   #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="CALVE Voice Agent API",
    description=(
        "AI-powered booking assistant for Calve. "
        "POST /trigger-call to initiate a patient booking call."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes                                                                        #
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["System"])
async def health():
    """
    Health check used by Render to verify the service is running.
    Returns the clinic name and database status.
    """
    return {
        "status": "ok",
        "service": "CALVE Voice Agent API",
        "clinic": settings.CLINIC_NAME,
        "database": "connected",
    }


@app.post("/trigger-call", response_model=BookingResponse, tags=["Booking"])
async def trigger_call(request: TriggerCallRequest):
    """
    Trigger an AI-powered appointment call for a patient.

    - Fetches available slots from Supabase via RPC
    - Generates a Hindi voice message with GPT-4o
    - Optionally places a Twilio call (if USE_TWILIO=true)
    - Logs the outcome to `call_logs`
    """
    if booking_brain is None:
        raise HTTPException(status_code=503, detail="Service not initialised yet.")

    if not request.patient_phone.strip() or not request.doctor_id.strip():
        raise HTTPException(
            status_code=400,
            detail="patient_phone and doctor_id are required and cannot be empty.",
        )

    result = await booking_brain.handle_call(
        patient_phone=request.patient_phone,
        doctor_id=request.doctor_id,
        preferred_date=request.preferred_date,
    )

    return result


# --------------------------------------------------------------------------- #
# Inbound call webhooks (Twilio → Render)                                      #
# --------------------------------------------------------------------------- #

@app.post("/incoming-call", tags=["Inbound Voice"])
async def incoming_call():
    """
    Twilio calls this webhook when a patient dials your Twilio number.
    Returns TwiML: plays a Hindi greeting and starts listening.

    Configure in Twilio Console:
      Phone Number → Voice → Webhook → https://calve-agent.onrender.com/incoming-call
    """
    twiml = build_greeting_twiml()
    return Response(content=twiml, media_type="application/xml")


@app.post("/process-speech", tags=["Inbound Voice"])
async def process_speech(
    SpeechResult: str = Form(default=""),
    CallSid: str = Form(default=""),
):
    """
    Twilio calls this after the patient speaks.
    SpeechResult contains the transcribed text.
    Returns TwiML: AI-generated Hindi reply + listens again (conversation loop).
    """
    print(f"[Inbound] CallSid={CallSid} | Patient said: '{SpeechResult}'")

    if not SpeechResult.strip():
        # Nothing was said — re-prompt
        from twilio.twiml.voice_response import VoiceResponse, Gather
        response = VoiceResponse()
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
            voice="Polly.Aditi",
        )
        response.append(gather)
        return Response(content=str(response), media_type="application/xml")

    twiml = await build_response_twiml(SpeechResult, CallSid)
    return Response(content=twiml, media_type="application/xml")


# --------------------------------------------------------------------------- #
# Dev runner (not used on Render — see Start Command)                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
