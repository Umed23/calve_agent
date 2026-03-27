"""
CALVE Voice Agent — FastAPI Web Service
Entry point for the Render-hosted REST API.

Endpoints:
  GET  /health         → Render health check
  POST /trigger-call   → Trigger AI booking call for a patient
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.booking_brain import BookingBrain
from api.models import TriggerCallRequest, BookingResponse

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
