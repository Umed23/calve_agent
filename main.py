"""
CALVE Voice Agent — FastAPI Web Service for Twilio
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse, Gather

from config.settings import settings
from api.booking_brain import BookingBrain
from api.models import TriggerCallRequest, BookingResponse

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate env on startup
settings.validate()

# ── Observability counters (in-memory, reset on restart) ───────────────
_start_time = time.monotonic()

class _Stats:
    calls_processed: int = 0
    calls_failed: int = 0
    calls_empty_speech: int = 0
    last_call_at: str | None = None

    @classmethod
    def record_success(cls):
        cls.calls_processed += 1
        cls.last_call_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def record_failure(cls):
        cls.calls_failed += 1
        cls.last_call_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def record_empty(cls):
        cls.calls_empty_speech += 1

stats = _Stats()

# ── Global ──────────────────────────────────────────────────────────────
booking_brain: BookingBrain | None = None

# Convenience: read configured voice from settings
TWILIO_VOICE = settings.TWILIO_VOICE          # default: Polly.Kajal
TWILIO_LANG = "hi-IN"


# -----------------------------------------------------------------------
# TwiML helpers
# -----------------------------------------------------------------------
def build_greeting_twiml() -> str:
    response = VoiceResponse()
    try:
        response.say(
            "नमस्ते। आप शर्मा क्लिनिक में बुला रहे हैं। कृपया अपनी समस्या बताएं।",
            language=TWILIO_LANG,
            voice=TWILIO_VOICE,
        )
        gather = Gather(
            input="speech",
            language=TWILIO_LANG,
            action="/process-speech",
            method="POST",
            speech_timeout="auto",
            timeout=5,
        )
        response.append(gather)
        # Fallback: no speech detected
        response.say(
            "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया पुनः प्रयास करें।",
            language=TWILIO_LANG,
            voice=TWILIO_VOICE,
        )
        response.redirect("/incoming-call")
        logger.info("✅ Greeting TwiML generated")
    except Exception as e:
        logger.error(f"❌ Error in greeting TwiML: {e}")
        response.say("Hello. Please try again later.")
        response.hangup()
    return str(response)


async def build_response_twiml(speech_result: str, call_sid: str) -> str:
    response = VoiceResponse()
    try:
        # ── Empty speech ──────────────────────────────────────────────
        if not speech_result or not speech_result.strip():
            logger.warning(f"[{call_sid}] Empty speech — reprompting")
            stats.record_empty()
            gather = Gather(
                input="speech",
                language=TWILIO_LANG,
                action="/process-speech",
                method="POST",
                speech_timeout="auto",
                timeout=5,
            )
            gather.say(
                "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया दोबारा बोलें।",
                language=TWILIO_LANG,
                voice=TWILIO_VOICE,
            )
            response.append(gather)
            return str(response)

        if booking_brain is None:
            raise RuntimeError("BookingBrain not initialized")

        # ── AI response ───────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"[{call_sid}] Processing: {speech_result[:100]}…")
        ai_response = await booking_brain.process_patient_speech(speech_result)
        latency = time.monotonic() - t0
        logger.info(f"[{call_sid}] GPT-4o latency: {latency:.2f}s")

        if not ai_response:
            ai_response = "क्षमा करें, मुझे समझ नहीं आया। कृपया दोबारा बोलें।"

        # ── HANGUP sentinel: quota/billing errors — end call cleanly ─────────
        if ai_response.startswith("HANGUP:"):
            message = ai_response[len("HANGUP:"):]
            response.say(message, language=TWILIO_LANG, voice=TWILIO_VOICE)
            response.hangup()
            stats.record_failure()
            logger.warning(f"[{call_sid}] 🔴 Hanging up due to service error ({latency:.2f}s)")
            return str(response)

        response.say(ai_response, language=TWILIO_LANG, voice=TWILIO_VOICE)

        # Continue conversation loop
        gather = Gather(
            input="speech",
            language=TWILIO_LANG,
            action="/process-speech",
            method="POST",
            speech_timeout="auto",
            timeout=5,
        )
        gather.say(
            "क्या आपका कोई और सवाल है?",
            language=TWILIO_LANG,
            voice=TWILIO_VOICE,
        )
        response.append(gather)

        # Graceful goodbye if caller goes silent
        response.say(
            "धन्यवाद! हमें आपसे बात करके अच्छा लगा। अलविदा!",
            language=TWILIO_LANG,
            voice=TWILIO_VOICE,
        )

        stats.record_success()
        logger.info(f"[{call_sid}] ✅ Response sent ({latency:.2f}s): {ai_response[:80]}…")
        return str(response)

    except Exception as e:
        stats.record_failure()
        logger.error(f"[{call_sid}] ❌ Error in build_response_twiml: {e}", exc_info=True)
        response.say(
            "माफ कीजिए, कोई समस्या हुई। कृपया बाद में कोशिश करें।",
            language=TWILIO_LANG,
            voice=TWILIO_VOICE,
        )
        response.hangup()
        return str(response)


# -----------------------------------------------------------------------
# Lifespan — startup / shutdown
# -----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global booking_brain
    logger.info("🚀 Starting CALVE Voice Agent …")
    try:
        booking_brain = BookingBrain(use_twilio=settings.USE_TWILIO)
        logger.info(f"✅ BookingBrain ready  |  TTS voice: {TWILIO_VOICE}  |  STT: {settings.STT_ENGINE}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BookingBrain: {e}", exc_info=True)
    yield
    logger.info("🛑 CALVE Voice Agent shutting down.")


# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------
app = FastAPI(title="CALVE Voice Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics (/metrics) ──────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge

    # Custom metrics
    CALLS_TOTAL = Counter(
        "calve_calls_total",
        "Total processed speech turns",
        ["status"],          # labels: success | failure | empty_speech
    )
    AI_LATENCY = Histogram(
        "calve_ai_response_seconds",
        "GPT-4o response latency per speech turn",
        buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    )
    BRAIN_UP = Gauge("calve_brain_initialized", "1 if BookingBrain is ready, else 0")

    Instrumentator().instrument(app).expose(app)   # mounts GET /metrics
    logger.info("📊 Prometheus /metrics endpoint enabled")
    PROMETHEUS_ENABLED = True
except ImportError:
    logger.warning("⚠️  prometheus-fastapi-instrumentator not installed — /metrics disabled")
    PROMETHEUS_ENABLED = False


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Rich health check — use this to see exactly what the agent is doing.
    Grafana / UptimeRobot / Render health checks all hit this endpoint.
    """
    uptime = time.monotonic() - _start_time
    brain_ok = booking_brain is not None

    # Update Prometheus gauge if available
    if PROMETHEUS_ENABLED:
        BRAIN_UP.set(1 if brain_ok else 0)

    return {
        # ── Core ─────────────────────────────────────────────────────
        "status": "ok" if brain_ok else "degraded",
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 1),
        # ── Configuration ─────────────────────────────────────────────
        "clinic": settings.CLINIC_NAME,
        "tts_engine": settings.TTS_ENGINE,          # facebook = local NeuralMouth (local_agent only)
        "tts_voice_twilio": TWILIO_VOICE,           # voice used in Twilio <Say> tags
        "stt_engine": settings.STT_ENGINE,
        "language": settings.LANGUAGE,
        # ── Brain ────────────────────────────────────────────────────
        "brain_initialized": brain_ok,
        "use_twilio": settings.USE_TWILIO,
        "prometheus_metrics": PROMETHEUS_ENABLED,
        # ── Call stats (reset on each deploy) ─────────────────────────
        "calls_processed": stats.calls_processed,
        "calls_failed": stats.calls_failed,
        "calls_empty_speech": stats.calls_empty_speech,
        "last_call_at": stats.last_call_at,
        # ── Timestamp ────────────────────────────────────────────────
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/incoming-call")
async def incoming_call():
    """Twilio calls this when a patient first dials the number."""
    twiml = build_greeting_twiml()
    return Response(content=twiml, media_type="application/xml")


@app.get("/process-speech")
async def process_speech_get():
    """Twilio validates webhook URLs with a GET — return 200 to pass."""
    return Response(content="OK", media_type="text/plain")


@app.post("/process-speech")
async def process_speech(
    SpeechResult: str = Form(default=""),
    CallSid: str = Form(default=""),
):
    """Main speech processing webhook — called by Twilio after patient speaks."""
    is_empty = not SpeechResult or not SpeechResult.strip()

    if PROMETHEUS_ENABLED and is_empty:
        CALLS_TOTAL.labels(status="empty_speech").inc()

    failed_before = stats.calls_failed
    t0 = time.monotonic()
    twiml = await build_response_twiml(SpeechResult, CallSid)
    elapsed = time.monotonic() - t0

    if PROMETHEUS_ENABLED and not is_empty:
        AI_LATENCY.observe(elapsed)
        if stats.calls_failed > failed_before:
            CALLS_TOTAL.labels(status="failure").inc()
        else:
            CALLS_TOTAL.labels(status="success").inc()

    return Response(content=twiml, media_type="application/xml")


@app.post("/trigger-call", response_model=BookingResponse)
async def trigger_call(req: TriggerCallRequest):
    """Outbound call trigger — places a Twilio call to a patient."""
    if booking_brain is None:
        return BookingResponse(success=False, message="BookingBrain not initialized", slots=[])
    try:
        result = await booking_brain.handle_call(
            patient_phone=req.patient_phone,
            doctor_id=req.doctor_id,
            preferred_date=req.preferred_date,
        )
        return BookingResponse(**result)
    except Exception as e:
        logger.error(f"[trigger-call] Error: {e}", exc_info=True)
        return BookingResponse(success=False, message=f"Error: {e}", slots=[])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)