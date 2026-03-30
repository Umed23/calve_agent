import logging
import os
from datetime import datetime, date
from supabase import create_client, Client
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class BookingBrain:
    """
    Core service layer: fetches available slots from Supabase,
    generates a Hindi voice message via OpenAI, optionally places
    a Twilio call, and logs the outcome to `call_logs`.
    """

    def __init__(self, use_twilio: bool = False):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set."
            )

        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if self.llm_provider == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY must be set if LLM_PROVIDER=gemini")
            self.openai = AsyncOpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.model_name = "gemini-2.5-flash"
        else:
            self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"

        self.use_twilio = use_twilio

        if use_twilio:
            try:
                from twilio.rest import Client as TwilioClient

                self.twilio = TwilioClient(
                    os.getenv("TWILIO_ACCOUNT_SID"),
                    os.getenv("TWILIO_AUTH_TOKEN"),
                )
            except ImportError:
                raise RuntimeError(
                    "USE_TWILIO=true but 'twilio' package is not installed."
                )

        logger.info(f"✅ BookingBrain initialized (Provider: {self.llm_provider.upper()}, Model: {self.model_name})")

    # ------------------------------------------------------------------
    # Supabase helpers
    # ------------------------------------------------------------------

    async def get_available_slots(self, doctor_id: str, target_date: str) -> list[str]:
        """
        Calls the Supabase RPC `get_available_slots` and returns a list
        of voice-friendly time strings in HH24:MI format, e.g. ['09:00', '10:30'].
        """
        try:
            logger.info(f"[{doctor_id}] Fetching available slots for date: {target_date}")
            response = self.supabase.rpc(
                "get_available_slots",
                {"p_doctor_id": doctor_id, "p_date": target_date},
            ).execute()

            if response.data:
                slots = [row["slot_time"] for row in response.data]
                logger.info(f"[{doctor_id}] Found {len(slots)} slots: {slots}")
                return slots
            logger.info(f"[{doctor_id}] No slots available for {target_date}")
            return []
        except Exception as e:
            logger.error(f"[BookingBrain] Error fetching slots: {e}", exc_info=True)
            return []

    async def log_call(
        self,
        patient_phone: str,
        doctor_id: str,
        message: str,
        success: bool,
    ) -> None:
        """Persists a call record to the `call_logs` table."""
        try:
            self.supabase.table("call_logs").insert(
                {
                    "clinic_id": os.getenv("CLINIC_ID", "clinic_001"),
                    "patient_phone": patient_phone,
                    "doctor_id": doctor_id,
                    "agent_response": message,
                    "success": success,
                    "language_code": os.getenv("LANGUAGE", "hi"),
                    "call_date": datetime.utcnow().isoformat(),
                }
            ).execute()
            logger.info(f"[Call Log] Successfully recorded call for {patient_phone} (Success: {success})")
        except Exception as e:
            # Logging failures should never crash the main flow
            logger.error(f"[BookingBrain] Call logging error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # AI message generation
    # ------------------------------------------------------------------

    async def generate_message(self, slots: list[str]) -> str:
        """
        Asks GPT-4o to produce a short, spoken-style Hindi message
        listing available appointment slots.
        """
        slots_str = ", ".join(slots) if slots else "उपलब्ध नहीं"
        clinic_name = os.getenv("CLINIC_NAME", "Sharma Clinic")

        system_prompt = (
            "You are a professional clinic receptionist for "
            f"{clinic_name} in India. "
            "Always respond ONLY in Hindi using Devanagari script. "
            "No English words, no markdown, no emojis. "
            "Keep responses under 50 words — they will be read aloud."
        )

        user_prompt = (
            f"बताएं कि निम्न समय पर अपॉइंटमेंट उपलब्ध है: {slots_str}. "
            "मरीज़ को विनम्रता से इन स्लॉट्स की जानकारी दें।"
        )

        response = await self.openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        generated_msg = response.choices[0].message.content.strip()
        logger.info(f"[BookingBrain] Generated slot message: {generated_msg}")
        return generated_msg

    # ------------------------------------------------------------------
    # Inbound speech handler (called by /process-speech webhook)
    # ------------------------------------------------------------------

    async def process_patient_speech(self, speech_text: str) -> str:
        """
        Takes the patient's transcribed speech from Twilio, sends it to
        GPT-4o acting as a Hindi receptionist, and returns a short spoken
        Hindi reply.  Called by main.py on every POST /process-speech.
        """
        clinic_name = os.getenv("CLINIC_NAME", "Sharma Clinic")

        system_prompt = (
            f"You are a warm, professional receptionist for {clinic_name} in India. "
            "Always respond ONLY in Hindi using Devanagari script. "
            "No English, no markdown, no emojis. "
            "Keep answers under 2 sentences — they will be spoken aloud. "
            "You can help with: booking appointments, checking availability, "
            "clinic timings, doctor information, and general queries. "
            "If you cannot help, politely say so and suggest calling back during clinic hours."
        )

        try:
            logger.info(f"[BookingBrain] Processing patient speech snippet: '{speech_text[:50]}...'")
            ai_resp = await self.openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": speech_text},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            reply = ai_resp.choices[0].message.content.strip()
            logger.info(f"[BookingBrain] AI generated reply: '{reply}'")
            return reply
        except Exception as e:
            err_str = str(e)
            logger.error(f"[BookingBrain] process_patient_speech error: {e}", exc_info=True)
            # Quota/billing exhausted — signal caller to hang up, no point retrying
            if "insufficient_quota" in err_str or "429" in err_str:
                return "HANGUP:क्षमा करें, अभी सेवा उपलब्ध नहीं है। कृपया कल पुनः कॉल करें। धन्यवाद।"
            return "क्षमा करें, मुझे एक तकनीकी समस्या हो रही है। कृपया थोड़ी देर बाद पुनः प्रयास करें।"

    # ------------------------------------------------------------------
    # Outbound call entry point
    # ------------------------------------------------------------------

    async def handle_call(
        self,
        patient_phone: str,
        doctor_id: str,
        preferred_date: str,
    ) -> dict:
        """
        Orchestrates the full booking call flow:
        1. Fetch available slots from Supabase
        2. Generate a Hindi AI message
        3. (Optional) Place a Twilio call
        4. Log the outcome
        Returns a dict compatible with BookingResponse.
        """
        try:
            logger.info(f"[BookingBrain.handle_call] Initiated outbound flow for {patient_phone} (Doctor: {doctor_id}, Date: {preferred_date})")
            
            # Step 1 — Fetch slots
            slots = await self.get_available_slots(doctor_id, preferred_date)

            if not slots:
                message = (
                    "क्षमा करें, इस तारीख को डॉक्टर के पास कोई समय उपलब्ध नहीं है। "
                    "कृपया कोई और तारीख चुनें।"
                )
                logger.info(f"[BookingBrain.handle_call] No slots found, aborting standard flow. Message: {message}")
                await self.log_call(patient_phone, doctor_id, message, False)
                return {"success": False, "message": message, "slots": []}

            # Step 2 — Generate AI message
            message = await self.generate_message(slots)

            # Step 3 — Optionally place Twilio call
            call_sid = None
            if self.use_twilio:
                logger.info(f"[Twilio] Starting outbound call dispatch for {patient_phone}...")
                try:
                    twiml = (
                        f'<Response><Say language="hi-IN">{message}</Say></Response>'
                    )
                    call = self.twilio.calls.create(
                        to=patient_phone,
                        from_=os.getenv("TWILIO_PHONE_NUMBER"),
                        twiml=twiml,
                    )
                    call_sid = call.sid
                    logger.info(f"[Twilio] ✅ Outbound call placed successfully: {call_sid}")
                except Exception as e:
                    logger.error(f"[Twilio] 🔴 Outbound call failed (non-fatal): {e}", exc_info=True)
            else:
                logger.debug("[Twilio] USE_TWILIO=false, skipping Twilio API call")

            # Step 4 — Log
            await self.log_call(patient_phone, doctor_id, message, True)

            return {
                "success": True,
                "message": message,
                "slots": slots,
                "call_sid": call_sid,
            }

        except Exception as e:
            logger.error(f"[BookingBrain] handle_call unexpected error: {e}", exc_info=True)
            await self.log_call(patient_phone, doctor_id, str(e), False)
            return {"success": False, "message": "Internal error", "error": str(e)}
