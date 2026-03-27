import os
from datetime import datetime, date
from supabase import create_client, Client
from openai import AsyncOpenAI


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
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

        print("✅ BookingBrain ready")

    # ------------------------------------------------------------------
    # Supabase helpers
    # ------------------------------------------------------------------

    async def get_available_slots(self, doctor_id: str, target_date: str) -> list[str]:
        """
        Calls the Supabase RPC `get_available_slots` and returns a list
        of voice-friendly time strings in HH24:MI format, e.g. ['09:00', '10:30'].
        """
        try:
            response = self.supabase.rpc(
                "get_available_slots",
                {"p_doctor_id": doctor_id, "p_date": target_date},
            ).execute()

            if response.data:
                # RPC returns rows; extract the slot_time column
                return [row["slot_time"] for row in response.data]
            return []
        except Exception as e:
            print(f"[BookingBrain] Error fetching slots: {e}")
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
        except Exception as e:
            # Logging failures should never crash the main flow
            print(f"[BookingBrain] Call logging error: {e}")

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Main entry point
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
            # Step 1 — Fetch slots
            slots = await self.get_available_slots(doctor_id, preferred_date)

            if not slots:
                message = (
                    "क्षमा करें, इस तारीख को डॉक्टर के पास कोई समय उपलब्ध नहीं है। "
                    "कृपया कोई और तारीख चुनें।"
                )
                await self.log_call(patient_phone, doctor_id, message, False)
                return {"success": False, "message": message, "slots": []}

            # Step 2 — Generate AI message
            message = await self.generate_message(slots)

            # Step 3 — Optionally place Twilio call
            call_sid = None
            if self.use_twilio:
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
                    print(f"[Twilio] Call placed: {call_sid}")
                except Exception as e:
                    print(f"[Twilio] Call failed (non-fatal): {e}")

            # Step 4 — Log
            await self.log_call(patient_phone, doctor_id, message, True)

            return {
                "success": True,
                "message": message,
                "slots": slots,
                "call_sid": call_sid,
            }

        except Exception as e:
            print(f"[BookingBrain] handle_call error: {e}")
            await self.log_call(patient_phone, doctor_id, str(e), False)
            return {"success": False, "message": "Internal error", "error": str(e)}
