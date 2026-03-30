import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # OpenAI / LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Clinic
    CLINIC_ID: str = os.getenv("CLINIC_ID", "clinic_001")
    CLINIC_NAME: str = os.getenv("CLINIC_NAME", "Sharma Clinic")

    # Voice
    LANGUAGE: str = os.getenv("LANGUAGE", "hi")
    STT_ENGINE: str = os.getenv("STT_ENGINE", "artpark")
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "facebook")
    # Twilio <Say> voice
    # Polly.Aditi = standard, works on ALL Twilio accounts (deprecated by AWS but still active on Twilio)
    # Polly.Kajal = neural quality but requires Twilio Neural Voice add-on (premium)
    TWILIO_VOICE: str = os.getenv("TWILIO_VOICE", "Polly.Aditi")

    # Twilio (Optional)
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    USE_TWILIO: bool = os.getenv("USE_TWILIO", "false").lower() == "true"

    # Server
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    def validate(self):
        required = [
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "OPENAI_API_KEY",
            "CLINIC_ID",
        ]
        missing = [var for var in required if not getattr(self, var, None)]
        if missing:
            raise RuntimeError(
                f"❌ Missing required environment variables: {missing}\n"
                "Please set them in your .env file or Render dashboard."
            )
        print("✅ Configuration loaded successfully")


settings = Settings()
