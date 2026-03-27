"""
api/voice_handler.py — Inbound Call TwiML Handler

Flow:
  1. Patient dials Twilio number
  2. Twilio hits POST /incoming-call → returns greeting + <Gather>
  3. Patient speaks
  4. Twilio hits POST /process-speech with SpeechResult
  5. AI generates Hindi response → returns <Say> + <Gather> (loop)
"""

import os
from openai import AsyncOpenAI
from twilio.twiml.voice_response import VoiceResponse, Gather


# Shared OpenAI client
_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLINIC_NAME = os.getenv("CLINIC_NAME", "Sharma Clinic")
LANGUAGE = "hi-IN"        # Twilio language code for Hindi (India)
VOICE = "Polly.Aditi"     # Best Hindi voice on Twilio (Amazon Polly)


def build_greeting_twiml() -> str:
    """
    Returns TwiML for the opening greeting.
    Twilio calls this when the patient first dials in.
    """
    response = VoiceResponse()

    response.say(
        f"नमस्ते! {CLINIC_NAME} में आपका स्वागत है। "
        "मैं आपकी अपॉइंटमेंट में सहायता करने के लिए यहाँ हूँ। "
        "कृपया बोलें — आप क्या जानना चाहते हैं?",
        language=LANGUAGE,
        voice=VOICE,
    )

    gather = Gather(
        input="speech",
        language=LANGUAGE,
        action="/process-speech",
        method="POST",
        speech_timeout="auto",
        timeout=5,
    )
    response.append(gather)

    # Fallback if no speech detected
    response.say(
        "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया पुनः प्रयास करें।",
        language=LANGUAGE,
        voice=VOICE,
    )
    response.redirect("/incoming-call")

    return str(response)


async def build_response_twiml(speech_text: str, call_sid: str) -> str:
    """
    Takes the patient's spoken text, sends it to GPT-4o,
    and returns TwiML with the AI's Hindi reply + next Gather.
    """
    system_prompt = (
        f"You are a warm, professional receptionist for {CLINIC_NAME} in India. "
        "Always respond ONLY in Hindi using Devanagari script. "
        "No English, no markdown, no emojis. "
        "Keep answers under 2 sentences — they will be spoken aloud. "
        "You can help with: booking appointments, checking availability, "
        "clinic timings, doctor information, and general queries. "
        "If you cannot help, politely say so and suggest calling back during clinic hours."
    )

    try:
        ai_response = await _openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": speech_text},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        reply = ai_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[VoiceHandler] OpenAI error: {e}")
        reply = "क्षमा करें, मुझे एक तकनीकी समस्या हो रही है। कृपया थोड़ी देर बाद पुनः प्रयास करें।"

    response = VoiceResponse()

    # Speak the AI reply
    response.say(reply, language=LANGUAGE, voice=VOICE)

    # Listen again (conversation loop)
    gather = Gather(
        input="speech",
        language=LANGUAGE,
        action="/process-speech",
        method="POST",
        speech_timeout="auto",
        timeout=5,
    )
    gather.say(
        "क्या आपका कोई और सवाल है?",
        language=LANGUAGE,
        voice=VOICE,
    )
    response.append(gather)

    # Goodbye if no response
    response.say(
        "धन्यवाद! हमें आपसे बात करके अच्छा लगा। अलविदा!",
        language=LANGUAGE,
        voice=VOICE,
    )

    return str(response)
