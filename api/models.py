from pydantic import BaseModel
from typing import Optional


class TriggerCallRequest(BaseModel):
    patient_phone: str
    doctor_id: str
    preferred_date: str = "2026-03-30"


class BookingResponse(BaseModel):
    success: bool
    message: str
    slots: Optional[list] = None
    appointment_id: Optional[str] = None
    call_sid: Optional[str] = None
    error: Optional[str] = None
