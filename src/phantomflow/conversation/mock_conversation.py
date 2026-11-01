"""Mock AI Conversation and Task Engine (Member 3 stub).

Conforms to IConversationEngine protocol.
Demonstrates SwitchSync Hinglish intent extraction, slot corrections,
Fast Lane deterministic processing, and Smart Path token streaming.
"""

import asyncio
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, Optional

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, UserUtterance
from phantomflow.contracts.conversation_contracts import IConversationEngine, IntentResult, TaskResponse

logger = logging.getLogger("phantomflow.conversation")


class MockConversationEngine(IConversationEngine):
    """Mock implementation of Member 3's AI Conversation & Task Engine."""

    def __init__(
        self,
        simulated_llm_delay_ms: float = 20.0,
        enable_live_llm: bool = False,
    ):
        self.simulated_llm_delay_ms = simulated_llm_delay_ms
        self.enable_live_llm = enable_live_llm or os.getenv("ENABLE_LIVE_LLM", "false").lower() in ("true", "1", "yes")

    async def resolve_intent(
        self,
        utterance: UserUtterance,
        session_context: Dict[str, Any],
    ) -> IntentResult:
        """Parse utterance for Railway, Healthcare, or General intents with SwitchSync."""
        text = utterance.text.strip().lower()
        existing_slots = session_context.get("slots", {})

        # 1. Language switch request (Turn 3)
        # e.g., "Now please explain everything in English", "Switch to Hindi", "Speak in English"
        if "english" in text and any(w in text for word in ["explain", "switch", "speak", "please", "now"] for w in [word]):
            return IntentResult(
                domain=session_context.get("current_domain") or Domain.RAILWAY,
                intent_name="switch_language",
                confidence=0.99,
                extracted_slots={"preferred_language": "English"},
                is_correction=False,
                preferred_routing=RoutingPath.FAST,
                detected_language=LanguageCode.EN,
            )

        # 2. Correction detection (USP 1: SwitchSync)
        # e.g., "Actually Saturday. And morning mein.", "Wait! I want to change my destination."
        if any(word in text for word in ["actually", "wait", "nahi", "change", "instead"]):
            extracted = {}
            # Check for day correction
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                if day in text:
                    extracted["travel_date"] = day.capitalize()
            # Check for time of day preference
            for tod in ["morning", "evening", "afternoon", "night"]:
                if tod in text:
                    extracted["time_preference"] = tod.capitalize()
            # Check for destination change
            if "destination" in text or "hyderabad" in text or "mumbai" in text or "chennai" in text:
                if "hyderabad" in text:
                    extracted["destination"] = "Hyderabad"
                elif "mumbai" in text:
                    extracted["destination"] = "Mumbai"
                elif "chennai" in text:
                    extracted["destination"] = "Chennai"
                elif "bengaluru" in text or "bangalore" in text:
                    extracted["destination"] = "Bengaluru"
                else:
                    extracted["destination"] = "Hyderabad"

            if extracted:
                return IntentResult(
                    domain=session_context.get("current_domain") or Domain.RAILWAY,
                    intent_name="update_slots",
                    confidence=0.98,
                    extracted_slots=extracted,
                    is_correction=True,
                    preferred_routing=RoutingPath.FAST,
                    detected_language=utterance.language,
                )

        # 3. Railway: PNR status query
        pnr_match = re.search(r"\b\d{10}\b", text)
        if "pnr" in text or pnr_match:
            pnr = pnr_match.group(0) if pnr_match else "1234567890"
            return IntentResult(
                domain=Domain.RAILWAY,
                intent_name="check_pnr",
                confidence=0.99,
                extracted_slots={"pnr": pnr},
                is_correction=False,
                preferred_routing=RoutingPath.FAST,
                detected_language=utterance.language,
            )

        # 4. Railway: Train search / booking
        if any(word in text for word in ["train", "railway", "delhi", "mumbai", "bengaluru", "bangalore", "chennai", "hyderabad", "saturday", "sunday", "friday", "gaadi", "jaana"]):
            slots = {}
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                if day in text:
                    slots["travel_date"] = day.capitalize()
            for tod in ["morning", "evening", "afternoon", "night"]:
                if tod in text:
                    slots["time_preference"] = tod.capitalize()

            if "delhi" in text:
                slots["origin"] = "New Delhi"
            elif "bengaluru" in text or "bangalore" in text:
                slots["origin"] = "Bengaluru"

            if "mumbai" in text:
                slots["destination"] = "Mumbai"
            elif "chennai" in text:
                slots["destination"] = "Chennai"
            elif "hyderabad" in text:
                slots["destination"] = "Hyderabad"

            return IntentResult(
                domain=Domain.RAILWAY,
                intent_name="search_train",
                confidence=0.95,
                extracted_slots=slots,
                is_correction=False,
                preferred_routing=RoutingPath.FAST,
                detected_language=utterance.language,
            )

        # 4. Complex / Ambiguous query (routed to Smart Path)
        if any(word in text for word in ["explain", "why", "symptoms", "compare", "suggest", "detail"]):
            return IntentResult(
                domain=Domain.HEALTHCARE,
                intent_name="medical_consultation",
                confidence=0.85,
                extracted_slots={},
                is_correction=False,
                preferred_routing=RoutingPath.SMART,
                detected_language=utterance.language,
            )

        # 5. Doctor Appointment
        if any(word in text for word in ["doctor", "appointment", "clinic", "dr.", "specialist", "dawai"]):
            slots = {}
            if "cardiologist" in text or "dil" in text:
                slots["specialty"] = "Cardiologist"
            elif "sharma" in text:
                slots["doctor_name"] = "Dr. Sharma"

            return IntentResult(
                domain=Domain.HEALTHCARE,
                intent_name="book_appointment",
                confidence=0.92,
                extracted_slots=slots,
                is_correction=False,
                preferred_routing=RoutingPath.FAST,
                detected_language=utterance.language,
            )

        # Default fallback intent
        return IntentResult(
            domain=Domain.GENERAL,
            intent_name="general_query",
            confidence=0.75,
            extracted_slots={},
            is_correction=False,
            preferred_routing=RoutingPath.FAST,
            detected_language=utterance.language,
        )

    async def execute_fast_lane(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> Optional[TaskResponse]:
        """Execute deterministic Python responses for standard requests (USP 3: Fast Lane)."""
        slots = session_context.get("slots", {})

        # Handle correction / slot updates (SwitchSync)
        if intent.is_correction and intent.intent_name in ("update_date", "update_slots"):
            new_date = intent.extracted_slots.get("travel_date") or slots.get("travel_date", "selected date")
            new_time = intent.extracted_slots.get("time_preference") or slots.get("time_preference", "")
            origin = intent.extracted_slots.get("origin") or slots.get("origin", "source")
            dest = intent.extracted_slots.get("destination") or slots.get("destination", "destination")
            time_suffix = f" {new_time}" if new_time else ""

            if "destination" in intent.extracted_slots:
                text = f"Theek hai, destination update karke {dest} kar diya gaya hai. {origin} se {dest} ke liye{time_suffix} details check kar raha hoon."
            else:
                text = f"Theek hai, travel date update karke {new_date}{time_suffix} kar di gayi hai. {origin} se {dest} ke liye details check kar raha hoon."

            return TaskResponse(
                text=text,
                slots_to_update=intent.extracted_slots,
                domain=intent.domain,
                is_complete=True,
            )

        # Handle Language Switch (Turn 3)
        if intent.intent_name == "switch_language":
            origin = slots.get("origin", "Bengaluru")
            dest = slots.get("destination", "Chennai")
            date = slots.get("travel_date", "Saturday")
            time_pref = slots.get("time_preference", "morning")
            time_str = f" {time_pref}" if time_pref else ""
            return TaskResponse(
                text=f"Certainly! I have switched to English. Your journey is confirmed from {origin} to {dest} on {date}{time_str}. Shall I proceed with the reservation?",
                slots_to_update=intent.extracted_slots,
                domain=intent.domain,
                is_complete=True,
            )

        # Handle PNR query
        if intent.intent_name == "check_pnr":
            pnr = intent.extracted_slots.get("pnr", "1234567890")
            return TaskResponse(
                text=f"Aapka PNR {pnr} confirmed hai. Coach B2, Seat number 45. Train samay par hai.",
                slots_to_update={"pnr": pnr, "status": "CONFIRMED"},
                domain=Domain.RAILWAY,
                is_complete=True,
            )

        # Handle Train Search
        if intent.intent_name == "search_train":
            date = intent.extracted_slots.get("travel_date") or slots.get("travel_date", "Saturday")
            origin = intent.extracted_slots.get("origin") or slots.get("origin", "New Delhi")
            dest = intent.extracted_slots.get("destination") or slots.get("destination", "Mumbai")
            train_name = "Shatabdi Express" if (origin == "Bengaluru" or dest == "Chennai") else "12952 Rajdhani Express"
            return TaskResponse(
                text=f"{date} ko {origin} se {dest} ke liye {train_name} uplabdh hai. Kya aap booking aage badhana chahte hain?",
                slots_to_update=intent.extracted_slots,
                domain=Domain.RAILWAY,
                is_complete=True,
            )

        # Handle Doctor Appointment
        if intent.intent_name == "book_appointment":
            return TaskResponse(
                text="Dr. Sharma ke saath kal subah 10:30 AM ka appointment slot available hai. Kya main confirm kar doon?",
                slots_to_update={"doctor_name": "Dr. Sharma", "appointment_slot": "10:30 AM"},
                domain=Domain.HEALTHCARE,
                is_complete=True,
            )

        # Return None to trigger Smart Path fallback if needed
        return None

    async def execute_smart_path_stream(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream LLM tokens (using ultra-fast Groq if enabled, otherwise simulated stream)."""
        api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

        if self.enable_live_llm and api_key and not api_key.startswith("your_"):
            try:
                import aiohttp
                import json

                messages = [
                    {
                        "role": "system",
                        "content": "You are PhantomFlow, a helpful, real-time voice assistant in India. Respond concisely in 1-2 short sentences.",
                    },
                    {
                        "role": "user",
                        "content": intent.extracted_slots.get("query") or intent.intent_name or "Help with query",
                    },
                ]
                async with aiohttp.ClientSession() as http_sess:
                    async with http_sess.post(
                        f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, "stream": True, "max_tokens": 80},
                        timeout=aiohttp.ClientTimeout(total=4.0),
                    ) as resp:
                        if resp.status == 200:
                            async for raw_line in resp.content:
                                line = raw_line.decode("utf-8").strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    chunk_data = json.loads(line[6:])
                                    delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                            return
            except Exception as exc:
                logger.warning("Live Groq streaming failed: %s. Using simulated tokens.", exc)

        tokens = [
            "Aapke ", "symptoms ", "ke ", "adhaar ", "par, ",
            "aapko ", "turant ", "ek ", "General Physician ",
            "ya ", "Cardiologist ", "se ", "paramarsh ", "lena ", "chahiye."
        ]
        for token in tokens:
            if self.simulated_llm_delay_ms > 0:
                await asyncio.sleep(self.simulated_llm_delay_ms / 1000.0)
            yield token
