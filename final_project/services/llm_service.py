from __future__ import annotations

from typing import Any

from config import settings


class LLMService:
    def available(self) -> bool:
        return bool(settings.LLM_API_KEY)

    def profile_summary(self, person: dict[str, Any]) -> dict[str, Any]:
        if not self.available():
            return {
                "available": False,
                "summary": self.local_profile_summary(person),
                "label": "Generated locally from retrieved profile fields. LLM key is not configured.",
            }
        return {
            "available": False,
            "summary": self.local_profile_summary(person),
            "label": "LLM credentials are configured, but live LLM calls are disabled in this local prototype.",
        }

    def local_profile_summary(self, person: dict[str, Any]) -> str:
        facts = []
        for label, key in [
            ("profession", "Profession"),
            ("skills", "Skills"),
            ("experience", "Experience"),
            ("education", "Education"),
            ("projects", "Projects"),
            ("certifications", "Certifications"),
            ("bio", "Bio"),
        ]:
            value = person.get(key)
            if value:
                facts.append(f"{label}: {value}")
        if not facts:
            return "No technical profile information is available for this person yet."
        return "This profile contains stored information about " + "; ".join(facts) + "."
