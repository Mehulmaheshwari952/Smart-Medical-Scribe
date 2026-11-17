import json
import re
from typing import Optional
from groq import Groq
from sqlalchemy.orm import Session
from app.config import settings
from app.models import StructuredFact
from app.schemas import GenerationOutput, RetrievedChunk, SOAPNote, SuggestionItem


class GenerationError(RuntimeError):
    """Raised when clinical suggestion generation fails."""


CLINICIAN_SUGGESTION_SYSTEM_PROMPT = """You are a clinical decision-support assistant reviewing a draft SOAP note alongside a patient's verified historical EHR records and structured facts (allergies, active medications, labs, chronic diagnoses).

RULES:
1. Provide concise, actionable clinical suggestions to enrich or cross-check the treatment plan (e.g., drug-allergy conflicts, overdue labs, renal dosing checks, prior treatment failures).
2. EVERY suggestion MUST cite at least one exact source ID from the provided [VALID SOURCE IDS] list (either a chunk_id or a fact:<id>).
3. NEVER invent source IDs. If no historical record supports a suggestion, do not include it.
4. Populate "flags" with concise high-priority safety alerts (e.g., "Allergy conflict: Penicillin") when applicable, or [] if none.
5. Output ONLY valid JSON matching:
{
  "suggestions": [
    {
      "text": "string",
      "rationale": "string",
      "source_ids": ["exact_id_from_context"]
    }
  ],
  "flags": ["string"]
}"""


PATIENT_SUMMARY_SYSTEM_PROMPT = """You are a compassionate clinical communicator translating a doctor's visit note into a clear, plain-language summary for the patient.

RULES:
1. Use simple, non-technical language (8th-grade reading level).
2. Cover: (a) What we discussed today, (b) Next steps / medications & instructions, and (c) When to follow up or seek care.
3. Do NOT invent instructions, doses, or diagnoses not present in the provided SOAP note. Keep it under 180 words."""


def _strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class ClinicalGenerationService:
    def __init__(self, client: Optional[Groq] = None, model: Optional[str] = None):
        self.client = client or Groq(api_key=settings.groq_api_key)
        self.model = model or settings.groq_llm_model

    def get_active_structured_facts(self, db: Session, patient_id: str) -> list[StructuredFact]:
        return (
            db.query(StructuredFact)
            .filter(StructuredFact.patient_id == patient_id, StructuredFact.active.is_(True))
            .all()
        )

    def _build_context_block(
        self,
        soap: SOAPNote,
        structured_facts: list[StructuredFact],
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[str, set[str]]:
        valid_ids: set[str] = set()
        lines: list[str] = ["=== CURRENT DRAFT SOAP NOTE ===", soap.model_dump_json(indent=2), ""]

        lines.append("=== STRUCTURED EHR FACTS (SQL) ===")
        if not structured_facts:
            lines.append("None recorded.")
        else:
            for fact in structured_facts:
                sid = f"fact:{fact.id}"
                valid_ids.add(sid)
                val_str = f"{fact.value} {fact.unit}".strip()
                lines.append(
                    f"- [ID: {sid}] ({fact.fact_type.upper()}) {fact.name}: {val_str} "
                    f"(recorded {fact.recorded_at.date().isoformat()})"
                )

        lines.append("\n=== RETRIEVED HISTORICAL EHR CHUNKS (RAG) ===")
        if not retrieved_chunks:
            lines.append("None retrieved.")
        else:
            for chunk in retrieved_chunks:
                valid_ids.add(chunk.chunk_id)
                lines.append(
                    f"- [ID: {chunk.chunk_id}] ({chunk.record_type.upper()} | section={chunk.section} | "
                    f"date={chunk.date.date().isoformat()}): {chunk.text}"
                )

        lines.append(f"\n=== VALID SOURCE IDS ===\n{sorted(valid_ids)}")
        return "\n".join(lines), valid_ids

    def generate_suggestions(
        self,
        db: Session,
        patient_id: str,
        soap: SOAPNote,
        retrieved_chunks: list[RetrievedChunk],
    ) -> GenerationOutput:
        structured_facts = self.get_active_structured_facts(db, patient_id)
        context_prompt, valid_ids = self._build_context_block(soap, structured_facts, retrieved_chunks)

        # If patient has zero historical facts and zero retrieved chunks, return empty safely
        if not valid_ids:
            return GenerationOutput(suggestions=[], flags=[])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLINICIAN_SUGGESTION_SYSTEM_PROMPT},
                {"role": "user", "content": context_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(_strip_code_fences(raw_content))
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Failed to parse suggestions JSON: {exc}") from exc

        raw_suggestions = parsed.get("suggestions") or []
        raw_flags = [str(f).strip() for f in (parsed.get("flags") or []) if str(f).strip()]

        # Deterministic guardrail: keep only verified source_ids; drop suggestions with 0 valid citations
        validated_suggestions: list[SuggestionItem] = []
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            cited = [str(sid).strip() for sid in (item.get("source_ids") or [])]
            verified_cited = [sid for sid in cited if sid in valid_ids]

            if text and rationale and verified_cited:
                validated_suggestions.append(
                    SuggestionItem(
                        text=text,
                        rationale=rationale,
                        source_ids=verified_cited,
                    )
                )

        return GenerationOutput(suggestions=validated_suggestions, flags=raw_flags)

    def generate_patient_summary(
        self,
        soap: SOAPNote,
        suggestions: Optional[list[SuggestionItem]] = None,
    ) -> str:
        """Optional second prompt producing a plain-language patient-facing visit summary."""
        notes_ctx = f"SOAP NOTE:\n{soap.model_dump_json(indent=2)}"
        if suggestions:
            sug_lines = "\n".join(f"- {s.text}" for s in suggestions)
            notes_ctx += f"\n\nCLINICIAN-APPROVED CARE NOTES:\n{sug_lines}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PATIENT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": notes_ctx},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()