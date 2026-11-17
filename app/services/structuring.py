import json
import re
from typing import Any, Optional
from groq import Groq
from pydantic import ValidationError
from app.config import settings
from app.schemas import SOAPNote


class StructuringError(RuntimeError):
    """Raised when the LLM fails to return valid SOAP JSON."""


SYSTEM_PROMPT = """You are an expert clinical documentation assistant converting a raw, un-diarized doctor-patient consultation transcript into a structured SOAP note in strict JSON format.

RULES:
1. Infer doctor vs. patient speakers from conversational context (questions, exam findings, prescriptions = clinician; reported symptoms, history = patient).
2. NEVER guess or fabricate clinical details. If a scalar string field (chief_complaint, hpi, relevant_history, vitals, follow_up, medication dose/frequency) is not explicitly mentioned, output "unknown". If a list field (symptoms, exam_findings, lab_results, diagnoses, conditions, treatment, medications, tests_ordered) has no items mentioned, output [].
3. Output ONLY valid JSON matching the exact schema below, with "status": "draft".
4. Generate a concise, keyword-rich "retrieval_query" (5-12 words) capturing the patient's active conditions, symptoms, and proposed medications/tests to search their historical EHR. If nothing clinical is present, set "retrieval_query" to "unknown".

REQUIRED JSON SCHEMA:
{
  "subjective": {
    "chief_complaint": "string or 'unknown'",
    "symptoms": ["string"],
    "hpi": "string or 'unknown'",
    "relevant_history": "string or 'unknown'"
  },
  "objective": {
    "vitals": "string or 'unknown'",
    "exam_findings": ["string"],
    "lab_results": ["string"]
  },
  "assessment": {
    "diagnoses": ["string"],
    "conditions": ["string"]
  },
  "plan": {
    "treatment": ["string"],
    "medications": [{"name": "string", "dose": "string or 'unknown'", "frequency": "string or 'unknown'"}],
    "tests_ordered": ["string"],
    "follow_up": "string or 'unknown'"
  },
  "retrieval_query": "string",
  "status": "draft"
}"""


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "TRANSCRIPT:\n"
            "What brings you in today? I've had this burning pain when I urinate for the past two days, "
            "and I'm going every hour. Any fever, chills, or back pain? No fever, no flank pain. I had a UTI "
            "once last year. Okay, your vitals look fine, temp is 98.4, blood pressure 118 over 74. Urine dipstick "
            "is positive for nitrites and leukocyte esterase. This looks like an uncomplicated urinary tract infection. "
            "I'll send nitrofurantoin 100 mg twice a day for five days. Drink plenty of fluids and let us know if "
            "symptoms don't improve in 48 hours."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "subjective": {
                    "chief_complaint": "Dysuria and urinary frequency",
                    "symptoms": ["burning urination", "urinary frequency"],
                    "hpi": "2-day history of dysuria and hourly urinary frequency; denies fever, chills, or flank pain.",
                    "relevant_history": "History of urinary tract infection last year",
                },
                "objective": {
                    "vitals": "Temp 98.4 F, BP 118/74 mmHg",
                    "exam_findings": [],
                    "lab_results": ["Urine dipstick positive for nitrites and leukocyte esterase"],
                },
                "assessment": {
                    "diagnoses": ["Uncomplicated urinary tract infection"],
                    "conditions": [],
                },
                "plan": {
                    "treatment": ["Increase oral fluid intake"],
                    "medications": [
                        {
                            "name": "Nitrofurantoin",
                            "dose": "100 mg",
                            "frequency": "twice daily for 5 days",
                        }
                    ],
                    "tests_ordered": [],
                    "follow_up": "Contact clinic if symptoms do not improve within 48 hours",
                },
                "retrieval_query": "urinary tract infection nitrofurantoin antibiotic allergy renal function",
                "status": "draft",
            }
        ),
    },
    {
        "role": "user",
        "content": (
            "TRANSCRIPT:\n"
            "Hey, my right knee has been aching since I twisted it playing soccer on Sunday. Does it swell or lock up? "
            "A little swollen yesterday, no locking. Let me take a look—mild joint effusion, tenderness along the medial "
            "joint line, ligaments feel intact. Probably a mild medial meniscus or MCL sprain. Rest, ice it for 20 minutes "
            "a few times a day, and take ibuprofen if needed. Let's get an X-ray today to rule out fracture and check back in two weeks."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "subjective": {
                    "chief_complaint": "Right knee pain",
                    "symptoms": ["right knee ache", "mild swelling"],
                    "hpi": "Right knee pain after twisting injury while playing soccer on Sunday; mild swelling yesterday, denies locking.",
                    "relevant_history": "unknown",
                },
                "objective": {
                    "vitals": "unknown",
                    "exam_findings": [
                        "Mild right knee joint effusion",
                        "Medial joint line tenderness",
                        "Intact knee ligaments",
                    ],
                    "lab_results": [],
                },
                "assessment": {
                    "diagnoses": ["Suspected mild medial meniscus or MCL sprain"],
                    "conditions": [],
                },
                "plan": {
                    "treatment": ["Rest", "Ice 20 minutes several times daily"],
                    "medications": [
                        {
                            "name": "Ibuprofen",
                            "dose": "unknown",
                            "frequency": "as needed",
                        }
                    ],
                    "tests_ordered": ["Right knee X-ray"],
                    "follow_up": "2 weeks",
                },
                "retrieval_query": "right knee injury NSAID ibuprofen gastrointestinal renal contraindications",
                "status": "draft",
            }
        ),
    },
]


def _strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _sanitize_nulls(data: Any) -> Any:
    """Replaces accidental JSON null scalars with 'unknown' so Pydantic validation stays strict."""
    if isinstance(data, dict):
        return {
            k: ("unknown" if v is None and not k.endswith("s") else _sanitize_nulls(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_sanitize_nulls(item) for item in data if item is not None]
    return data


class SOAPStructuringService:
    def __init__(self, client: Optional[Groq] = None, model: Optional[str] = None):
        self.client = client or Groq(api_key=settings.groq_api_key)
        self.model = model or settings.groq_llm_model

    def generate_soap(self, transcript: str) -> SOAPNote:
        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            return SOAPNote(status="draft")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": f"TRANSCRIPT:\n{cleaned_transcript}"},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(_strip_code_fences(raw_content))
            sanitized = _sanitize_nulls(parsed)
            sanitized["status"] = "draft"
            return SOAPNote.model_validate(sanitized)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuringError(f"Failed to validate SOAP JSON from LLM: {exc}") from exc