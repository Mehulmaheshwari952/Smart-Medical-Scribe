from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class MedicationItem(BaseModel):
    name: str = "unknown"
    dose: str = "unknown"
    frequency: str = "unknown"


class SubjectiveSection(BaseModel):
    chief_complaint: str = "unknown"
    symptoms: list[str] = Field(default_factory=list)
    hpi: str = "unknown"
    relevant_history: str = "unknown"


class ObjectiveSection(BaseModel):
    vitals: str = "unknown"
    exam_findings: list[str] = Field(default_factory=list)
    lab_results: list[str] = Field(default_factory=list)


class AssessmentSection(BaseModel):
    diagnoses: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class PlanSection(BaseModel):
    treatment: list[str] = Field(default_factory=list)
    medications: list[MedicationItem] = Field(default_factory=list)
    tests_ordered: list[str] = Field(default_factory=list)
    follow_up: str = "unknown"


class SOAPNote(BaseModel):
    subjective: SubjectiveSection = Field(default_factory=SubjectiveSection)
    objective: ObjectiveSection = Field(default_factory=ObjectiveSection)
    assessment: AssessmentSection = Field(default_factory=AssessmentSection)
    plan: PlanSection = Field(default_factory=PlanSection)
    retrieval_query: str = "unknown"
    status: Literal["recording", "draft", "approved"] = "draft"


class SuggestionItem(BaseModel):
    text: str
    rationale: str
    source_ids: list[str] = Field(min_length=1)


class GenerationOutput(BaseModel):
    suggestions: list[SuggestionItem] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    patient_id: str
    record_id: str
    record_type: Literal["note", "visit", "lab", "diagnosis", "allergy"]
    date: datetime
    section: str
    text: str
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    final_score: float = 0.0