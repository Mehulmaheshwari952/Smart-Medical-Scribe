from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db, init_db
from app.models import Consultation, EHRRecord, Patient, StructuredFact
from app.schemas import SOAPNote, SuggestionItem
from app.services.generation import ClinicalGenerationService, GenerationError
from app.services.ingestion import EHRIngestionService
from app.services.retrieval import HybridRetrievalService
from app.services.stt import AudioValidationError, GroqSTTService, append_transcript_chunk
from app.services.structuring import SOAPStructuringService, StructuringError
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.services.similar_cases import SimilarCaseInput, SimilarCaseResult, SimilarCasesService 

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ServiceContainer:
    stt: GroqSTTService
    structuring: SOAPStructuringService
    ingestion: EHRIngestionService
    retrieval: HybridRetrievalService
    generation: ClinicalGenerationService
    similar_cases: SimilarCasesService


def create_default_services() -> ServiceContainer:
    ingestion = EHRIngestionService()
    return ServiceContainer(
        stt=GroqSTTService(),
        structuring=SOAPStructuringService(),
        ingestion=ingestion,
        retrieval=HybridRetrievalService(ingestion_service=ingestion),
        generation=ClinicalGenerationService(),
        similar_cases=SimilarCasesService(chroma_client=ingestion.chroma_client),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not hasattr(app.state, "services"):
        app.state.services = create_default_services()
    yield


app = FastAPI(
    title="Smart Medical Scribe & Patient Dashboard API",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def get_services(request: Request) -> ServiceContainer:
    if not hasattr(request.app.state, "services"):
        request.app.state.services = create_default_services()
    return request.app.state.services


# --- Request / Response Payloads ---

class CreateConsultationRequest(BaseModel):
    patient_id: str


class PatchConsultationRequest(BaseModel):
    soap_json: Optional[SOAPNote] = None
    suggestions: Optional[list[SuggestionItem]] = None
    flags: Optional[list[str]] = None
    patient_summary: Optional[str] = None


class ConsultationResponse(BaseModel):
    id: str
    patient_id: str
    status: str
    transcript: str
    soap_json: Optional[dict[str, Any]] = None
    retrieval_query: str
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    patient_summary: Optional[str] = None
    created_at: datetime
    approved_at: Optional[datetime] = None


def _serialize_consultation(c: Consultation) -> ConsultationResponse:
    return ConsultationResponse(
        id=c.id,
        patient_id=c.patient_id,
        status=c.status,
        transcript=c.transcript,
        soap_json=c.soap_json,
        retrieval_query=c.retrieval_query,
        suggestions=c.suggestions_json or [],
        flags=c.flags_json or [],
        patient_summary=c.patient_summary,
        created_at=c.created_at,
        approved_at=c.approved_at,
    )


# --- Endpoints ---

@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.name.asc()).all()
    return [
        {"id": p.id, "name": p.name, "dob": p.dob, "sex": p.sex}
        for p in patients
    ]

@app.post("/similar-cases/ingest")
def ingest_similar_cases(
    cases: list[SimilarCaseInput],
    services: ServiceContainer = Depends(get_services),
):
    try:
        count = services.similar_cases.ingest_cases(cases)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ingested": count, "collection": SimilarCasesService.COLLECTION_NAME}

@app.get("/similar-cases/search", response_model=list[SimilarCaseResult])
def search_similar_cases(
    query: str = Query(..., min_length=2),
    top_k: int = Query(default=3, ge=1, le=10),
    services: ServiceContainer = Depends(get_services),
):
    return services.similar_cases.search_similar(query=query, top_k=top_k)

@app.post("/consultations", response_model=ConsultationResponse, status_code=201)
def create_consultation(
    payload: CreateConsultationRequest,
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{payload.patient_id}' not found.")

    consultation = Consultation(patient_id=payload.patient_id, status="recording")
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return _serialize_consultation(consultation)


@app.post("/consultations/{consultation_id}/audio")
async def upload_audio_chunk(
    consultation_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    services: ServiceContainer = Depends(get_services),
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if consultation.status == "approved":
        raise HTTPException(status_code=400, detail="Cannot append audio to an approved consultation.")

    audio_bytes = await file.read()
    filename = file.filename or "chunk.wav"

    try:
        chunk_text = services.stt.transcribe_chunk(
            filename=filename,
            audio_bytes=audio_bytes,
            previous_transcript=consultation.transcript,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    consultation.transcript = append_transcript_chunk(consultation.transcript, chunk_text)
    db.commit()
    db.refresh(consultation)

    return {
        "consultation_id": consultation.id,
        "chunk_transcript": chunk_text,
        "full_transcript": consultation.transcript,
    }


@app.post("/consultations/{consultation_id}/finalize", response_model=ConsultationResponse)
def finalize_consultation(
    consultation_id: str,
    include_patient_summary: bool = Query(default=False),
    db: Session = Depends(get_db),
    services: ServiceContainer = Depends(get_services),
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if consultation.status == "approved":
        raise HTTPException(status_code=400, detail="Consultation is already approved.")

    try:
        soap = services.structuring.generate_soap(consultation.transcript)
    except StructuringError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    retrieved_chunks = services.retrieval.retrieve(
        patient_id=consultation.patient_id,
        query=soap.retrieval_query,
    )

    try:
        gen_output = services.generation.generate_suggestions(
            db=db,
            patient_id=consultation.patient_id,
            soap=soap,
            retrieved_chunks=retrieved_chunks,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    patient_summary = consultation.patient_summary
    if include_patient_summary:
        patient_summary = services.generation.generate_patient_summary(
            soap=soap,
            suggestions=gen_output.suggestions,
        )

    consultation.status = "draft"
    consultation.soap_json = soap.model_dump()
    consultation.retrieval_query = soap.retrieval_query
    consultation.suggestions_json = [s.model_dump() for s in gen_output.suggestions]
    consultation.flags_json = gen_output.flags
    consultation.patient_summary = patient_summary

    db.commit()
    db.refresh(consultation)
    return _serialize_consultation(consultation)


@app.get("/consultations/{consultation_id}", response_model=ConsultationResponse)
def get_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    return _serialize_consultation(consultation)


@app.patch("/consultations/{consultation_id}", response_model=ConsultationResponse)
def patch_consultation(
    consultation_id: str,
    payload: PatchConsultationRequest,
    db: Session = Depends(get_db),
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if consultation.status == "approved":
        raise HTTPException(status_code=400, detail="Approved consultations are locked and cannot be edited.")

    if payload.soap_json is not None:
        updated_soap = payload.soap_json.model_copy(update={"status": "draft"})
        consultation.soap_json = updated_soap.model_dump()
        consultation.retrieval_query = updated_soap.retrieval_query

    if payload.suggestions is not None:
        consultation.suggestions_json = [s.model_dump() for s in payload.suggestions]

    if payload.flags is not None:
        consultation.flags_json = payload.flags

    if payload.patient_summary is not None:
        consultation.patient_summary = payload.patient_summary

    if consultation.status == "recording":
        consultation.status = "draft"

    db.commit()
    db.refresh(consultation)
    return _serialize_consultation(consultation)


@app.post("/consultations/{consultation_id}/approve", response_model=ConsultationResponse)
def approve_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    services: ServiceContainer = Depends(get_services),
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if consultation.status == "approved":
        return _serialize_consultation(consultation)
    if not consultation.soap_json:
        raise HTTPException(status_code=400, detail="Cannot approve a consultation without a finalized SOAP note.")

    soap = SOAPNote.model_validate(consultation.soap_json).model_copy(update={"status": "approved"})
    consultation.soap_json = soap.model_dump()
    consultation.status = "approved"
    consultation.approved_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(consultation)

    # Human-in-the-loop indexing: only APPROVED notes enter the patient EHR store
    services.ingestion.index_approved_consultation(db, consultation)
    return _serialize_consultation(consultation)


@app.get("/patients/{patient_id}/history")
def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    facts = (
        db.query(StructuredFact)
        .filter(StructuredFact.patient_id == patient_id)
        .order_by(StructuredFact.recorded_at.desc())
        .all()
    )
    records = (
        db.query(EHRRecord)
        .filter(EHRRecord.patient_id == patient_id)
        .order_by(EHRRecord.recorded_at.desc())
        .all()
    )
    consultations = (
        db.query(Consultation)
        .filter(Consultation.patient_id == patient_id)
        .order_by(Consultation.created_at.desc())
        .all()
    )

    return {
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "dob": patient.dob,
            "sex": patient.sex,
        },
        "structured_facts": [
            {
                "id": f.id,
                "fact_type": f.fact_type,
                "name": f.name,
                "value": f.value,
                "unit": f.unit,
                "recorded_at": f.recorded_at,
                "active": f.active,
            }
            for f in facts
        ],
        "ehr_records": [
            {
                "id": r.id,
                "record_type": r.record_type,
                "section": r.section,
                "content": r.content,
                "recorded_at": r.recorded_at,
                "source_consultation_id": r.source_consultation_id,
            }
            for r in records
        ],
        "consultations": [_serialize_consultation(c) for c in consultations],
    }