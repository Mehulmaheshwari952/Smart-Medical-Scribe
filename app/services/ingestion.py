import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
import chromadb
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Consultation, EHRRecord, StructuredFact
from app.schemas import SOAPNote

SECTION_HEADER_RE = re.compile(
    r"(?im)^(subjective|objective|assessment|plan|chief complaint|hpi|"
    r"history of present illness|past medical history|medications|allergies|"
    r"physical exam|labs|lab results|impression|discharge summary)\s*:\s*"
)


@dataclass
class TextChunk:
    chunk_id: str
    patient_id: str
    record_id: str
    record_type: str
    date: str  # ISO-8601 string
    section: str
    text: str


@dataclass
class PatientBM25Index:
    bm25: Optional[BM25Okapi]
    chunks: list[TextChunk]


def tokenize_clinical_text(text: str) -> list[str]:
    """Lowercase tokenizer preserving hyphenated medical terms and numbers (e.g., hba1c, 500mg)."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _split_long_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    pieces: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            # Prefer breaking on sentence or whitespace boundary
            split_at = max(cleaned.rfind(". ", start, end), cleaned.rfind(" ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        piece = cleaned[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_ehr_record(
    record_id: str,
    patient_id: str,
    record_type: str,
    content: str,
    recorded_at: datetime,
    default_section: str = "general",
    max_chars: int = 600,
) -> list[TextChunk]:
    """
    Splits an EHR record into clinical sections (e.g., Subjective, Objective, Assessment, Plan).
    Falls back to bounded window splitting only if a section exceeds max_chars.
    """
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    iso_date = recorded_at.isoformat()

    matches = list(SECTION_HEADER_RE.finditer(content))
    raw_sections: list[tuple[str, str]] = []

    if not matches:
        raw_sections.append((default_section.lower(), content.strip()))
    else:
        # Capture any preamble before the first header
        if matches[0].start() > 0:
            preamble = content[: matches[0].start()].strip()
            if preamble:
                raw_sections.append((default_section.lower(), preamble))

        for idx, match in enumerate(matches):
            sec_name = match.group(1).strip().lower()
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()
            if body:
                raw_sections.append((sec_name, body))

    chunks: list[TextChunk] = []
    chunk_idx = 0
    for sec_name, sec_text in raw_sections:
        for sub_text in _split_long_text(sec_text, max_chars=max_chars):
            chunks.append(
                TextChunk(
                    chunk_id=f"{record_id}::{chunk_idx}",
                    patient_id=patient_id,
                    record_id=record_id,
                    record_type=record_type,
                    date=iso_date,
                    section=sec_name,
                    text=sub_text,
                )
            )
            chunk_idx += 1

    return chunks


class EmbeddingProvider:
    """Lazy-loaded bge-small-en-v1.5 wrapper ensuring identical embeddings for index and query."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class EHRIngestionService:
    COLLECTION_NAME = "patient_ehr_chunks"

    def __init__(
        self,
        chroma_client: Optional[chromadb.ClientAPI] = None,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ):
        self.chroma_client = chroma_client or chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._default_embedder = EmbeddingProvider() if embed_fn is None else None
        self.embed_fn = embed_fn or self._default_embedder.encode

    def ingest_structured_fact(
        self,
        db: Session,
        patient_id: str,
        fact_type: str,
        name: str,
        value: str = "unknown",
        unit: str = "",
        recorded_at: Optional[datetime] = None,
    ) -> StructuredFact:
        """Stores structured clinical facts (allergies, meds, labs) in SQL only—never in vector RAG."""
        fact = StructuredFact(
            patient_id=patient_id,
            fact_type=fact_type,
            name=name,
            value=value,
            unit=unit,
            recorded_at=recorded_at or datetime.now(timezone.utc),
            active=True,
        )
        db.add(fact)
        db.commit()
        db.refresh(fact)
        return fact

    def ingest_ehr_record(
        self,
        db: Session,
        patient_id: str,
        record_type: str,
        content: str,
        section: str = "general",
        recorded_at: Optional[datetime] = None,
        source_consultation_id: Optional[str] = None,
    ) -> tuple[EHRRecord, list[TextChunk]]:
        """Persists a free-text EHR record in SQL and indexes its section chunks into Chroma."""
        ts = recorded_at or datetime.now(timezone.utc)
        record = EHRRecord(
            patient_id=patient_id,
            record_type=record_type,
            section=section,
            content=content,
            recorded_at=ts,
            source_consultation_id=source_consultation_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        chunks = chunk_ehr_record(
            record_id=record.id,
            patient_id=patient_id,
            record_type=record_type,
            content=content,
            recorded_at=ts,
            default_section=section,
        )
        self._upsert_chunks(chunks)
        return record, chunks

    def index_approved_consultation(
        self, db: Session, consultation: Consultation
    ) -> tuple[EHRRecord, list[TextChunk]]:
        """Human-in-the-loop gate: only APPROVED consultations may enter the patient EHR store."""
        if consultation.status != "approved":
            raise ValueError(
                f"Cannot index consultation '{consultation.id}' with status '{consultation.status}'. "
                "Clinician approval is required first."
            )
        if not consultation.soap_json:
            raise ValueError(f"Consultation '{consultation.id}' has no SOAP note to index.")

        soap = SOAPNote.model_validate(consultation.soap_json)
        meds_str = ", ".join(
            f"{m.name} {m.dose} ({m.frequency})" for m in soap.plan.medications
        ) or "none"

        formatted_note = (
            f"SUBJECTIVE: Chief complaint: {soap.subjective.chief_complaint}. "
            f"Symptoms: {', '.join(soap.subjective.symptoms) or 'none'}. "
            f"HPI: {soap.subjective.hpi}. History: {soap.subjective.relevant_history}.\n"
            f"OBJECTIVE: Vitals: {soap.objective.vitals}. "
            f"Exam: {', '.join(soap.objective.exam_findings) or 'none'}. "
            f"Labs: {', '.join(soap.objective.lab_results) or 'none'}.\n"
            f"ASSESSMENT: Diagnoses: {', '.join(soap.assessment.diagnoses) or 'none'}. "
            f"Conditions: {', '.join(soap.assessment.conditions) or 'none'}.\n"
            f"PLAN: Treatment: {', '.join(soap.plan.treatment) or 'none'}. "
            f"Medications: {meds_str}. "
            f"Tests ordered: {', '.join(soap.plan.tests_ordered) or 'none'}. "
            f"Follow-up: {soap.plan.follow_up}."
        )

        return self.ingest_ehr_record(
            db=db,
            patient_id=consultation.patient_id,
            record_type="visit",
            content=formatted_note,
            section="soap_note",
            recorded_at=consultation.approved_at or datetime.now(timezone.utc),
            source_consultation_id=consultation.id,
        )

    def _upsert_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embed_fn([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "patient_id": c.patient_id,
                    "record_id": c.record_id,
                    "record_type": c.record_type,
                    "date": c.date,
                    "section": c.section,
                }
                for c in chunks
            ],
        )

    def build_patient_bm25(self, patient_id: str) -> PatientBM25Index:
        """Builds an in-memory BM25Okapi index on demand strictly scoped to patient_id."""
        results = self.collection.get(
            where={"patient_id": patient_id},
            include=["documents", "metadatas"],
        )
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        chunks: list[TextChunk] = []
        for cid, doc, meta in zip(ids, docs, metas):
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    patient_id=meta["patient_id"],
                    record_id=meta["record_id"],
                    record_type=meta["record_type"],
                    date=meta["date"],
                    section=meta["section"],
                    text=doc,
                )
            )

        if not chunks:
            return PatientBM25Index(bm25=None, chunks=[])

        tokenized_corpus = [tokenize_clinical_text(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus, epsilon=0.25)
        return PatientBM25Index(bm25=bm25, chunks=chunks)