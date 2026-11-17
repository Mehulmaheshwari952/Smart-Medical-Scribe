import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    dob = Column(String, nullable=False)
    sex = Column(String, nullable=False)

    structured_facts = relationship("StructuredFact", back_populates="patient", cascade="all, delete-orphan")
    ehr_records = relationship("EHRRecord", back_populates="patient", cascade="all, delete-orphan")
    consultations = relationship("Consultation", back_populates="patient", cascade="all, delete-orphan")


class StructuredFact(Base):
    """Deterministic clinical facts queried via SQL, bypassing vector search."""
    __tablename__ = "structured_facts"

    id = Column(String, primary_key=True, default=_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False, index=True)
    fact_type = Column(String, nullable=False)  # allergy | medication | lab | chronic_diagnosis
    name = Column(String, nullable=False)
    value = Column(String, default="unknown")
    unit = Column(String, default="")
    recorded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    patient = relationship("Patient", back_populates="structured_facts")


class EHRRecord(Base):
    """Free-text EHR narratives indexed into Chroma + BM25."""
    __tablename__ = "ehr_records"

    id = Column(String, primary_key=True, default=_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False, index=True)
    record_type = Column(String, nullable=False)  # note | visit | lab | diagnosis | allergy
    section = Column(String, default="general", nullable=False)
    content = Column(Text, nullable=False)
    recorded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    source_consultation_id = Column(String, ForeignKey("consultations.id"), nullable=True)

    patient = relationship("Patient", back_populates="ehr_records")


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(String, primary_key=True, default=_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False, index=True)
    status = Column(String, default="recording", nullable=False)  # recording | draft | approved
    transcript = Column(Text, default="", nullable=False)
    soap_json = Column(JSON, nullable=True)
    retrieval_query = Column(String, default="", nullable=False)
    suggestions_json = Column(JSON, default=list, nullable=False)
    flags_json = Column(JSON, default=list, nullable=False)
    patient_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    approved_at = Column(DateTime(timezone=True), nullable=True)

    patient = relationship("Patient", back_populates="consultations")