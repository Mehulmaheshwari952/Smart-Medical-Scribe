import re
from dataclasses import dataclass
from typing import Callable, Optional
import chromadb
from pydantic import BaseModel, Field
from app.config import settings
from app.services.ingestion import EmbeddingProvider

# Basic PHI/PII regex patterns to enforce de-identification before indexing into similar_cases
PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        # SSN
    re.compile(r"\b(?:\+?\d{1,2}\s?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # Phone
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),       # Email
    re.compile(r"(?i)\b(?:mrn|medical record number)\s*[:#]?\s*[a-z0-9-]+\b"), # MRN
]


class PHIViolationError(ValueError):
    """Raised when unredacted PHI/PII is detected in a case meant for similar_cases."""


def assert_deidentified(text: str) -> None:
    """Blocks indexing if obvious unredacted identifiers (SSN, phone, email, MRN) are present."""
    for pattern in PHI_PATTERNS:
        if pattern.search(text):
            raise PHIViolationError(
                f"Potential unredacted identifier matched pattern '{pattern.pattern}'. "
                "Only de-identified or synthetic cases may be indexed into similar_cases."
            )


class SimilarCaseInput(BaseModel):
    case_id: str
    source: str = "mimic-iv-demo"  # e.g., mimic-iv-demo | mtsamples | synthetic
    specialty: str = "general"
    diagnosis: str = "unknown"
    summary: str = Field(min_length=10)


class SimilarCaseResult(BaseModel):
    case_id: str
    source: str
    specialty: str
    diagnosis: str
    summary: str
    similarity_score: float


class SimilarCasesService:
    COLLECTION_NAME = "similar_cases"

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

    def ingest_cases(self, cases: list[SimilarCaseInput]) -> int:
        """Validates de-identification and upserts reference cases into the separate similar_cases collection."""
        if not cases:
            return 0

        for case in cases:
            assert_deidentified(case.summary)
            assert_deidentified(case.diagnosis)

        embeddings = self.embed_fn([c.summary for c in cases])
        self.collection.upsert(
            ids=[c.case_id for c in cases],
            embeddings=embeddings,
            documents=[c.summary for c in cases],
            metadatas=[
                {
                    "source": c.source,
                    "specialty": c.specialty,
                    "diagnosis": c.diagnosis,
                }
                for c in cases
            ],
        )
        return len(cases)

    def search_similar(self, query: str, top_k: int = 3) -> list[SimilarCaseResult]:
        cleaned = query.strip()
        if not cleaned or cleaned.lower() == "unknown":
            return []

        total = self.collection.count()
        if total == 0:
            return []

        k = min(top_k, total)
        query_vec = self.embed_fn([cleaned])[0]
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        results: list[SimilarCaseResult] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance = 1 - cosine_similarity
            sim_score = round(max(0.0, 1.0 - float(dist)), 4)
            results.append(
                SimilarCaseResult(
                    case_id=cid,
                    source=meta.get("source", "unknown"),
                    specialty=meta.get("specialty", "general"),
                    diagnosis=meta.get("diagnosis", "unknown"),
                    summary=doc,
                    similarity_score=sim_score,
                )
            )
        return results