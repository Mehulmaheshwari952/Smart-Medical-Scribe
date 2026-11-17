from datetime import datetime, timezone
from typing import Callable, Optional
from app.config import settings
from app.schemas import RetrievedChunk
from app.services.ingestion import EHRIngestionService, TextChunk, tokenize_clinical_text

RerankerFn = Callable[[list[RetrievedChunk], str, Optional[datetime]], list[RetrievedChunk]]


def compute_rrf(
    vector_chunks: list[TextChunk],
    bm25_chunks: list[TextChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Fuses ranked vector and BM25 candidate lists via Reciprocal Rank Fusion (1-indexed ranks).
    """
    fused: dict[str, RetrievedChunk] = {}

    for rank_idx, chunk in enumerate(vector_chunks, start=1):
        parsed_dt = datetime.fromisoformat(chunk.date)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)

        item = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            patient_id=chunk.patient_id,
            record_id=chunk.record_id,
            record_type=chunk.record_type,  # type: ignore[arg-type]
            date=parsed_dt,
            section=chunk.section,
            text=chunk.text,
            vector_rank=rank_idx,
            rrf_score=1.0 / (k + rank_idx),
        )
        fused[chunk.chunk_id] = item

    for rank_idx, chunk in enumerate(bm25_chunks, start=1):
        rrf_contrib = 1.0 / (k + rank_idx)
        if chunk.chunk_id in fused:
            fused[chunk.chunk_id].bm25_rank = rank_idx
            fused[chunk.chunk_id].rrf_score += rrf_contrib
        else:
            parsed_dt = datetime.fromisoformat(chunk.date)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)

            fused[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                patient_id=chunk.patient_id,
                record_id=chunk.record_id,
                record_type=chunk.record_type,  # type: ignore[arg-type]
                date=parsed_dt,
                section=chunk.section,
                text=chunk.text,
                bm25_rank=rank_idx,
                rrf_score=rrf_contrib,
            )

    # Sort descending by rrf_score (tie-break deterministically by chunk_id)
    return sorted(fused.values(), key=lambda c: (-c.rrf_score, c.chunk_id))


def baseline_time_aware_rerank(
    candidates: list[RetrievedChunk],
    query: str,
    now: Optional[datetime] = None,
) -> list[RetrievedChunk]:
    """
    BASELINE RERANKER (pluggable slot):
    final = rrf * 0.5^(age_days / half_life[record_type]).
    Allergy and chronic-diagnosis records have half_life=None and do not decay.
    """
    ref_now = now or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    half_lives = settings.half_life_days

    for candidate in candidates:
        half_life = half_lives.get(candidate.record_type)
        if half_life is None or half_life <= 0:
            decay_factor = 1.0
        else:
            chunk_dt = candidate.date
            if chunk_dt.tzinfo is None:
                chunk_dt = chunk_dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (ref_now - chunk_dt).total_seconds() / 86400.0)
            decay_factor = 0.5 ** (age_days / half_life)

        candidate.final_score = candidate.rrf_score * decay_factor

    return sorted(candidates, key=lambda c: (-c.final_score, c.chunk_id))


# Default pluggable alias matching the specification signature: rerank(candidates, query, now) -> candidates
rerank: RerankerFn = baseline_time_aware_rerank


class HybridRetrievalService:
    def __init__(
        self,
        ingestion_service: EHRIngestionService,
        reranker_fn: RerankerFn = rerank,
        rrf_k: int = settings.rrf_k,
    ):
        self.ingestion = ingestion_service
        self.reranker_fn = reranker_fn
        self.rrf_k = rrf_k

    def _vector_search(self, patient_id: str, query: str, n_results: int = 20) -> list[TextChunk]:
        # Check how many records exist for this patient so Chroma doesn't complain if n_results > count
        patient_docs = self.ingestion.collection.get(
            where={"patient_id": patient_id},
            include=[],
        )
        total_for_patient = len(patient_docs.get("ids") or [])
        if total_for_patient == 0:
            return []

        k = min(n_results, total_for_patient)
        query_vec = self.ingestion.embed_fn([query])[0]
        res = self.ingestion.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where={"patient_id": patient_id},
            include=["documents", "metadatas"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        return [
            TextChunk(
                chunk_id=cid,
                patient_id=meta["patient_id"],
                record_id=meta["record_id"],
                record_type=meta["record_type"],
                date=meta["date"],
                section=meta["section"],
                text=doc,
            )
            for cid, doc, meta in zip(ids, docs, metas)
        ]

    def _bm25_search(self, patient_id: str, query: str, n_results: int = 20) -> list[TextChunk]:
        bm25_index = self.ingestion.build_patient_bm25(patient_id)
        if bm25_index.bm25 is None or not bm25_index.chunks:
            return []

        query_tokens = tokenize_clinical_text(query)
        if not query_tokens:
            return []

        scores = bm25_index.bm25.get_scores(query_tokens)
        scored_pairs = [
            (score, chunk)
            for score, chunk in zip(scores, bm25_index.chunks)
            if score > 0
        ]
        scored_pairs.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [chunk for _, chunk in scored_pairs[:n_results]]

    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: int = settings.retrieval_top_k,
        now: Optional[datetime] = None,
    ) -> list[RetrievedChunk]:
        cleaned_query = query.strip()
        if not cleaned_query or cleaned_query.lower() == "unknown":
            return []

        vector_top20 = self._vector_search(patient_id, cleaned_query, n_results=20)
        bm25_top20 = self._bm25_search(patient_id, cleaned_query, n_results=20)

        fused_candidates = compute_rrf(vector_top20, bm25_top20, k=self.rrf_k)
        if not fused_candidates:
            return []

        reranked = self.reranker_fn(fused_candidates, cleaned_query, now)
        return reranked[:top_k]