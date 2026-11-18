from datetime import datetime, timezone
from typing import Callable, Optional
import numpy as np
from app.config import settings
from app.schemas import RetrievedChunk
from app.services.ingestion import EmbeddingProvider

ACUTE_ANCHOR = "recent laboratory test results, acute vital signs, and current biomarker values"
CHRONIC_ANCHOR = "long-term chronic medical history, prior medication failures, and baseline diagnoses"
SAFETY_ANCHOR = "severe drug allergy, anaphylaxis, contraindication, adverse drug reaction, or organ toxicity"


class SemanticAdaptiveReranker:
    """
    Zero-keyword, embedding-driven reranker using bge-small-en-v1.5:
    1. Continuous semantic temporal half-life (acute vs. longitudinal query intent).
    2. Continuous semantic safety alignment (query-to-risk x chunk-to-risk).
    3. Distribution-based relative score gating to filter low-signal noise.
    """

    def __init__(self, embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None):
        self._provider = EmbeddingProvider() if embed_fn is None else None
        self.embed_fn = embed_fn or self._provider.encode
        self._anchors: Optional[np.ndarray] = None  # shape: (3, dim)

    def _get_anchors(self) -> np.ndarray:
        if self._anchors is None:
            vecs = np.array(
                self.embed_fn([ACUTE_ANCHOR, CHRONIC_ANCHOR, SAFETY_ANCHOR]),
                dtype=np.float32,
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            self._anchors = vecs / norms
        return self._anchors

    def __call__(
        self,
        candidates: list[RetrievedChunk],
        query: str,
        now: Optional[datetime] = None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        ref_now = now or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)

        anchors = self._get_anchors()  # [0]=acute, [1]=chronic, [2]=safety
        texts = [query] + [c.text for c in candidates]
        vecs = np.array(self.embed_fn(texts), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

        q_vec = vecs[0]
        chunk_vecs = vecs[1:]

        # Continuous temporal shift in [-1, 1]: +1 = purely acute/lab, -1 = purely longitudinal/chronic
        acute_sim = float(np.dot(q_vec, anchors[0]))
        chronic_sim = float(np.dot(q_vec, anchors[1]))
        temporal_shift = np.clip((acute_sim - chronic_sim) * 4.0, -1.0, 1.0)

        # Query alignment with clinical safety/risk concept in [0, 1]
        q_safety_sim = max(0.0, float(np.dot(q_vec, anchors[2])))
        base_half_lives = settings.half_life_days

        for idx, candidate in enumerate(candidates):
            base_hl = base_half_lives.get(candidate.record_type)
            if base_hl is None or base_hl <= 0:
                decay_factor = 1.0
            else:
                # Smoothly scales half-life from 0.5x (acute intent) to 2.0x (chronic intent)
                adaptive_hl = base_hl * (2.0 ** (-temporal_shift))
                chunk_dt = candidate.date
                if chunk_dt.tzinfo is None:
                    chunk_dt = chunk_dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (ref_now - chunk_dt).total_seconds() / 86400.0)
                decay_factor = 0.5 ** (age_days / adaptive_hl)

            # Continuous semantic safety boost when both query and chunk align with clinical risk
            c_safety_sim = max(0.0, float(np.dot(chunk_vecs[idx], anchors[2])))
            safety_boost = 1.0 + (q_safety_sim * c_safety_sim)

            candidate.final_score = candidate.rrf_score * decay_factor * safety_boost

        ranked = sorted(candidates, key=lambda c: (-c.final_score, c.chunk_id))

        # Relative distribution gating: drop noise chunks below 45% of top candidate's score
        top_score = ranked[0].final_score
        if top_score > 0:
            ranked = [c for c in ranked if c.final_score >= 0.45 * top_score]

        return ranked


custom_rerank = SemanticAdaptiveReranker()