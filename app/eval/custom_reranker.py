from datetime import datetime
from typing import Optional
from app.schemas import RetrievedChunk
from app.services.retrieval import baseline_time_aware_rerank


def custom_rerank(
    candidates: list[RetrievedChunk],
    query: str,
    now: Optional[datetime] = None,
) -> list[RetrievedChunk]:
    """
    EXPERIMENT SLOT:
    Replace this stub with your custom improvement (e.g., adaptive temporal weighting,
    clinical entity/section boosting, vector quantization, or retrieval gating)
    and run `python -m app.eval.runner --mode retrieval` to compare vs. baseline.
    """
    # Currently delegates to baseline so the harness works out of the box
    return baseline_time_aware_rerank(candidates, query, now=now)