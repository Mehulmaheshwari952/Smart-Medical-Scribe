import argparse
import statistics
import time
from datetime import timedelta
from typing import Callable
import chromadb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.eval.synthetic_data import (
    REF_NOW,
    RETRIEVAL_BENCHMARK_PATIENTS,
    STRUCTURING_BENCHMARK_CASES,
)
from app.models import Base, Patient
from app.services.ingestion import EHRIngestionService, EmbeddingProvider
from app.services.retrieval import HybridRetrievalService, RerankerFn, baseline_time_aware_rerank, compute_rrf
from app.services.structuring import SOAPStructuringService
import json
from pathlib import Path
from app.eval.custom_reranker import SemanticAdaptiveReranker

def _normalize_item(text: str) -> str:
    return " ".join(text.lower().strip().split())


def compute_set_prf(gold_items: list[str], pred_items: list[str]) -> tuple[float, float, float]:
    """Computes token-overlap / substring-tolerant Precision, Recall, and F1 for clinical field lists."""
    gold_norm = [_normalize_item(g) for g in gold_items if g and g.lower() != "unknown"]
    pred_norm = [_normalize_item(p) for p in pred_items if p and p.lower() != "unknown"]

    if not gold_norm and not pred_norm:
        return 1.0, 1.0, 1.0
    if not pred_norm:
        return 0.0, 0.0, 0.0
    if not gold_norm:
        return 0.0, 1.0, 0.0

    matched_gold = set()
    tp = 0
    for p in pred_norm:
        for idx, g in enumerate(gold_norm):
            if idx not in matched_gold and (p in g or g in p):
                tp += 1
                matched_gold.add(idx)
                break

    precision = tp / len(pred_norm)
    recall = tp / len(gold_norm)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_structuring(structuring_service: SOAPStructuringService) -> dict:
    field_scores: dict[str, list[tuple[float, float, float]]] = {
        "symptoms": [],
        "diagnoses": [],
        "medications": [],
    }
    latencies_ms: list[float] = []

    for case in STRUCTURING_BENCHMARK_CASES:
        t0 = time.perf_counter()
        soap = structuring_service.generate_soap(case["transcript"])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        pred_meds = [m.name for m in soap.plan.medications]
        field_scores["symptoms"].append(compute_set_prf(case["gold"]["symptoms"], soap.subjective.symptoms))
        field_scores["diagnoses"].append(compute_set_prf(case["gold"]["diagnoses"], soap.assessment.diagnoses))
        field_scores["medications"].append(compute_set_prf(case["gold"]["medications"], pred_meds))

    summary = {"mean_latency_ms": round(statistics.mean(latencies_ms), 2), "fields": {}}
    for field, triples in field_scores.items():
        p = statistics.mean(t[0] for t in triples)
        r = statistics.mean(t[1] for t in triples)
        f1 = statistics.mean(t[2] for t in triples)
        summary["fields"][field] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
    return summary


def evaluate_retrieval_reranker(
    ingestion: EHRIngestionService,
    reranker_fn: RerankerFn,
    top_k: int = 5,
) -> dict:
    retriever = HybridRetrievalService(ingestion_service=ingestion, reranker_fn=reranker_fn)

    recalls_at_1: list[float] = []
    recalls_at_3: list[float] = []
    recalls_at_5: list[float] = []
    precisions_at_3: list[float] = []
    precisions_at_5: list[float] = []
    mrr_scores: list[float] = []

    stage_latencies: dict[str, list[float]] = {
        "vector_ms": [],
        "bm25_ms": [],
        "fusion_ms": [],
        "rerank_ms": [],
        "total_ms": [],
    }

    for patient_case in RETRIEVAL_BENCHMARK_PATIENTS:
        pid = patient_case["patient_id"]
        query = patient_case["query"]
        gold_records = set(patient_case["relevant_record_ids"])

        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        vec_cands = retriever._vector_search(pid, query, n_results=20)
        t_vec = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bm25_cands = retriever._bm25_search(pid, query, n_results=20)
        t_bm25 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        fused = compute_rrf(vec_cands, bm25_cands, k=retriever.rrf_k)
        t_fus = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        reranked = reranker_fn(fused, query, REF_NOW)[:top_k]
        t_rer = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total0) * 1000.0

        stage_latencies["vector_ms"].append(t_vec)
        stage_latencies["bm25_ms"].append(t_bm25)
        stage_latencies["fusion_ms"].append(t_fus)
        stage_latencies["rerank_ms"].append(t_rer)
        stage_latencies["total_ms"].append(t_total)

        # Deduplicate retrieved record_ids preserving rank order
        ranked_record_ids: list[str] = []
        for chunk in reranked:
            if chunk.record_id not in ranked_record_ids:
                ranked_record_ids.append(chunk.record_id)

        recalls_at_1.append(len(set(ranked_record_ids[:1]) & gold_records) / len(gold_records))
        recalls_at_3.append(len(set(ranked_record_ids[:3]) & gold_records) / len(gold_records))
        recalls_at_5.append(len(set(ranked_record_ids[:5]) & gold_records) / len(gold_records))
        top3_ids = ranked_record_ids[:3]
        top5_ids = ranked_record_ids[:5]
        precisions_at_3.append(len(set(top3_ids) & gold_records) / max(1, len(top3_ids)))
        precisions_at_5.append(len(set(top5_ids) & gold_records) / max(1, len(top5_ids)))

        rr = 0.0
        for rank_idx, rec_id in enumerate(ranked_record_ids, start=1):
            if rec_id in gold_records:
                rr = 1.0 / rank_idx
                break
        mrr_scores.append(rr)

    return {
        "recall@1": round(statistics.mean(recalls_at_1), 4),
        "recall@3": round(statistics.mean(recalls_at_3), 4),
        "recall@5": round(statistics.mean(recalls_at_5), 4),
        "precision@3": round(statistics.mean(precisions_at_3), 4),
        "precision@5": round(statistics.mean(precisions_at_5), 4),
        "mrr": round(statistics.mean(mrr_scores), 4),
        "latency_ms": {k: round(statistics.mean(v), 3) for k, v in stage_latencies.items()},
    }


def setup_synthetic_ehr_store(embed_fn: Callable[[list[str]], list[list[float]]]) -> EHRIngestionService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    chroma = chromadb.EphemeralClient()
    try:
        chroma.delete_collection(EHRIngestionService.COLLECTION_NAME)
    except Exception:
        pass

    ingestion = EHRIngestionService(chroma_client=chroma, embed_fn=embed_fn)

    for p in RETRIEVAL_BENCHMARK_PATIENTS:
        db.add(Patient(id=p["patient_id"], name=p["name"], dob=p["dob"], sex=p["sex"]))
        db.commit()
        for rec in p["records"]:
            rec_dt = REF_NOW - timedelta(days=rec["days_ago"])
            # Preserve exact record ID from benchmark fixture
            from app.models import EHRRecord
            from app.services.ingestion import chunk_ehr_record

            row = EHRRecord(
                id=rec["id"],
                patient_id=p["patient_id"],
                record_type=rec["type"],
                section="general",
                content=rec["text"],
                recorded_at=rec_dt,
            )
            db.add(row)
            db.commit()
            chunks = chunk_ehr_record(
                record_id=rec["id"],
                patient_id=p["patient_id"],
                record_type=rec["type"],
                content=rec["text"],
                recorded_at=rec_dt,
            )
            ingestion._upsert_chunks(chunks)

    db.close()
    return ingestion


def run_benchmark(mode: str = "retrieval", use_real_embeddings: bool = True) -> dict:
    results: dict = {}

    if mode in ("retrieval", "all"):
        if use_real_embeddings:
            embed_fn = EmbeddingProvider().encode
        else:
            # Fast lexical-hash fallback for instant CI unit tests
            def embed_fn(texts: list[str]) -> list[list[float]]:
                vecs = []
                for t in texts:
                    v = [0.0] * 32
                    for tok in t.lower().split():
                        v[hash(tok) % 32] += 1.0
                    vecs.append(v)
                return vecs

        ingestion = setup_synthetic_ehr_store(embed_fn=embed_fn)
        adaptive_reranker = SemanticAdaptiveReranker(embed_fn=embed_fn)
        baseline_metrics = evaluate_retrieval_reranker(ingestion, baseline_time_aware_rerank)
        custom_metrics = evaluate_retrieval_reranker(ingestion, adaptive_reranker)
        results["retrieval"] = {
            "baseline": baseline_metrics,
            "custom": custom_metrics,
        }

    if mode in ("structuring", "all"):
        structuring_service = SOAPStructuringService()
        results["structuring"] = evaluate_structuring(structuring_service)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Medical Scribe Evaluation Harness")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "structuring", "all"],
        default="retrieval",
        help="Run retrieval comparison (offline), structuring F1 (calls Groq), or all.",
    )
    parser.add_argument(
        "--fast-embed",
        action="store_true",
        help="Use lightweight hash embeddings instead of downloading bge-small-en-v1.5.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save evaluation report JSON (e.g., data/eval_baseline.json).",
    )
    
    args = parser.parse_args()
    
    report = run_benchmark(mode=args.mode, use_real_embeddings=not args.fast_embed)

    if "retrieval" in report:
        print("\n=== RETRIEVAL BENCHMARK (BASELINE vs CUSTOM RERANKER) ===")
        for variant in ("baseline", "custom"):
            m = report["retrieval"][variant]
            print(
                f"[{variant.upper():8s}] Recall@1={m['recall@1']:.3f} | "
                f"Recall@3={m['recall@3']:.3f} | Recall@5={m['recall@5']:.3f} | "
                f"Prec@3={m['precision@3']:.3f} | Prec@5={m['precision@5']:.3f} | "
                f"MRR={m['mrr']:.3f} | Latency(ms)={m['latency_ms']}"
            )

    if "structuring" in report:
        print("\n=== SOAP STRUCTURING BENCHMARK (20 MTSamples Cases) ===")
        s = report["structuring"]
        print(f"Mean Latency: {s['mean_latency_ms']} ms")
        for field, scores in s["fields"].items():
            print(
                f"  {field:12s} -> Precision={scores['precision']:.3f} | "
                f"Recall={scores['recall']:.3f} | F1={scores['f1']:.3f}"
            )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark report to {out_path}")


if __name__ == "__main__":
    main()