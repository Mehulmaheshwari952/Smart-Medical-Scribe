from datetime import datetime, timedelta, timezone
from app.database import SessionLocal, init_db
from app.models import Patient
from app.services.ingestion import EHRIngestionService


def seed() -> None:
    init_db()
    db = SessionLocal()
    ingestion = EHRIngestionService()
    now = datetime.now(timezone.utc)

    demo_patients = [
        {
            "id": "pat-001",
            "name": "Aarav Sharma (48M - T2DM / Penicillin Allergy)",
            "dob": "1978-03-14",
            "sex": "M",
            "facts": [
                ("allergy", "Penicillin", "Severe hives / urticaria", ""),
                ("medication", "Metformin", "500", "mg BID"),
                ("chronic_diagnosis", "Type 2 Diabetes Mellitus", "Active", ""),
            ],
            "records": [
                (
                    "lab",
                    14,
                    "LABS: HbA1c 8.7%, Fasting plasma glucose 196 mg/dL, eGFR 78 mL/min, Serum creatinine 1.0 mg/dL.",
                ),
                (
                    "visit",
                    60,
                    "SUBJECTIVE: Routine diabetes follow-up, mild polyuria.\n"
                    "ASSESSMENT: Uncontrolled Type 2 diabetes mellitus.\n"
                    "PLAN: Continue Metformin 500 mg BID, recheck HbA1c in 3 months, consider adding SGLT2 inhibitor if >8%.",
                ),
            ],
        },
        {
            "id": "pat-002",
            "name": "Meera Nair (62F - CKD Stage 3b / Hypertension)",
            "dob": "1964-08-21",
            "sex": "F",
            "facts": [
                ("allergy", "Lisinopril", "Angioedema", ""),
                ("chronic_diagnosis", "Chronic Kidney Disease Stage 3b", "Baseline eGFR 36", ""),
                ("medication", "Amlodipine", "5", "mg daily"),
            ],
            "records": [
                (
                    "lab",
                    10,
                    "LABS: Serum creatinine 1.9 mg/dL, eGFR 35 mL/min, Potassium 4.8 mmol/L.",
                ),
                (
                    "visit",
                    90,
                    "ASSESSMENT: Bilateral knee osteoarthritis and CKD stage 3b.\n"
                    "PLAN: Avoid systemic NSAIDs (ibuprofen/naproxen) due to nephrotoxicity risk.",
                ),
            ],
        },
        {
            "id": "pat-003",
            "name": "Rohan Verma (34M - Moderate Persistent Asthma)",
            "dob": "1992-11-09",
            "sex": "M",
            "facts": [
                ("allergy", "Aspirin", "Bronchospasm", ""),
                ("medication", "Albuterol HFA", "90", "mcg PRN"),
            ],
            "records": [
                (
                    "visit",
                    25,
                    "SUBJECTIVE: Using rescue albuterol 4-5 times per week, nocturnal coughing.\n"
                    "ASSESSMENT: Moderate persistent asthma, suboptimally controlled.\n"
                    "PLAN: Initiate daily low-dose inhaled corticosteroid (budesonide) controller therapy.",
                ),
            ],
        },
    ]

    for p in demo_patients:
        existing = db.query(Patient).filter_by(id=p["id"]).first()
        if existing:
            continue
        db.add(Patient(id=p["id"], name=p["name"], dob=p["dob"], sex=p["sex"]))
        db.commit()

        for ftype, name, val, unit in p["facts"]:
            ingestion.ingest_structured_fact(db, p["id"], ftype, name, val, unit)

        for rtype, days_ago, text in p["records"]:
            ingestion.ingest_ehr_record(
                db,
                p["id"],
                rtype,
                text,
                recorded_at=now - timedelta(days=days_ago),
            )

    db.close()
    print("Demo patients seeded successfully.")


if __name__ == "__main__":
    seed()