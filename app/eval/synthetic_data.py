from datetime import datetime, timedelta, timezone

REF_NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)

# 20 hand-labeled MTSamples-style consultation transcripts for Structuring Precision/Recall/F1
STRUCTURING_BENCHMARK_CASES: list[dict] = [
    {
        "id": "mts-01",
        "transcript": "Patient is a 45-year-old male complaining of burning chest pain after meals and sour taste for 3 weeks. Exam unremarkable. Assessment: Gastroesophageal reflux disease. Plan: Omeprazole 20 mg once daily before breakfast.",
        "gold": {
            "symptoms": ["burning chest pain", "sour taste"],
            "diagnoses": ["gastroesophageal reflux disease"],
            "medications": ["omeprazole"],
        },
    },
    {
        "id": "mts-02",
        "transcript": "I've had a throbbing unilateral headache with nausea and light sensitivity since morning. Vitals normal, neuro intact. Looks like an acute migraine. Take sumatriptan 50 mg now and rest in a dark room.",
        "gold": {
            "symptoms": ["throbbing unilateral headache", "nausea", "light sensitivity"],
            "diagnoses": ["acute migraine"],
            "medications": ["sumatriptan"],
        },
    },
    {
        "id": "mts-03",
        "transcript": "Three days of dysuria, urinary urgency, and suprapubic discomfort. No fever. Urinalysis positive for nitrites. Diagnosis: Uncomplicated cystitis. Prescribing nitrofurantoin 100 mg twice daily for 5 days.",
        "gold": {
            "symptoms": ["dysuria", "urinary urgency", "suprapubic discomfort"],
            "diagnoses": ["uncomplicated cystitis"],
            "medications": ["nitrofurantoin"],
        },
    },
    {
        "id": "mts-04",
        "transcript": "Patient presents with productive cough, fever of 101.2, and right lower lobe crackles. Chest X-ray shows right lower lobe infiltrate. Community-acquired pneumonia. Start azithromycin 500 mg day one then 250 mg daily.",
        "gold": {
            "symptoms": ["productive cough", "fever"],
            "diagnoses": ["community-acquired pneumonia"],
            "medications": ["azithromycin"],
        },
    },
    {
        "id": "mts-05",
        "transcript": "Follow-up for type 2 diabetes mellitus. Reports increased thirst and frequent urination. HbA1c came back at 8.6 percent. Increase metformin to 1000 mg twice daily.",
        "gold": {
            "symptoms": ["increased thirst", "frequent urination"],
            "diagnoses": ["type 2 diabetes mellitus"],
            "medications": ["metformin"],
        },
    },
    {
        "id": "mts-06",
        "transcript": "Complains of itchy watery eyes and sneezing every spring. Nasal turbinates pale and boggy. Allergic rhinitis. Use fluticasone nasal spray 2 sprays each nostril daily and loratadine 10 mg daily.",
        "gold": {
            "symptoms": ["itchy watery eyes", "sneezing"],
            "diagnoses": ["allergic rhinitis"],
            "medications": ["fluticasone", "loratadine"],
        },
    },
    {
        "id": "mts-07",
        "transcript": "Severe right great toe pain and swelling that started overnight. Erythematous warm first MTP joint. Acute gout flare. Start indomethacin 50 mg three times daily with food.",
        "gold": {
            "symptoms": ["right great toe pain", "swelling"],
            "diagnoses": ["acute gout flare"],
            "medications": ["indomethacin"],
        },
    },
    {
        "id": "mts-08",
        "transcript": "Shortness of breath and wheezing after exposure to cold air. Bilateral expiratory wheezes on auscultation. Acute asthma exacerbation. Administer albuterol nebulizer and prescribe prednisone 40 mg daily for 5 days.",
        "gold": {
            "symptoms": ["shortness of breath", "wheezing"],
            "diagnoses": ["acute asthma exacerbation"],
            "medications": ["albuterol", "prednisone"],
        },
    },
    {
        "id": "mts-09",
        "transcript": "Fatigue, weight gain of 10 pounds, and cold intolerance over 4 months. TSH elevated at 11.4. Primary hypothyroidism. Start levothyroxine 50 mcg every morning on an empty stomach.",
        "gold": {
            "symptoms": ["fatigue", "weight gain", "cold intolerance"],
            "diagnoses": ["primary hypothyroidism"],
            "medications": ["levothyroxine"],
        },
    },
    {
        "id": "mts-10",
        "transcript": "Routine hypertension check. Blood pressure is 148 over 92 today. Denies headache or chest pain. Uncontrolled essential hypertension. Add amlodipine 5 mg once daily.",
        "gold": {
            "symptoms": [],
            "diagnoses": ["uncontrolled essential hypertension"],
            "medications": ["amlodipine"],
        },
    },
    {
        "id": "mts-11",
        "transcript": "Painful vesicular rash on the left thoracic dermatome for 2 days with burning pain. Herpes zoster shingles. Start valacyclovir 1000 mg three times daily for 7 days.",
        "gold": {
            "symptoms": ["painful vesicular rash", "burning pain"],
            "diagnoses": ["herpes zoster"],
            "medications": ["valacyclovir"],
        },
    },
    {
        "id": "mts-12",
        "transcript": "Sore throat, fever, and painful swallowing for 2 days. Tonsillar exudates and anterior cervical lymphadenopathy. Rapid strep positive. Streptococcal pharyngitis. Amoxicillin 500 mg twice daily for 10 days.",
        "gold": {
            "symptoms": ["sore throat", "fever", "painful swallowing"],
            "diagnoses": ["streptococcal pharyngitis"],
            "medications": ["amoxicillin"],
        },
    },
    {
        "id": "mts-13",
        "transcript": "Spinning dizziness triggered when rolling over in bed, lasting 20 seconds. Positive Dix-Hallpike test on the right. Benign paroxysmal positional vertigo. Performed Epley maneuver; meclizine 25 mg as needed.",
        "gold": {
            "symptoms": ["spinning dizziness"],
            "diagnoses": ["benign paroxysmal positional vertigo"],
            "medications": ["meclizine"],
        },
    },
    {
        "id": "mts-14",
        "transcript": "Lower back pain radiating down the left leg past the knee after lifting boxes. Positive straight leg raise on left. Lumbar radiculopathy. Prescribe gabapentin 300 mg at bedtime and cyclobenzaprine 5 mg as needed.",
        "gold": {
            "symptoms": ["lower back pain", "radiating leg pain"],
            "diagnoses": ["lumbar radiculopathy"],
            "medications": ["gabapentin", "cyclobenzaprine"],
        },
    },
    {
        "id": "mts-15",
        "transcript": "Red, itchy, scaly plaques on bilateral elbows and knees for years. Plaque psoriasis. Apply triamcinolone 0.1 percent cream twice daily to affected areas.",
        "gold": {
            "symptoms": ["red itchy scaly plaques"],
            "diagnoses": ["plaque psoriasis"],
            "medications": ["triamcinolone"],
        },
    },
    {
        "id": "mts-16",
        "transcript": "Epigastric abdominal pain worse on an empty stomach, relieved by food. H pylori stool antigen positive. Peptic ulcer disease due to Helicobacter pylori. Start pantoprazole 40 mg, clarithromycin 500 mg, and amoxicillin 1000 mg.",
        "gold": {
            "symptoms": ["epigastric abdominal pain"],
            "diagnoses": ["peptic ulcer disease"],
            "medications": ["pantoprazole", "clarithromycin", "amoxicillin"],
        },
    },
    {
        "id": "mts-17",
        "transcript": "Bilateral ankle swelling and dyspnea on exertion for 1 week. BNP elevated, bibasilar rales. Acute decompensated heart failure. Give furosemide 40 mg once daily and restrict sodium.",
        "gold": {
            "symptoms": ["bilateral ankle swelling", "dyspnea on exertion"],
            "diagnoses": ["acute decompensated heart failure"],
            "medications": ["furosemide"],
        },
    },
    {
        "id": "mts-18",
        "transcript": "Right ear pain and muffled hearing after swimming. Tenderness on tragal pressure and edematous ear canal. Acute otitis externa. Prescribe ciprofloxacin-dexamethasone otic drops 4 drops twice daily.",
        "gold": {
            "symptoms": ["right ear pain", "muffled hearing"],
            "diagnoses": ["acute otitis externa"],
            "medications": ["ciprofloxacin-dexamethasone"],
        },
    },
    {
        "id": "mts-19",
        "transcript": "Persistent sadness, loss of interest in hobbies, and insomnia for 6 weeks. PHQ-9 score 17. Major depressive disorder. Start sertraline 50 mg daily and refer for cognitive behavioral therapy.",
        "gold": {
            "symptoms": ["persistent sadness", "loss of interest", "insomnia"],
            "diagnoses": ["major depressive disorder"],
            "medications": ["sertraline"],
        },
    },
    {
        "id": "mts-20",
        "transcript": "Left calf pain and swelling after a 10-hour flight. Doppler ultrasound confirms non-compressible left popliteal vein. Deep vein thrombosis. Start apixaban 10 mg twice daily for 7 days then 5 mg twice daily.",
        "gold": {
            "symptoms": ["left calf pain", "swelling"],
            "diagnoses": ["deep vein thrombosis"],
            "medications": ["apixaban"],
        },
    },
]


# 12 synthetic patients with longitudinal EHR records and known relevant record IDs for retrieval eval
RETRIEVAL_BENCHMARK_PATIENTS: list[dict] = [
    {
        "patient_id": "eval-p01",
        "name": "Patient 01",
        "dob": "1968-04-12",
        "sex": "M",
        "records": [
            {"id": "p01-r1", "type": "allergy", "days_ago": 900, "text": "ALLERGIES: Severe anaphylactic reaction to Penicillin and amoxicillin."},
            {"id": "p01-r2", "type": "visit", "days_ago": 45, "text": "ASSESSMENT: Recurrent acute bacterial sinusitis.\nPLAN: Treated previously with doxycycline with full resolution."},
            {"id": "p01-r3", "type": "visit", "days_ago": 200, "text": "ASSESSMENT: Routine annual wellness exam. Lipids normal."},
            {"id": "p01-r4", "type": "lab", "days_ago": 300, "text": "LABS: Complete blood count and basic metabolic panel within normal limits."},
        ],
        "query": "acute sinusitis penicillin allergy antibiotic choice",
        "relevant_record_ids": ["p01-r1", "p01-r2"],
    },
    {
        "patient_id": "eval-p02",
        "name": "Patient 02",
        "dob": "1959-09-23",
        "sex": "F",
        "records": [
            {"id": "p02-r1", "type": "diagnosis", "days_ago": 1100, "text": "PAST MEDICAL HISTORY: Chronic kidney disease stage 3b, baseline eGFR 38 mL/min."},
            {"id": "p02-r2", "type": "lab", "days_ago": 20, "text": "LABS: Serum creatinine 1.8 mg/dL, eGFR 36 mL/min, potassium 4.9 mmol/L."},
            {"id": "p02-r3", "type": "visit", "days_ago": 120, "text": "ASSESSMENT: Osteoarthritis of bilateral knees. Avoid NSAIDs due to renal impairment."},
            {"id": "p02-r4", "type": "visit", "days_ago": 400, "text": "ASSESSMENT: Upper respiratory viral illness, resolved."},
        ],
        "query": "knee pain NSAID renal function creatinine eGFR contraindication",
        "relevant_record_ids": ["p02-r1", "p02-r2", "p02-r3"],
    },
    {
        "patient_id": "eval-p03",
        "name": "Patient 03",
        "dob": "1974-02-17",
        "sex": "M",
        "records": [
            {"id": "p03-r1", "type": "lab", "days_ago": 15, "text": "LABS: HbA1c 9.1%, fasting glucose 214 mg/dL."},
            {"id": "p03-r2", "type": "lab", "days_ago": 550, "text": "LABS: HbA1c 6.8%, fasting glucose 132 mg/dL."},
            {"id": "p03-r3", "type": "visit", "days_ago": 30, "text": "ASSESSMENT: Type 2 diabetes mellitus with poor glycemic control.\nPLAN: Discussed adding GLP-1 agonist if HbA1c remains above 8%."},
            {"id": "p03-r4", "type": "visit", "days_ago": 250, "text": "ASSESSMENT: Mild tension headache, advised hydration."},
        ],
        "query": "type 2 diabetes HbA1c glycemic control medication adjustment",
        "relevant_record_ids": ["p03-r1", "p03-r3"],
    },
    {
        "patient_id": "eval-p04",
        "name": "Patient 04",
        "dob": "1981-11-05",
        "sex": "F",
        "records": [
            {"id": "p04-r1", "type": "allergy", "days_ago": 800, "text": "ALLERGIES: Lisinopril caused severe angioedema and lip swelling."},
            {"id": "p04-r2", "type": "visit", "days_ago": 25, "text": "ASSESSMENT: Essential hypertension, uncontrolled on hydrochlorothiazide alone."},
            {"id": "p04-r3", "type": "visit", "days_ago": 500, "text": "ASSESSMENT: Seasonal dermatitis, resolved with topical hydrocortisone."},
        ],
        "query": "hypertension blood pressure ACE inhibitor lisinopril angioedema",
        "relevant_record_ids": ["p04-r1", "p04-r2"],
    },
    {
        "patient_id": "eval-p05",
        "name": "Patient 05",
        "dob": "1963-06-30",
        "sex": "M",
        "records": [
            {"id": "p05-r1", "type": "diagnosis", "days_ago": 700, "text": "PAST MEDICAL HISTORY: Atrial fibrillation on chronic apixaban anticoagulation."},
            {"id": "p05-r2", "type": "visit", "days_ago": 18, "text": "ASSESSMENT: History of upper GI bleed secondary to peptic ulcer disease."},
            {"id": "p05-r3", "type": "lab", "days_ago": 14, "text": "LABS: Hemoglobin 10.4 g/dL, ferritin 18 ng/mL consistent with iron deficiency."},
            {"id": "p05-r4", "type": "visit", "days_ago": 310, "text": "ASSESSMENT: Cataract evaluation referral."},
        ],
        "query": "atrial fibrillation apixaban gastrointestinal bleed hemoglobin anemia",
        "relevant_record_ids": ["p05-r1", "p05-r2", "p05-r3"],
    },
    {
        "patient_id": "eval-p06",
        "name": "Patient 06",
        "dob": "1992-08-19",
        "sex": "F",
        "records": [
            {"id": "p06-r1", "type": "visit", "days_ago": 12, "text": "ASSESSMENT: Moderate persistent asthma.\nPLAN: Uses albuterol rescue inhaler 4 times weekly; needs daily inhaled corticosteroid."},
            {"id": "p06-r2", "type": "visit", "days_ago": 620, "text": "ASSESSMENT: Mild intermittent asthma, well controlled."},
            {"id": "p06-r3", "type": "allergy", "days_ago": 1200, "text": "ALLERGIES: Aspirin-exacerbated respiratory disease causing bronchospasm."},
            {"id": "p06-r4", "type": "visit", "days_ago": 180, "text": "ASSESSMENT: Sprained left ankle, recovered."},
        ],
        "query": "asthma exacerbation inhaler corticosteroid aspirin bronchospasm",
        "relevant_record_ids": ["p06-r1", "p06-r3"],
    },
    {
        "patient_id": "eval-p07",
        "name": "Patient 07",
        "dob": "1977-01-28",
        "sex": "F",
        "records": [
            {"id": "p07-r1", "type": "lab", "days_ago": 10, "text": "LABS: TSH 14.8 mIU/L, Free T4 0.6 ng/dL indicating undertreated hypothyroidism."},
            {"id": "p07-r2", "type": "visit", "days_ago": 95, "text": "ASSESSMENT: Hypothyroidism on levothyroxine 50 mcg; patient reports taking it with morning coffee and calcium."},
            {"id": "p07-r3", "type": "visit", "days_ago": 410, "text": "ASSESSMENT: Normal pap smear and gynecological exam."},
        ],
        "query": "hypothyroidism TSH levothyroxine absorption fatigue",
        "relevant_record_ids": ["p07-r1", "p07-r2"],
    },
    {
        "patient_id": "eval-p08",
        "name": "Patient 08",
        "dob": "1955-12-03",
        "sex": "M",
        "records": [
            {"id": "p08-r1", "type": "diagnosis", "days_ago": 1500, "text": "PAST MEDICAL HISTORY: Coronary artery disease status post drug-eluting stent to LAD."},
            {"id": "p08-r2", "type": "lab", "days_ago": 22, "text": "LABS: LDL cholesterol 138 mg/dL, above target of 70 mg/dL for secondary prevention."},
            {"id": "p08-r3", "type": "visit", "days_ago": 40, "text": "ASSESSMENT: Statin-associated myalgias on atorvastatin 40 mg; switched to rosuvastatin 10 mg."},
            {"id": "p08-r4", "type": "visit", "days_ago": 280, "text": "ASSESSMENT: Seborrheic keratosis on back, benign."},
        ],
        "query": "coronary artery disease LDL cholesterol statin myalgia rosuvastatin",
        "relevant_record_ids": ["p08-r1", "p08-r2", "p08-r3"],
    },
    {
        "patient_id": "eval-p09",
        "name": "Patient 09",
        "dob": "1986-05-14",
        "sex": "M",
        "records": [
            {"id": "p09-r1", "type": "visit", "days_ago": 8, "text": "ASSESSMENT: Recurrent gout flares (3 episodes in 6 months).\nPLAN: Candidate for urate-lowering therapy with allopurinol."},
            {"id": "p09-r2", "type": "lab", "days_ago": 8, "text": "LABS: Serum uric acid 9.4 mg/dL."},
            {"id": "p09-r3", "type": "visit", "days_ago": 500, "text": "ASSESSMENT: Acute pharyngitis, viral."},
        ],
        "query": "recurrent gout uric acid allopurinol urate lowering",
        "relevant_record_ids": ["p09-r1", "p09-r2"],
    },
    {
        "patient_id": "eval-p10",
        "name": "Patient 10",
        "dob": "1971-07-09",
        "sex": "F",
        "records": [
            {"id": "p10-r1", "type": "allergy", "days_ago": 1000, "text": "ALLERGIES: Sulfa drugs (trimethoprim-sulfamethoxazole) cause Stevens-Johnson-like blistering rash."},
            {"id": "p10-r2", "type": "visit", "days_ago": 35, "text": "ASSESSMENT: Recurrent uncomplicated urinary tract infections.\nPLAN: Urine culture grew E. coli sensitive to nitrofurantoin."},
            {"id": "p10-r3", "type": "visit", "days_ago": 220, "text": "ASSESSMENT: Routine dental clearance."},
        ],
        "query": "urinary tract infection sulfa allergy antibiotic sensitivity",
        "relevant_record_ids": ["p10-r1", "p10-r2"],
    },
    {
        "patient_id": "eval-p11",
        "name": "Patient 11",
        "dob": "1966-10-21",
        "sex": "M",
        "records": [
            {"id": "p11-r1", "type": "diagnosis", "days_ago": 850, "text": "PAST MEDICAL HISTORY: Heart failure with reduced ejection fraction (LVEF 35%)."},
            {"id": "p11-r2", "type": "visit", "days_ago": 14, "text": "ASSESSMENT: Compensated HFrEF on carvedilol and sacubitril-valsartan.\nPLAN: Monitor weight daily."},
            {"id": "p11-r3", "type": "lab", "days_ago": 14, "text": "LABS: NT-proBNP 620 pg/mL, potassium 4.4 mmol/L."},
            {"id": "p11-r4", "type": "visit", "days_ago": 360, "text": "ASSESSMENT: Seasonal influenza vaccination administered."},
        ],
        "query": "heart failure reduced ejection fraction BNP carvedilol fluid overload",
        "relevant_record_ids": ["p11-r1", "p11-r2", "p11-r3"],
    },
    {
        "patient_id": "eval-p12",
        "name": "Patient 12",
        "dob": "1989-03-30",
        "sex": "F",
        "records": [
            {"id": "p12-r1", "type": "visit", "days_ago": 28, "text": "ASSESSMENT: Chronic migraine without aura (10 headache days per month).\nPLAN: Failed propranolol due to fatigue; consider topiramate or CGRP inhibitor."},
            {"id": "p12-r2", "type": "visit", "days_ago": 480, "text": "ASSESSMENT: Episodic migraine, started propranolol prophylaxis."},
            {"id": "p12-r3", "type": "visit", "days_ago": 150, "text": "ASSESSMENT: Normal vision screening."},
        ],
        "query": "chronic migraine prophylaxis propranolol failure preventive therapy",
        "relevant_record_ids": ["p12-r1"],
    },
]