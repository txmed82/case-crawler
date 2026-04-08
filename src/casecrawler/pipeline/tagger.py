from __future__ import annotations

import re

from casecrawler.models.document import Chunk

SPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "neurosurgery": [
        "neurosurg", "craniotomy", "craniectomy", "laminectomy", "spinal fusion",
        "deep brain stimulation", "ventriculostomy", "neurosurgical",
    ],
    "neurology": [
        "neurology", "neurolog", "epilepsy", "seizure", "stroke", "multiple sclerosis",
        "parkinson", "dementia", "alzheimer", "neuropathy", "migraine",
        "meningitis", "encephalitis", "brain tumor", "glioblastoma",
    ],
    "cardiology": [
        "cardiology", "cardiolog", "myocardial infarction", "heart failure",
        "atrial fibrillation", "coronary artery", "echocardiogram", "cardiac",
        "arrhythmia", "pacemaker", "stent", "angioplasty",
    ],
    "cardiothoracic_surgery": [
        "cardiothoracic", "cardiac surgery", "bypass graft", "valve replacement",
        "thoracotomy", "sternotomy", "aortic dissection",
    ],
    "orthopedics": [
        "orthopedic", "orthopaedic", "fracture", "arthroplasty", "joint replacement",
        "osteotomy", "meniscus", "ligament", "tendon repair", "bone graft",
    ],
    "oncology": [
        "oncology", "oncolog", "chemotherapy", "radiation therapy", "radiotherapy",
        "tumor", "cancer", "malignancy", "metastasis", "lymphoma", "leukemia",
        "carcinoma", "sarcoma", "biopsy",
    ],
    "emergency_medicine": [
        "emergency medicine", "emergency department", "trauma", "resuscitation",
        "triage", "acute care", "emergency physician",
    ],
    "critical_care": [
        "critical care", "intensive care", "icu", "mechanical ventilation",
        "sepsis", "vasopressor", "hemodynamic", "organ failure",
    ],
    "pulmonology": [
        "pulmonology", "pulmonolog", "respiratory", "asthma", "copd", "pneumonia",
        "bronchitis", "pulmonary fibrosis", "pulmonary embolism", "lung",
    ],
    "gastroenterology": [
        "gastroenterology", "gastroenterolog", "inflammatory bowel", "crohn",
        "ulcerative colitis", "colonoscopy", "endoscopy", "hepatitis", "cirrhosis",
        "pancreatitis", "gastrointestinal",
    ],
    "nephrology": [
        "nephrology", "nephrolog", "kidney", "renal", "dialysis", "glomerulonephritis",
        "nephrotic syndrome", "chronic kidney disease",
    ],
    "endocrinology": [
        "endocrinology", "endocrinolog", "diabetes", "thyroid", "insulin",
        "hyperthyroidism", "hypothyroidism", "adrenal", "pituitary", "hormone",
    ],
    "infectious_disease": [
        "infectious disease", "infection", "antibiotic", "antimicrobial",
        "bacteremia", "fungal", "viral", "hiv", "tuberculosis", "malaria",
        "septicemia",
    ],
    "psychiatry": [
        "psychiatry", "psychiatr", "depression", "anxiety", "schizophrenia",
        "bipolar", "psychosis", "antidepressant", "antipsychotic", "mental health",
    ],
    "pediatrics": [
        "pediatric", "paediatric", "neonatal", "infant", "child", "childhood",
        "adolescent", "congenital",
    ],
    "obstetrics_gynecology": [
        "obstetrics", "gynecology", "gynaecology", "pregnancy", "prenatal",
        "postpartum", "cesarean", "ovarian", "uterine", "endometriosis",
    ],
    "anesthesiology": [
        "anesthesiology", "anesthesia", "anaesthesia", "sedation", "intubation",
        "regional anesthesia", "epidural", "general anesthesia",
    ],
    "radiology": [
        "radiology", "radiolog", "mri", "ct scan", "x-ray", "ultrasound",
        "imaging", "interventional radiology", "fluoroscopy",
    ],
    "dermatology": [
        "dermatology", "dermatolog", "skin", "eczema", "psoriasis", "melanoma",
        "dermatitis", "rash", "acne",
    ],
    "ophthalmology": [
        "ophthalmology", "ophthalmolog", "retinal", "glaucoma", "cataract",
        "intraocular", "vitrectomy", "corneal",
    ],
    "urology": [
        "urology", "urolog", "prostate", "bladder", "kidney stone", "nephrectomy",
        "cystoscopy", "urinary", "incontinence",
    ],
}


class Tagger:
    def tag(self, chunk: Chunk) -> Chunk:
        text_lower = chunk.text.lower()
        found: set[str] = set(chunk.metadata.specialty)

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            for keyword in keywords:
                if re.search(re.escape(keyword), text_lower):
                    found.add(specialty)
                    break  # no need to check more keywords for this specialty

        if found == set(chunk.metadata.specialty):
            return chunk  # no changes

        return chunk.model_copy(update={"metadata": chunk.metadata.model_copy(update={"specialty": sorted(found)})})

    def tag_all(self, chunks: list[Chunk]) -> list[Chunk]:
        return [self.tag(c) for c in chunks]
