"""
preprocessor.py
--------------
Handles ingestion and chunking for:
  - MedQuAD  (XML files, one per medical category)
  - DrugBank (CSV export or simplified XML)

Output: list of chunk dicts ready for embedding.

Chunk schema:
  {
    "chunk_id":   str,       # unique id
    "text":       str,       # text to embed / retrieve
    "source":     str,       # "medquad" | "drugbank"
    "category":   str,       # medical category or drug name
    "chunk_type": str,       # "qa" | "interaction" | "description" | "mechanism"
    "metadata":   dict       # extra fields (drug names, severity, etc.)
  }
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Optional

import tiktoken
from lxml import etree
from rapidfuzz import fuzz
from tqdm import tqdm

# ── Tokenizer for chunk sizing ───────────────────────────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")

def _token_len(text: str) -> int:
    return len(_ENC.encode(text))

def _chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── Text cleaning ─────────────────────────────────────────────────────────────
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


# ── Sliding-window chunker ────────────────────────────────────────────────────
def sliding_window_chunks(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[str]:
    """Split long text into overlapping token windows."""
    tokens = _ENC.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = _ENC.decode(tokens[start:end])
        chunks.append(chunk_text.strip())
        if end == len(tokens):
            break
        start += max_tokens - overlap_tokens
    return [c for c in chunks if len(c) > 40]


# ─────────────────────────────────────────────────────────────────────────────
# MedQuAD
# ─────────────────────────────────────────────────────────────────────────────
def load_medquad(medquad_dir: str) -> list[dict]:
    """
    Parse all MedQuAD XML files in a directory.
    Each file is a medical category (e.g. diabetes, cancer).
    Returns a flat list of chunk dicts.
    """
    chunks = []
    xml_files = list(Path(medquad_dir).rglob("*.xml"))

    if not xml_files:
        print(f"[MedQuAD] No XML files found in {medquad_dir}. "
              "Using built-in sample data for demo.")
        return _medquad_sample_data()

    print(f"[MedQuAD] Found {len(xml_files)} XML files.")

    for xml_file in tqdm(xml_files, desc="Parsing MedQuAD"):
        try:
            tree = etree.parse(str(xml_file))
            root = tree.getroot()

            # Detect category from filename or <Focus> tag
            focus_el = root.find(".//Focus")
            category = focus_el.text.strip() if focus_el is not None else xml_file.stem

            for qa_pair in root.findall(".//QAPair"):
                q_el = qa_pair.find("Question")
                a_el = qa_pair.find("Answer")
                if q_el is None or a_el is None:
                    continue

                question = clean_text(q_el.text or "")
                answer   = clean_text(a_el.text or "")

                if len(answer) < 60:          # skip trivially short answers
                    continue

                combined = f"Q: {question}\nA: {answer}"

                if _token_len(combined) <= 400:
                    chunks.append({
                        "chunk_id":   _chunk_id(combined),
                        "text":       combined,
                        "source":     "medquad",
                        "category":   category,
                        "chunk_type": "qa",
                        "metadata":   {"question": question},
                    })
                else:
                    # Long answer → split, prepend question to each window
                    for sub in sliding_window_chunks(answer):
                        text = f"Q: {question}\nA: {sub}"
                        chunks.append({
                            "chunk_id":   _chunk_id(text),
                            "text":       text,
                            "source":     "medquad",
                            "category":   category,
                            "chunk_type": "qa",
                            "metadata":   {"question": question},
                        })

        except Exception as e:
            print(f"[MedQuAD] Error parsing {xml_file}: {e}")

    # Deduplicate by similarity
    chunks = _deduplicate(chunks)
    print(f"[MedQuAD] {len(chunks)} chunks after deduplication.")
    return chunks


def _medquad_sample_data() -> list[dict]:
    """Fallback sample data when no XML files are present."""
    samples = [
        ("What is hypertension?", "Hypertension, or high blood pressure, is a condition where the force of the blood against artery walls is consistently too high. Normal blood pressure is below 120/80 mmHg. Hypertension is diagnosed when readings are consistently 130/80 mmHg or higher. It is often called the 'silent killer' because it typically has no symptoms but significantly increases the risk of heart disease and stroke."),
        ("How is diabetes managed?", "Diabetes management involves monitoring blood glucose levels, taking medications (insulin or oral hypoglycemics), following a healthy diet low in refined carbohydrates, regular physical activity, and regular checkups. Type 1 diabetes always requires insulin. Type 2 can sometimes be managed with lifestyle changes alone, though many patients need oral medications like metformin or injectable GLP-1 agonists."),
        ("What causes asthma?", "Asthma is caused by inflammation and narrowing of the airways. Triggers include allergens (pollen, dust mites, mold, pet dander), respiratory infections, air pollution, exercise, cold air, and smoke. Genetic factors and early childhood exposure play a role. The airways become hypersensitive and react to stimuli that don't affect most people."),
        ("What are symptoms of heart failure?", "Heart failure symptoms include shortness of breath (especially when lying down or during activity), fatigue, swollen legs and ankles, rapid or irregular heartbeat, reduced ability to exercise, persistent cough or wheezing, and difficulty concentrating. Symptoms worsen as the condition progresses."),
        ("What is COPD?", "Chronic Obstructive Pulmonary Disease (COPD) is a chronic lung disease characterized by obstructed airflow. It includes chronic bronchitis and emphysema. The main cause is long-term exposure to irritating gases or particulate matter, most often cigarette smoke. Symptoms include breathing difficulty, cough, mucus production, and wheezing."),
        ("What is arthritis?", "Arthritis refers to inflammation of one or more joints, causing pain and stiffness. The most common types are osteoarthritis (wear-and-tear) and rheumatoid arthritis (autoimmune). Symptoms include joint pain, swelling, stiffness, and reduced range of motion. Treatment includes medications, physical therapy, and in severe cases, surgery."),
        ("What is depression?", "Depression is a mood disorder that causes persistent feelings of sadness, hopelessness, and loss of interest. It affects how a person thinks, feels, and handles daily activities. Symptoms include persistent sadness, loss of energy, changes in appetite and sleep, difficulty concentrating, and thoughts of suicide. Treatment involves therapy, medication (antidepressants), or a combination."),
        ("What is Alzheimer's disease?", "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It is the most common cause of dementia. Symptoms begin with mild memory loss and progress to severe cognitive impairment. Early signs include forgetting recent events, confusion, and getting lost in familiar places."),
    ]
    chunks = []
    for q, a in samples:
        text = f"Q: {q}\nA: {a}"
        chunks.append({
            "chunk_id":   _chunk_id(text),
            "text":       text,
            "source":     "medquad",
            "category":   "general_medicine",
            "chunk_type": "qa",
            "metadata":   {"question": q},
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# DrugBank
# ─────────────────────────────────────────────────────────────────────────────
_SEVERITY_PATTERN = re.compile(
    r"\b(contraindicated|major|moderate|minor|severe|serious|avoid|caution)\b",
    re.IGNORECASE,
)

def _extract_severity(text: str) -> str:
    m = _SEVERITY_PATTERN.search(text)
    return m.group(1).lower() if m else "unknown"


def load_drugbank_csv(csv_path: str) -> list[dict]:
    """
    Parse DrugBank CSV export.
    Expected columns (flexible): Name, Description, Indication,
    Mechanism of Action, Drug Interactions, Pharmacodynamics,
    Protein Binding, Half-life, Toxicity.
    """
    import pandas as pd

    if not Path(csv_path).exists():
        print(f"[DrugBank] CSV not found at {csv_path}. Using sample data.")
        return _drugbank_sample_data()

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    chunks = []
    text_cols = ["description", "indication", "mechanism_of_action",
                 "pharmacodynamics", "toxicity"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing DrugBank"):
        drug_name = str(row.get("name", "Unknown")).strip()

        for col in text_cols:
            if col not in row:
                continue
            val = str(row.get(col, "")).strip()
            if not val or val.lower() in ("nan", "not available", ""):
                continue
            val = clean_text(val)
            if len(val) < 40:
                continue

            text = f"Drug: {drug_name}\n{col.replace('_', ' ').title()}: {val}"
            chunks.append({
                "chunk_id":   _chunk_id(text),
                "text":       text,
                "source":     "drugbank",
                "category":   drug_name,
                "chunk_type": col,
                "metadata":   {"drug_name": drug_name, "field": col},
            })

        # Drug interactions — split on semicolons or newlines
        interactions_raw = str(row.get("drug_interactions", "")).strip()
        if interactions_raw and interactions_raw.lower() not in ("nan", ""):
            interactions_raw = clean_text(interactions_raw)
            entries = re.split(r"[;\n]+", interactions_raw)
            for entry in entries:
                entry = entry.strip()
                if len(entry) < 20:
                    continue
                text = f"Drug: {drug_name}\nInteraction: {entry}"
                severity = _extract_severity(entry)
                chunks.append({
                    "chunk_id":   _chunk_id(text),
                    "text":       text,
                    "source":     "drugbank",
                    "category":   drug_name,
                    "chunk_type": "interaction",
                    "metadata":   {
                        "drug_name": drug_name,
                        "severity":  severity,
                    },
                })

    chunks = _deduplicate(chunks)
    print(f"[DrugBank] {len(chunks)} chunks after deduplication.")
    return chunks


def _drugbank_sample_data() -> list[dict]:
    """Fallback sample drug data for demo."""
    drug_data = [
        {
            "name": "Metformin",
            "description": "Metformin is a biguanide antidiabetic used as a first-line treatment for type 2 diabetes. It works by decreasing hepatic glucose production, decreasing intestinal glucose absorption, and increasing insulin sensitivity.",
            "indication": "Type 2 diabetes mellitus, polycystic ovary syndrome (off-label), prediabetes prevention.",
            "mechanism": "Activates AMP-activated protein kinase (AMPK), which reduces gluconeogenesis in the liver and increases glucose uptake in peripheral tissues.",
            "interactions": [
                ("alcohol", "Increased risk of lactic acidosis when combined with alcohol. Avoid excessive alcohol use.", "major"),
                ("contrast_dye", "Iodinated contrast agents may cause acute kidney injury, increasing metformin accumulation and risk of lactic acidosis. Withhold metformin before and after contrast procedures.", "major"),
                ("cimetidine", "Cimetidine inhibits renal tubular secretion of metformin, increasing metformin plasma levels by up to 40%.", "moderate"),
            ],
            "toxicity": "Rare but serious risk of lactic acidosis, especially in renal impairment. Contraindicated in eGFR <30 mL/min/1.73m².",
        },
        {
            "name": "Lisinopril",
            "description": "Lisinopril is an ACE inhibitor used to treat hypertension, heart failure, and to improve survival after myocardial infarction.",
            "indication": "Hypertension, heart failure, post-MI, diabetic nephropathy.",
            "mechanism": "Inhibits angiotensin-converting enzyme (ACE), reducing conversion of angiotensin I to angiotensin II, leading to vasodilation and reduced aldosterone secretion.",
            "interactions": [
                ("potassium_supplements", "Concurrent use can cause dangerous hyperkalemia. Monitor potassium levels closely.", "major"),
                ("nsaids", "NSAIDs such as ibuprofen can reduce the antihypertensive effect of lisinopril and increase risk of acute kidney injury.", "moderate"),
                ("lithium", "ACE inhibitors reduce lithium excretion, increasing risk of lithium toxicity. Monitor lithium levels.", "major"),
                ("aliskiren", "Dual blockade of renin-angiotensin system increases risk of renal impairment, hypotension, and hyperkalemia. Contraindicated in diabetes.", "contraindicated"),
            ],
            "toxicity": "Can cause severe hypotension (first dose), angioedema (potentially life-threatening), hyperkalemia, and renal failure.",
        },
        {
            "name": "Atorvastatin",
            "description": "Atorvastatin is a statin used to lower cholesterol and reduce the risk of cardiovascular disease.",
            "indication": "Hypercholesterolemia, mixed dyslipidemia, cardiovascular risk reduction.",
            "mechanism": "Inhibits HMG-CoA reductase, the rate-limiting enzyme in cholesterol biosynthesis in the liver, reducing LDL cholesterol.",
            "interactions": [
                ("clarithromycin", "CYP3A4 inhibitors like clarithromycin increase atorvastatin plasma levels significantly, increasing risk of myopathy and rhabdomyolysis.", "major"),
                ("grapefruit_juice", "Grapefruit juice inhibits CYP3A4 and can increase atorvastatin levels. Avoid large quantities.", "moderate"),
                ("gemfibrozil", "Combination with fibrates increases risk of myopathy and rhabdomyolysis. Avoid if possible; use with caution and monitor CK levels.", "major"),
                ("warfarin", "Atorvastatin may enhance the anticoagulant effect of warfarin. Monitor INR when starting, stopping, or changing doses.", "moderate"),
            ],
            "toxicity": "Myopathy, rhabdomyolysis (rare but serious), elevated liver enzymes, new-onset diabetes.",
        },
        {
            "name": "Warfarin",
            "description": "Warfarin is an anticoagulant used to prevent blood clots. It requires careful monitoring due to a narrow therapeutic index and numerous drug-food interactions.",
            "indication": "Atrial fibrillation, deep vein thrombosis, pulmonary embolism, mechanical heart valves.",
            "mechanism": "Inhibits vitamin K epoxide reductase, preventing regeneration of active vitamin K needed for synthesis of clotting factors II, VII, IX, and X.",
            "interactions": [
                ("aspirin", "Aspirin increases bleeding risk when combined with warfarin. Major interaction — use lowest effective aspirin dose and monitor INR.", "major"),
                ("amiodarone", "Amiodarone inhibits CYP2C9 metabolism of warfarin, dramatically increasing INR. Requires 30-50% warfarin dose reduction and close monitoring.", "major"),
                ("fluconazole", "Strong CYP2C9 inhibitor — substantially increases warfarin effect. INR can double or triple. Major dose adjustment needed.", "major"),
                ("vitamin_k", "Vitamin K in foods (spinach, kale, broccoli) reduces warfarin effectiveness. Maintain consistent dietary intake.", "moderate"),
                ("nsaids", "All NSAIDs increase bleeding risk with warfarin through antiplatelet effects and GI irritation.", "major"),
            ],
            "toxicity": "Bleeding (major risk), including intracranial hemorrhage. Reversed with vitamin K or prothrombin complex concentrate.",
        },
        {
            "name": "Ibuprofen",
            "description": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation.",
            "indication": "Mild to moderate pain, fever, osteoarthritis, rheumatoid arthritis, dysmenorrhea.",
            "mechanism": "Non-selective COX-1 and COX-2 inhibitor. Reduces prostaglandin synthesis, leading to decreased inflammation, pain, and fever.",
            "interactions": [
                ("warfarin", "Increases bleeding risk through antiplatelet effects and GI irritation. Major interaction.", "major"),
                ("lisinopril", "NSAIDs reduce effectiveness of ACE inhibitors and increase risk of renal impairment, especially in elderly or dehydrated patients.", "moderate"),
                ("lithium", "NSAIDs reduce renal lithium clearance, increasing lithium levels and toxicity risk. Avoid or monitor closely.", "major"),
                ("methotrexate", "NSAIDs reduce methotrexate renal excretion, increasing toxicity risk significantly.", "major"),
            ],
            "toxicity": "GI ulceration and bleeding, renal impairment, cardiovascular events (MI, stroke) with long-term use, hypertension.",
        },
    ]

    chunks = []
    for drug in drug_data:
        name = drug["name"]

        for field, content in [("description", drug["description"]),
                                ("indication", drug["indication"]),
                                ("mechanism", drug["mechanism"]),
                                ("toxicity", drug["toxicity"])]:
            text = f"Drug: {name}\n{field.title()}: {content}"
            chunks.append({
                "chunk_id":   _chunk_id(text),
                "text":       text,
                "source":     "drugbank",
                "category":   name,
                "chunk_type": field,
                "metadata":   {"drug_name": name, "field": field},
            })

        for other_drug, interaction_text, severity in drug["interactions"]:
            text = f"Drug: {name}\nInteraction with {other_drug}: {interaction_text}"
            chunks.append({
                "chunk_id":   _chunk_id(text),
                "text":       text,
                "source":     "drugbank",
                "category":   name,
                "chunk_type": "interaction",
                "metadata":   {
                    "drug_name":      name,
                    "interacts_with": other_drug,
                    "severity":       severity,
                },
            })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def _deduplicate(chunks: list[dict], threshold: int = 92) -> list[dict]:
    """Remove near-duplicate chunks using fuzzy matching on the first 200 chars."""
    seen_texts = []
    deduped = []
    for chunk in chunks:
        preview = chunk["text"][:200]
        is_dup = any(fuzz.ratio(preview, s) >= threshold for s in seen_texts)
        if not is_dup:
            seen_texts.append(preview)
            deduped.append(chunk)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_corpus(
    medquad_dir: Optional[str] = None,
    drugbank_csv: Optional[str] = None,
) -> list[dict]:
    """Load and merge all data sources into a single chunk list."""
    all_chunks = []

    mq_dir = medquad_dir or os.getenv("MEDQUAD_PATH", "")
    if mq_dir:
        all_chunks.extend(load_medquad(mq_dir))
    else:
        print("[Corpus] No MEDQUAD_PATH set — loading sample MedQuAD data.")
        all_chunks.extend(_medquad_sample_data())

    db_csv = drugbank_csv or os.getenv("DRUGBANK_CSV_PATH", "")
    if db_csv:
        all_chunks.extend(load_drugbank_csv(db_csv))
    else:
        print("[Corpus] No DRUGBANK_CSV_PATH set — loading sample DrugBank data.")
        all_chunks.extend(_drugbank_sample_data())

    print(f"[Corpus] Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = build_corpus()
    out = Path("./embeddings")
    out.mkdir(exist_ok=True)
    with open(out / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to embeddings/chunks.json")
