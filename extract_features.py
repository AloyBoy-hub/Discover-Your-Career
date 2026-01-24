# Creates candidate_features.csv
import re
import pandas as pd
import numpy as np

SKILLS = [
    # programming
    "python", "java", "c++", "javascript", "typescript", "go", "sql",
    # data/ml
    "pytorch", "tensorflow", "sklearn", "scikit-learn", "pandas", "numpy",
    "llm", "nlp",
    # cloud/devops
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    # tools
    "git", "linux", "excel",
]

EDU_LEVELS = ["phd", "doctor", "master", "bachelor", "diploma", "polytechnic"]

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()

def extract_years_experience(text: str) -> float:
    """
    Very rough: finds patterns like '3 years', '5+ years', '2 yrs'
    Returns max found, else 0.
    """
    text = normalize(text)
    patterns = [
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+of\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)",
    ]
    vals = []
    for p in patterns:
        for m in re.finditer(p, text):
            try:
                vals.append(float(m.group(1)))
            except:
                pass
    return float(max(vals)) if vals else 0.0

def extract_edu_level(text: str) -> int:
    """
    Ordinal encoding:
    4 = PhD/Doctorate
    3 = Master
    2 = Bachelor
    1 = Diploma/Poly
    0 = unknown
    """
    text = normalize(text)
    if "phd" in text or "doctor" in text or "doctorate" in text:
        return 4
    if "master" in text or "msc" in text or "m.s." in text:
        return 3
    if "bachelor" in text or "bsc" in text or "b.s." in text or "undergraduate" in text:
        return 2
    if "diploma" in text or "polytechnic" in text:
        return 1
    return 0

def skill_vector(text: str) -> np.ndarray:
    t = normalize(text)
    vec = []
    for s in SKILLS:
        # basic containment; you can upgrade to word-boundary matching later
        vec.append(1.0 if s in t else 0.0)
    return np.array(vec, dtype=np.float32)

def main():
    candidates = pd.read_csv("data/candidates.csv")  # candidate_id, resume_text
    rows = []

    for _, r in candidates.iterrows():
        cid = r["candidate_id"]
        txt = r.get("resume_text", "")
        years = extract_years_experience(txt)
        edu = extract_edu_level(txt)
        skills = skill_vector(txt)

        row = {
            "candidate_id": cid,
            "years_experience": years,
            "edu_level": float(edu),
        }
        for i, s in enumerate(SKILLS):
            row[f"skill_{s}"] = float(skills[i])

        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv("data/candidate_features.csv", index=False)
    print(f"Saved candidate_features.csv with shape: {feat_df.shape}")

if __name__ == "__main__":
    main()
