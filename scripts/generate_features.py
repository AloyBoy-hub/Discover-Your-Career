# extract_features.py
from __future__ import annotations

import re
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.skill_service import extract_skills_from_text, SKILL_PATTERNS  # noqa: E402

# Your majors list (multi-valued)
MAJORS = [
    "Computer Science", "Information Technology", "Software Engineering", "Data Science",
    "Business", "Business Administration", "Finance", "Accounting", "Marketing", "Economics",
    "Management", "Psychology", "Sociology", "Political Science", "History", "English",
    "Communications", "Journalism", "Graphic Design", "Fine Arts",
    "Engineering", "Mechanical Engineering", "Electrical Engineering", "Civil Engineering",
    "Chemical Engineering", "Biomedical Engineering",
    "Biology", "Chemistry", "Physics", "Mathematics", "Statistics",
    "Nursing", "Medicine", "Pharmacy", "Law"
]


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Fixed vocabularies (stable ordering)
SKILL_NAMES = sorted(SKILL_PATTERNS.keys())
SKILL_COLS = [f"skill_{slugify(s)}" for s in SKILL_NAMES]

MAJOR_NAMES = sorted(MAJORS)
MAJOR_COLS = [f"major_{slugify(m)}" for m in MAJOR_NAMES]


def extract_years_experience(text: str) -> float:
    text = normalize(text)
    patterns = [
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+of\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)",
    ]
    vals: List[float] = []
    for p in patterns:
        for m in re.finditer(p, text):
            try:
                vals.append(float(m.group(1)))
            except Exception:
                pass
    return float(max(vals)) if vals else 0.0


def extract_edu_level(text: str) -> int:
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


def extract_majors(text: str) -> List[str]:
    t = normalize(text)
    found = []
    for m in MAJOR_NAMES:
        if m.lower() in t:
            found.append(m)
    return list(set(found))


def skill_multihot(text: str) -> np.ndarray:
    found = set(extract_skills_from_text(text))
    v = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    for i, s in enumerate(SKILL_NAMES):
        if s in found:
            v[i] = 1.0
    return v


def major_multihot(text: str) -> np.ndarray:
    found = set(extract_majors(text))
    v = np.zeros(len(MAJOR_NAMES), dtype=np.float32)
    for i, m in enumerate(MAJOR_NAMES):
        if m in found:
            v[i] = 1.0
    return v


def main():
    candidates = pd.read_csv("data/candidates.csv")  # candidate_id, resume_text
    rows = []

    for _, r in candidates.iterrows():
        cid = str(r["candidate_id"])
        txt = r.get("resume_text", "")
        txt = "" if not isinstance(txt, str) else txt

        years = extract_years_experience(txt)
        edu = float(extract_edu_level(txt))
        skills = skill_multihot(txt)
        majors = major_multihot(txt)

        row: Dict[str, float] = {
            "candidate_id": cid,
            "years_experience": float(years),
            "edu_level": float(edu),
        }
        for i, col in enumerate(SKILL_COLS):
            row[col] = float(skills[i])
        for i, col in enumerate(MAJOR_COLS):
            row[col] = float(majors[i])

        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv("data/candidate_features.csv", index=False)
    print(f"Saved data/candidate_features.csv shape={feat_df.shape}")


if __name__ == "__main__":
    main()
