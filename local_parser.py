import spacy
import pandas as pd
import sys
import os
from tqdm import tqdm

# Ensure we can import from backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from backend.skill_extractor import extract_skills_from_text
except ImportError:
    # Fallback if run from wrong dir
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
    from skill_extractor import extract_skills_from_text

# Common Majors List (Expanded)
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

def extract_majors(text):
    found = []
    text_lower = text.lower()
    
    # 1. Direct Keyword Match
    for major in MAJORS:
        # Check for word boundary to avoid "Biology" matching "Microbiology" incorrectly without context if simplistic
        # Simple string check is usually fine for majors
        if major.lower() in text_lower:
            found.append(major)
            
    # 2. Regex for "Bachelor of [X]" types (Optional refinement)
    # This is harder to normalize without a huge dict, so for now we rely on the list.
    
    return list(set(found))

def parse_resumes_local(input_csv, output_csv):
    print(f"Loading SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model not found. Please run: python3 -m spacy download en_core_web_sm")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # We will store results here
    parsed_data = []

    print(f"Processing sample of {len(df)} resumes...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row.get('resume_text', ''))
        
        # 1. Major Extraction
        majors = extract_majors(text)
        
        # 2. Skill Extraction
        skills = extract_skills_from_text(text)
        
        parsed_data.append({
            "candidate_id": row.get('candidate_id', idx),
            "majors": majors, 
            "skills": skills
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(parsed_data)
    
    # Save
    result_df.to_csv(output_csv, index=False)
    print(f"Done! Parsed data saved to {output_csv}")
    print(result_df.head())

if __name__ == "__main__":
    input_file = "data/candidates.csv"
    output_file = "data/candidates_parsed.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    parse_resumes_local(input_file, output_file)
