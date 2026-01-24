from pdfminer.high_level import extract_text
import re
import sys
import os

try:
    from .skill_extractor import extract_skills_from_text
except ImportError:
    try:
        from skill_extractor import extract_skills_from_text
    except ImportError:
        print("Warning: Could not import skill_extractor locally.")
        def extract_skills_from_text(text): return []

def parse_resume(file_path: str):
    """
    Extracts text from a PDF resume and identifies skills and contact info.
    """
    try:
        text = extract_text(file_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

    # 1. Extract Skills
    skills = extract_skills_from_text(text)
    
    # 2. Extract Email (Simple Regex)
    email = None
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = email_match.group(0)
        
    # 3. Extract Phone (Simple Regex for SG/International)
    phone = None
    # Matches +65 91234567, 9123 4567, etc.
    phone_match = re.search(r'(\+?65)?[ -]?\d{4}[ -]?\d{4}', text) 
    if phone_match:
        phone = phone_match.group(0)

    return {
        "text_preview": text[:500] + "...", # For debugging
        "email": email,
        "phone": phone,
        "skills": skills,
        "raw_text": text # In case we need it for the neural net later
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1]))
    else:
        print("Usage: python parser.py <path_to_resume.pdf>")
