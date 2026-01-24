import re

# Comprehensive list of technical skills to search for
# Using a dictionary where Key = Normalized Name, Value = List of regex patterns or synonyms
# If value is None, we just match the key with word boundaries
SKILL_PATTERNS = {
    # Fuzzy Match (Substrings allowed)
    "Python": [r"python"],
    "Java": [r"java"], 
    "JavaScript": [r"javascript", r"\bjs\b"], # Keep \b for short alias
    "TypeScript": [r"typescript", r"\bts\b"],
    "HTML": [r"html", r"html5"],
    "CSS": [r"css", r"css3"],
    "SQL": [r"sql"], # e.g. "MySQL" matches "SQL" - acceptable for fuzzy
    "NoSQL": [r"nosql", r"mongo"],
    "React": [r"react"], 
    "Angular": [r"angular"],
    "Vue": [r"vue"],
    "Node.js": [r"node\.js", r"nodejs"],
    "Django": [r"django"],
    "Flask": [r"flask"],
    "Spring Boot": [r"spring boot", r"spring"],
    "AWS": [r"aws", r"amazon web services"],
    "Azure": [r"azure"],
    "GCP": [r"gcp", r"google cloud"],
    "Docker": [r"docker"],
    "Kubernetes": [r"kubernetes", r"k8s"],
    "Git": [r"git"],
    "CI/CD": [r"ci/cd", r"cicd"],
    "Linux": [r"linux"],
    "Machine Learning": [r"machine learning", r"ml\b"],
    "AI": [r"\bai\b", r"artificial intelligence"], # Keep \b for AI (Avoid 'said', 'email')
    "Deep Learning": [r"deep learning", r"dl\b"],
    "Data Science": [r"data science"],
    "Big Data": [r"big data"],
    "Hadoop": [r"hadoop"],
    "Spark": [r"spark"],
    "TensorFlow": [r"tensorflow"],
    "PyTorch": [r"pytorch"],
    "Excel": [r"excel"],
    "Tableau": [r"tableau"],
    "Power BI": [r"power bi", r"powerbi"],
    # Strict Match (Short/Common words)
    "C++": [r"\bc\+\+\b"],
    "C#": [r"\bc#\b", r"\bcsharp"],
    "Go": [r"\bgo\b", r"golang"],
    "Rust": [r"rust"],
    "Swift": [r"swift"],
    "Kotlin": [r"kotlin"],
    "Ruby": [r"ruby", r"rails"],
    "PHP": [r"php"],
    "R": [r"\bR\b"], 
}

def extract_skills_from_text(text: str):
    """
    Scans the text for known skills using regex patterns.
    Returns a list of unique skill names found.
    """
    if not text:
        return []
    
    found_skills = set()
    
    # Normalize text for easier matching (optional, but regex flags usually handle case)
    # We will use case-insensitive matching for most things, but be careful with "Go" or "R"
    
    for skill_name, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            # \b matches word boundary. 
            # Case insensitive search
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.add(skill_name)
                break # Found one pattern for this skill, no need to check others
    
    return list(found_skills)

if __name__ == "__main__":
    # Test
    sample = "We are looking for a Software Engineer with experience in Python, AWS, and React.js. Knowing SQL is a plus."
    print(f"Sample: {sample}")
    print(f"Extracted: {extract_skills_from_text(sample)}")
