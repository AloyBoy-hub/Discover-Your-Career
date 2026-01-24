import re

# Comprehensive list of technical skills to search for
# Using a dictionary where Key = Normalized Name, Value = List of regex patterns or synonyms
# If value is None, we just match the key with word boundaries
SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b"], # careful not to match javascript with just java regex usually works if ordered
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "HTML": [r"\bhtml\b", r"\bhtml5\b"],
    "CSS": [r"\bcss\b", r"\bcss3\b"],
    "SQL": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bpostgres\b"],
    "NoSQL": [r"\bnosql\b", r"\bmongo\b", r"\bmongodb\b"],
    "React": [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"],
    "Angular": [r"\bangular\b", r"\bangularjs\b"],
    "Vue": [r"\bvue\b", r"\bvue\.js\b", r"\bvuejs\b"],
    "Node.js": [r"\bnode\.js\b", r"\bnodejs\b", r"\bnode\b"],
    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "Spring Boot": [r"\bspring boot\b", r"\bspring\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Git": [r"\bgit\b"],
    "CI/CD": [r"\bci/cd\b", r"\bcicd\b"],
    "Linux": [r"\blinux\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b"],
    "AI": [r"\bai\b", r"\bartificial intelligence\b"],
    "Deep Learning": [r"\bdeep learning\b", r"\bdl\b"],
    "Data Science": [r"\bdata science\b"],
    "Big Data": [r"\bbig data\b"],
    "Hadoop": [r"\bhadoop\b"],
    "Spark": [r"\bspark\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Excel": [r"\bexcel\b"],
    "Tableau": [r"\btableau\b"],
    "Power BI": [r"\bpower bi\b", r"\bpowerbi\b"],
    "C++": [r"\bc\+\+\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "Go": [r"\bgo\b", r"\bgolang\b"],
    "Rust": [r"\brust\b"],
    "Swift": [r"\bswift\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Ruby": [r"\bruby\b", r"\brails\b"],
    "PHP": [r"\bphp\b"],
    "R": [r"\bR\b"], # tricky, likely false positives, but capital R surrounded by boundaries might be ok.
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
