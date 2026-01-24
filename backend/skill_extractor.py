# backend/skill_extractor.py
import re

SKILL_PATTERNS = {
    "Python": [r"python"],
    "Java": [r"java"],
    "JavaScript": [r"javascript", r"\bjs\b"],
    "TypeScript": [r"typescript", r"\bts\b"],
    "HTML": [r"html", r"html5"],
    "CSS": [r"css", r"css3"],
    "SQL": [r"sql"],
    "NoSQL": [r"nosql", r"mongo"],
    "React": [r"react"],
    "Angular": [r"angular"],
    "Vue": [r"vue"],
    "Node.js": [r"node\.js", r"nodejs"],
    "Django": [r"django"],
    "Flask": [r"flask"],
    "Spring Boot": [r"spring boot", r"\bspring\b"],
    "AWS": [r"aws", r"amazon web services"],
    "Azure": [r"azure"],
    "GCP": [r"gcp", r"google cloud"],
    "Docker": [r"docker"],
    "Kubernetes": [r"kubernetes", r"k8s"],
    "Git": [r"\bgit\b"],
    "CI/CD": [r"ci/cd", r"cicd"],
    "Linux": [r"linux"],
    "Machine Learning": [r"machine learning", r"\bml\b"],
    "AI": [r"\bai\b", r"artificial intelligence"],
    "Deep Learning": [r"deep learning", r"\bdl\b"],
    "Data Science": [r"data science"],
    "Big Data": [r"big data"],
    "Hadoop": [r"hadoop"],
    "Spark": [r"\bspark\b"],
    "TensorFlow": [r"tensorflow"],
    "PyTorch": [r"pytorch"],
    "Excel": [r"excel"],
    "Tableau": [r"tableau"],
    "Power BI": [r"power bi", r"powerbi"],
    "C++": [r"\bc\+\+\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "Go": [r"\bgo\b", r"golang"],
    "Rust": [r"\brust\b"],
    "Swift": [r"\bswift\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Ruby": [r"\bruby\b", r"\brails\b"],
    "PHP": [r"\bphp\b"],
    "R": [r"\br\b"],
}


def extract_skills_from_text(text: str):
    if not text:
        return []
    found = set()
    for skill_name, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(skill_name)
                break
    return list(found)
