import os
import json
from openai import OpenAI
from pdfminer.high_level import extract_text

def analyze_with_llm(text):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    prompt = """
    You are a Resume Parser. Extract the following information from the resume text below and return it as a valid JSON object.
    
    Fields required:
    1. name (string)
    2. email (string)
    3. phone (string)
    4. education (list of objects): [{
        "degree": "string (e.g. Bachelor of Science, Diploma)", 
        "major": "string",
        "school": "string",
        "year": "string" 
    }]
    5. work_experience (list of objects): [{
        "role": "string",
        "company": "string",
        "duration": "string"
    }]
    6. skills (list of strings): Extract technical skills (programming languages, tools, frameworks).
    
    Return ONLY raw JSON. No markdown formatting.
    
    Resume Text:
    """ + text[:30000]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        content = response.choices[0].message.content
        # Clean response (sometimes LLM adds ```json ... ```)
        raw_json = content.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_json)
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def parse_resume(file_path: str):
    """
    Extracts text from PDF and uses DeepSeek to parse it.
    """
    try:
        text = extract_text(file_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

    print(f"Analyzing {file_path} with DeepSeek V3...")
    data = analyze_with_llm(text)
    
    if not data:
        return None
        
    return {
        "text_preview": text[:500] + "...",
        "email": data.get("email"),
        "phone": data.get("phone"),
        "skills": data.get("skills", []),
        "education": data.get("education", []),
        "work_experience": data.get("work_experience", []),
        "raw_text": text
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Check for API Key
        if "DEEPSEEK_API_KEY" not in os.environ:
             # Ask user for key if testing manually
             key = input("Enter DeepSeek API Key: ")
             os.environ["DEEPSEEK_API_KEY"] = key
             
        print(parse_resume(sys.argv[1]))
    else:
        print("Usage: python resume_parser.py <path_to_resume.pdf>")
