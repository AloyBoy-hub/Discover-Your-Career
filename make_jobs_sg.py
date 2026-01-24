# Build jobs_sg.csv
import pandas as pd

OUT = "jobs_sg.csv"

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_job_text(title, company, location, salary, desc, extra=""):
    parts = [
        f"TITLE: {clean_text(title)}",
        f"COMPANY: {clean_text(company)}",
        f"LOCATION: {clean_text(location)}",
        f"SALARY: {clean_text(salary)}",
        extra.strip(),
        f"DESCRIPTION:\n{clean_text(desc)}",
    ]
    return "\n".join([p for p in parts if p])

frames = []

# Prosple
p = pd.read_csv("data/prosple.csv")
p["job_text"] = p.apply(lambda r: make_job_text(
    r.get("title",""), r.get("company",""), r.get("location",""),
    r.get("salary",""), r.get("description",""),
    f"HIRING CRITERIA:\n{clean_text(r.get('hiringCriteria',''))}"
), axis=1)
p_out = pd.DataFrame({
    "title": p["title"],
    "company": p["company"],
    "location": p["location"],
    "salary": p.get("salary",""),
    "url": p.get("url",""),
    "job_text": p["job_text"],
})
frames.append(p_out)

# MyCareersFuture
m = pd.read_csv("data/mycareersfuture.csv")
m["job_text"] = m.apply(lambda r: make_job_text(
    r.get("Title",""), r.get("Company",""), r.get("Location",""),
    r.get("Salary",""), r.get("Roles & Responsibilities",""),
    f"EMPLOYMENT TYPE: {clean_text(r.get('Employment Type',''))}\n"
    f"JOB LEVEL: {clean_text(r.get('Job Level',''))}\n"
    f"EXPERIENCE: {clean_text(r.get('Experience',''))}\n"
    f"CATEGORY: {clean_text(r.get('Category',''))}"
), axis=1)
m_out = pd.DataFrame({
    "title": m["Title"],
    "company": m["Company"],
    "location": m["Location"],
    "salary": m.get("Salary",""),
    "url": m.get("URL",""),
    "job_text": m["job_text"],
})
frames.append(m_out)

# JobStreet
j = pd.read_csv("data/jobstreet.csv")
j["job_text"] = j.apply(lambda r: make_job_text(
    r.get("title",""), r.get("company",""), r.get("location",""),
    r.get("salary",""), r.get("description",""),
    f"JOB TYPE: {clean_text(r.get('job_type',''))}\n"
    f"CATEGORY: {clean_text(r.get('category',''))}\n"
    f"POSTED: {clean_text(r.get('posted',''))}"
), axis=1)
j_out = pd.DataFrame({
    "title": j["title"],
    "company": j["company"],
    "location": j["location"],
    "salary": j.get("salary",""),
    "url": j.get("url",""),
    "job_text": j["job_text"],
})
frames.append(j_out)

# Indeed
i = pd.read_csv("data/indeed.csv")
i["job_text"] = i.apply(lambda r: make_job_text(
    r.get("title",""), r.get("company",""), r.get("location",""),
    "", r.get("description",""),
    f"JOB TYPE: {clean_text(r.get('job_type',''))}\n"
    f"BENEFITS:\n{clean_text(r.get('benefits',''))}"
), axis=1)
i_out = pd.DataFrame({
    "title": i["title"],
    "company": i["company"],
    "location": i["location"],
    "salary": "",
    "url": i.get("url",""),
    "job_text": i["job_text"],
})
frames.append(i_out)

# Glassdoor
g = pd.read_csv("data/glassdoor.csv")
g["job_text"] = g.apply(lambda r: make_job_text(
    r.get("title",""), r.get("company",""), r.get("location",""),
    r.get("salary",""), r.get("description",""),
    f"RATING: {clean_text(r.get('rating',''))}"
), axis=1)
g_out = pd.DataFrame({
    "title": g["title"],
    "company": g["company"],
    "location": g["location"],
    "salary": g.get("salary",""),
    "url": g.get("url",""),
    "job_text": g["job_text"],
})
frames.append(g_out)

# eFinancialCareers
e = pd.read_csv("data/efinancialcareers.csv")
e["job_text"] = e.apply(lambda r: make_job_text(
    r.get("title",""), r.get("company",""), r.get("location",""),
    r.get("salary",""), r.get("job_description",""),
    f"EMPLOYMENT TYPE: {clean_text(r.get('employment_type',''))}\n"
    f"WORKPLACE MODEL: {clean_text(r.get('workplace_model',''))}\n"
    f"POSTED: {clean_text(r.get('posted',''))}"
), axis=1)
e_out = pd.DataFrame({
    "title": e["title"],
    "company": e["company"],
    "location": e["location"],
    "salary": e.get("salary",""),
    "url": e.get("url",""),
    "job_text": e["job_text"],
})
frames.append(e_out)

all_jobs = pd.concat(frames, ignore_index=True)

# Create job_id
all_jobs["job_id"] = all_jobs.index.map(lambda k: f"sgjob_{k}")

all_jobs.to_csv(OUT, index=False)
print("Wrote:", OUT, "rows=", len(all_jobs))
