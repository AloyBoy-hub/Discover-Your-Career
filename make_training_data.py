import pandas as pd

IN_PATH = "data/dataset.csv"   # your AI recruitment pipeline dataset
OUT_CAND = "data/candidates.csv"
OUT_JOBS = "data/jobs.csv"
OUT_PAIRS = "data/pairs.csv"

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

df = pd.read_csv(IN_PATH)

# --- Create stable IDs per row (simplest & safest for this dataset) ---
df["candidate_id"] = df.index.map(lambda i: f"cand_{i}")
df["job_id"] = df.index.map(lambda i: f"job_{i}")

# --- Label mapping ---
df["label"] = df["decision"].map({"select": 1, "reject": 0})
df = df.dropna(subset=["label"])  # safety

# --- Build candidate text ---
# You can tune this string format later; this is a solid start.
df["resume_text"] = (
    "NAME: " + df["Name"].apply(clean_text) + "\n"
    "RESUME:\n" + df["Resume"].apply(clean_text) + "\n\n"
    "INTERVIEW TRANSCRIPT:\n" + df["Transcript"].apply(clean_text)
)

# --- Build job text ---
df["job_text"] = (
    "ROLE TITLE: " + df["Role"].apply(clean_text) + "\n"
    "JOB DESCRIPTION:\n" + df["Job_Description"].apply(clean_text)
)

# --- Export candidates.csv ---
candidates = df[["candidate_id", "resume_text"]].copy()
candidates.to_csv(OUT_CAND, index=False)

# --- Export jobs.csv ---
jobs = df[["job_id", "Role", "job_text"]].rename(columns={"Role": "title"}).copy()
jobs.to_csv(OUT_JOBS, index=False)

# --- Export pairs.csv ---
pairs = df[["candidate_id", "job_id", "label"]].copy()
pairs.to_csv(OUT_PAIRS, index=False)

print("Wrote:", OUT_CAND, OUT_JOBS, OUT_PAIRS)
print("Label counts:\n", pairs["label"].value_counts())
