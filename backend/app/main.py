# app.py
from __future__ import annotations

import json
import os
import re
import sys
import shutil
import tempfile
import difflib
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from backend.ml.two_tower import TwoTower
from backend.ml.reranker import Reranker
from backend.utils.io_helpers import load_json, load_npy
from backend.services.roadmap_service import generate_roadmap  # Import the function

# Optional: LLM rerank (Stage B)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Config
# ============================================================
ART_DIR = os.getenv("ART_DIR", "artifacts")
DEPLOY_CFG_PATH = os.path.join(ART_DIR, "deploy_config.json")

JOBS_CSV = "data/jobs.csv"
TOPK_RETRIEVE_DEFAULT = 200

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TOPN = 10
LLM_SNIPPET_CHARS = 320

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


# ============================================================
# Feature extraction (MATCHES extract_features.py)
# ============================================================
from backend.services.skill_service import extract_skills_from_text, SKILL_PATTERNS  # noqa: E402

# Optional resume parser (file upload path)
try:
    from backend.services.parsing_service import parse_resume  # noqa: E402
except Exception:
    parse_resume = None

SKILLS = sorted(SKILL_PATTERNS.keys())


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


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


def skill_vector(text: str) -> np.ndarray:
    extracted_skills = set(extract_skills_from_text(text))
    vec = []
    for s in SKILLS:
        vec.append(1.0 if s in extracted_skills else 0.0)
    return np.array(vec, dtype=np.float32)


def build_feature_vector(resume_text: str, feature_columns: List[str]) -> np.ndarray:
    txt = resume_text if isinstance(resume_text, str) else ""
    years = extract_years_experience(txt)
    edu = float(extract_edu_level(txt))
    skills_vec = skill_vector(txt)

    values: Dict[str, float] = {
        "years_experience": float(years),
        "edu_level": float(edu),
    }
    for i, s in enumerate(SKILLS):
        values[f"skill_{s}"] = float(skills_vec[i])

    return np.array([values.get(col, 0.0) for col in feature_columns], dtype=np.float32)


# ============================================================
# Utilities
# ============================================================
def get_torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_text_snippet(text: str, max_chars: int = 320) -> str:
    t = (text or "").replace("\n", " ").replace("\r", " ").strip()
    return t[:max_chars]


def compute_similarity_matrix(values: List[float]) -> List[List[float]]:
    n = len(values)
    p = np.array(values, dtype=np.float32)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = abs(float(p[i] - p[j]))
    return M.tolist()


@torch.no_grad()
def generate_candidate_embedding(
    tokenizer,
    tower,
    resume_text: str,
    cand_feat_vec: np.ndarray,
    device: str,
    max_len: int,
) -> torch.Tensor:
    tok = tokenizer(
        [resume_text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    tok["features"] = torch.from_numpy(cand_feat_vec).unsqueeze(0).to(device)
    e = tower.encode_candidate(tok)
    return e.squeeze(0)


@torch.no_grad()
def predict_rerank_scores(reranker: Reranker, cand_emb: torch.Tensor, job_emb_topk: torch.Tensor) -> np.ndarray:
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)
    p = reranker(c, job_emb_topk)
    p = p.squeeze(-1)
    return p.detach().cpu().numpy()


# ============================================================
# LLM reranker (Stage B)
# ============================================================
def llm_available() -> bool:
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY", "").strip())


def llm_rerank_topk(*, user_extra: str, jobs_payload: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    if not llm_available():
        raise RuntimeError("LLM rerank requested but OPENAI_API_KEY or openai SDK is not available.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    schema = {
        "name": "rerank_result",
        "schema": {
            "type": "object",
            "properties": {
                "ranked_job_ids": {"type": "array", "items": {"type": "string"}, "minItems": top_n, "maxItems": top_n},
                "preference_scores": {"type": "array", "items": {"type": "number"}, "minItems": top_n, "maxItems": top_n},
                "violations": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "minItems": top_n, "maxItems": top_n},
                "reasons": {"type": "array", "items": {"type": "string"}, "minItems": top_n, "maxItems": top_n},
                "notes": {"type": "string"},
            },
            "required": ["ranked_job_ids", "preference_scores", "violations", "reasons", "notes"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    payload = {
        "user_extra": user_extra,
        "jobs": jobs_payload,
        "task": (
            f"Select and order the best {top_n} jobs for this candidate, using user_extra as preferences/constraints. "
            f"Choose ONLY from jobs[]. Do not invent new jobs. "
            f"preference_scores must be in [0,1], higher = better preference fit. "
            f"If a job violates a hard constraint, avoid including it in the top list."
        ),
    }

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema["strict"]
            }
        },
    )

    return json.loads(resp.output_text)


# ============================================================
# Schemas (JSON route)
# ============================================================
class RecommendRequest(BaseModel):
    resume_text: str
    top_k_retrieve: int = Field(TOPK_RETRIEVE_DEFAULT, ge=10, le=5000)
    features_override: Optional[List[float]] = None
    extra_info: Optional[str] = None
    use_llm_rerank: bool = True


class RecommendResponse(BaseModel):
    top10: List[Tuple[str, str, float]]   # (job_id, title, nn_prob)
    ranked_job_ids: List[str]             # len 10
    preference_scores: List[float]        # len 10
    violations: List[List[str]]           # len 10
    reasons: List[str]                    # len 10
    relative_matrix: List[List[float]]
    device: str
    llm_used: bool = False
    llm_notes: str = ""


# ============================================================
# Global cache
# ============================================================
DEVICE = get_torch_device()

jobs_df: Optional[pd.DataFrame] = None
job_ids: List[str] = []
job_titles: List[str] = []
job_texts: List[str] = []
job_snippets: List[str] = []
job_emb: Optional[np.ndarray] = None

feature_columns: List[str] = []
feat_dim: int = 0

tower: Optional[TwoTower] = None
reranker: Optional[Reranker] = None
tokenizer = None

CFG = None
MAX_LEN = None


# ============================================================
# Core recommend logic (shared)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs_df, job_ids, job_titles, job_texts, job_snippets, job_emb
    global feature_columns, feat_dim, tower, reranker, tokenizer
    global CFG, MAX_LEN

    # Load Config and Tokenizer
    try:
        CFG = load_json(DEPLOY_CFG_PATH)
        MAX_LEN = int(CFG["max_len"])

        tok_dir = os.path.join(ART_DIR, CFG["tokenizer_dir"])
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

        feature_columns = CFG["feature_columns"]
        feat_dim = len(feature_columns)
    except Exception as e:
        print(f"ERROR loading config/tokenizer: {e}")
        # Proceeding is hard without config, but let's try to survive if we can just mock
        pass

    jobs_df = pd.read_csv(JOBS_CSV)
    job_ids = jobs_df["job_id"].astype(str).tolist()
    job_titles = jobs_df["title"].astype(str).tolist() if "title" in jobs_df.columns else [""] * len(job_ids)
    job_texts = jobs_df["job_text"].astype(str).tolist()
    job_snippets = [format_text_snippet(t, LLM_SNIPPET_CHARS) for t in job_texts]

    # Two Tower
    try:
        tower = TwoTower(
            model_name=CFG["model_name"],
            emb_dim=CFG["tower"]["emb_dim"],
            cand_feat_dim=CFG["tower"]["cand_feat_dim"],
        ).to(DEVICE)
        
        tower_path = os.path.join(ART_DIR, CFG["tower"]["weights"])
        if os.path.exists(tower_path):
             tower.load_state_dict(
                torch.load(tower_path, map_location=DEVICE, weights_only=True)
             )
             tower.eval()
             print("[startup] TwoTower loaded.")
        else:
             print(f"[startup] WARNING: {tower_path} not found. TwoTower disabled.")
             tower = None
    except Exception as e:
        print(f"[startup] ERROR loading TwoTower: {e}")
        tower = None

    # Reranker
    try:
        reranker = Reranker(
            emb_dim=CFG["reranker"]["emb_dim"],
            hidden=CFG["reranker"]["hidden"],
            dropout=CFG["reranker"]["dropout"],
        ).to(DEVICE)
        
        reranker_path = os.path.join(ART_DIR, CFG["reranker"]["weights"])
        if os.path.exists(reranker_path):
            reranker.load_state_dict(
                torch.load(reranker_path, map_location=DEVICE, weights_only=True)
            )
            reranker.eval()
            print("[startup] Reranker loaded.")
        else:
            print(f"[startup] WARNING: {reranker_path} not found. Reranker disabled.")
            reranker = None
    except Exception as e:
        print(f"[startup] ERROR loading Reranker: {e}")
        reranker = None

    # Job Embeddings
    try:
        job_emb_path = os.path.join(ART_DIR, CFG["job_emb_path"])
        if os.path.exists(job_emb_path):
            job_emb = load_npy(job_emb_path)
            print("[startup] Job embeddings loaded.")
        else:
            print(f"[startup] WARNING: {job_emb_path} not found. Job embeddings disabled.")
            job_emb = None
    except Exception as e:
        print(f"[startup] ERROR loading job embeddings: {e}")
        job_emb = None

    yield
    print("[shutdown] server stopping")


# ============================================================
# App
# ============================================================
app = FastAPI(title="Techfest Job Recommender", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs and use POST /recommend or POST /parse_cv"}


@app.post("/api/v1/parse-resume")
async def extract_cv_text(file: UploadFile = File(...)):
    """
    Parses a resume file (PDF/DOCX/TXT) and returns the extracted text.
    Used by the frontend to get the text before the survey.
    """
    # Save temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        parsed_data = parse_resume(tmp_path)
        final_text = parsed_data.get("raw_text", "") if parsed_data else ""
        return {"text": final_text, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class JobResult(BaseModel):
    id: str
    title: str
    company: str
    location: str
    salaryRange: str
    description: str
    matchScore: float  # 0-100
    skillsRequired: List[str]


class RecommendResponse(BaseModel):
    # Old fields for backward compat or debugging
    top10: List[Tuple[str, str, float]]
    ranked_job_ids: List[str]
    relative_matrix: List[List[float]]
    
    # Rich results for frontend
    results: List[JobResult]
    
    # Metadata
    device: str
    llm_used: bool = False
    llm_notes: str = ""


@app.post("/api/v1/assessments", response_model=RecommendResponse)
async def process_career_assessment(
    cv_file: Optional[UploadFile] = File(None),
    cv_text: str = Form(""),
    preferences: str = Form("{}"),
    survey_answers: str = Form("{}")
):
    """
    Consolidated endpoint that receives CV (file or text), preferences, and survey answers.
    Combines everything into a recommendation request for the Stage B reranker.
    """
    # 1. Handle CV Text (either direct or from file)
    final_resume_text = cv_text
    if cv_file:
        suffix = os.path.splitext(cv_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(cv_file.file, tmp)
            tmp_path = tmp.name
        try:
            parsed = parse_resume(tmp_path)
            if parsed and parsed.get("raw_text"):
                final_resume_text = parsed["raw_text"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not final_resume_text.strip():
        raise HTTPException(status_code=400, detail="No resume text provided (direct or via file).")

    # 2. Extract and format extra info for the LLM
    try:
        prefs_dict = json.loads(preferences)
        answers_dict = json.loads(survey_answers)
    except Exception:
        prefs_dict = {}
        answers_dict = {}

    # Build a consolidated prompt component from survey answers
    survey_context = []
    if prefs_dict:
        industry = prefs_dict.get("industry")
        if industry: survey_context.append(f"Preferred Industry: {industry}")
        
        location = prefs_dict.get("location") or prefs_dict.get("country")
        if location: survey_context.append(f"Target Location: {location}")

        tech_stack = prefs_dict.get("techStack", [])
        if tech_stack:
            survey_context.append(f"Preferred Tech Stack: {', '.join(tech_stack)}")
            
        confident_skills = prefs_dict.get("confidentSkills", [])
        if confident_skills:
            survey_context.append(f"Skills candidate is confident in: {', '.join(confident_skills)}")

    if answers_dict:
        survey_context.append("Candidate survey responses:")
        for q_id, answer in answers_dict.items():
            # You can map q_id to friendly names if needed
            survey_context.append(f"- {q_id}: {answer}")

    extra_info_str = "\n".join(survey_context)

    # 3. Call existing recommendation logic
    # We construct a RecommendRequest and pass it to the internal logic
    req = RecommendRequest(
        resume_text=final_resume_text,
        extra_info=extra_info_str,
        use_llm_rerank=True
    )
    
    return get_job_recommendations(req)


@app.post("/api/v1/recommendations", response_model=RecommendResponse)
def get_job_recommendations(req: RecommendRequest):
    if tower is None or reranker is None or job_emb is None or tokenizer is None:
        # Check if we should allow a mock response (hard-coded dummy data)
        if os.getenv("ALLOW_MOCK_RESPONSE", "0") == "1":
             return RecommendResponse(
                 top10=[], ranked_job_ids=[], relative_matrix=[], results=[],
                 device="mock", llm_used=False, llm_notes="Mock mode"
             )
        
        # If we have the jobs data but no BERT models, let's allow a "Lightweight" fallback 
        # instead of 503-ing, so Stage B (LLM) can still work.
        if jobs_df is not None:
            print("[recommend] WARNING: Weights missing. Using Lightweight Fallback (First 20 jobs).")
        else:
            raise HTTPException(status_code=503, detail="Models and Jobs data not loaded yet.")

    # 1. Feature Vector
    cand_feat_vec = None
    if req.features_override:
         if len(req.features_override) == feat_dim:
             cand_feat_vec = np.array(req.features_override, dtype=np.float32)
    
    if cand_feat_vec is None:
        cand_feat_vec = build_feature_vector(req.resume_text, feature_columns)

    # 2. Candidate Embedding & Retrieve Top-K (Stage A)
    if tower is not None and job_emb is not None and tokenizer is not None:
        cand_emb = generate_candidate_embedding(
            tokenizer, tower, req.resume_text, cand_feat_vec, DEVICE, max_len=MAX_LEN
        )

        # 3. Retrieve Top-K (Dot Product)
        scores = job_emb @ cand_emb.detach().cpu().numpy()
        K = min(req.top_k_retrieve, len(scores))
        topk_idx = np.argpartition(scores, -K)[-K:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

        # 4. Rerank Top-K (Stage B - Model)
        if reranker is not None:
            job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(DEVICE).float()
            p_hired = predict_rerank_scores(reranker, cand_emb, job_emb_topk)
        else:
            p_hired = np.array([0.5] * len(topk_idx))
    else:
        # Fallback retrieval: Just take first 20 jobs
        topk_idx = np.arange(min(20, len(job_ids)))
        p_hired = np.array([0.5] * len(topk_idx))

    # 5. NN Top-10 Selection (Distinct High-Scoring Jobs)
    # Collect exactly 10 unique jobs from the sorted results pool
    nn_pool_indices = np.argsort(p_hired)[::-1][:100] # Take a larger pool to find 10 unique
    
    top10_tuples = []
    results_list = []
    seen_content = set()
    actual_indices_in_pool = [] # Track which pool indices we actually picked
    
    for i in nn_pool_indices:
        prob = float(p_hired[i])
        global_idx = int(topk_idx[i])
        jid = job_ids[global_idx]
        title = job_titles[global_idx]
        text_snippet = job_snippets[global_idx]
        
        # 1. Normalize
        norm_title = title.lower().strip()
        norm_snippet = text_snippet.lower().strip()
        
        # 2. Fuzzy Deduplication (Role Title Focus)
        is_duplicate = False
        for saved_title in seen_content:
            # Check Title Similarity - High threshold to catch very similar roles
            title_ratio = difflib.SequenceMatcher(None, norm_title, saved_title).ratio()
            if title_ratio > 0.9: 
                is_duplicate = True
                break
        
        if is_duplicate:
            continue

        seen_content.add(norm_title)
        actual_indices_in_pool.append(i)

        # Build Tuple
        top10_tuples.append((jid, title, prob))
        
        # Extract skills for this job
        job_skills = list(set(extract_skills_from_text(text_snippet)))[:5]
        
        # Build Rich Result
        res = JobResult(
            id=jid,
            title=title,
            company="Techfest Partner",
            location="Singapore",
            salaryRange="$5k - $10k", # Generic range
            description=text_snippet,
            matchScore=round(prob * 100, 1),
            skillsRequired=job_skills
        )
        results_list.append(res)
        
        if len(results_list) >= 10:
            break

    valid_job_ids = [r.id for r in results_list]
    # rel_matrix will be calculated at the end based on final results_list
    
    # 6. LLM Reranking (Optional)
    use_llm = req.use_llm_rerank and req.extra_info and req.extra_info.strip() and llm_available()
    
    if use_llm:
        # Prepare payload for LLM from the Top-K (Distinct ones)
        llm_pool_indices = np.argsort(p_hired)[::-1][:100] 
        jobs_payload = []
        llm_seen = set()
        
        for i in llm_pool_indices:
            jj = int(topk_idx[i])
            title = job_titles[jj].lower().strip()
            snippet = job_snippets[jj]
            
            if title in llm_seen:
                continue
            
            # Check fuzzy title similarity for the LLM pool as well
            llm_duplicate = False
            for saved_llm_title in llm_seen:
                if difflib.SequenceMatcher(None, title, saved_llm_title).ratio() > 0.9:
                    llm_duplicate = True
                    break
            
            if llm_duplicate:
                continue

            llm_seen.add(title)
            
            jobs_payload.append({
                "job_id": job_ids[jj],
                "title": title,
                "snippet": snippet,
                "nn_prob": float(p_hired[i])
            })
            
            if len(jobs_payload) >= 20: # Send top 20 UNIQUE jobs to LLM
                break
            
        try:
            llm_res = llm_rerank_topk(
                user_extra=req.extra_info,
                jobs_payload=jobs_payload,
                top_n=10
            )
            
            # Re-order results based on LLM
            ranked_ids = llm_res.get("ranked_job_ids", [])
            # Map back to JobResult
            # We need a lookup for the 20 jobs we sent
            lookup = {j["job_id"]: j for j in jobs_payload}
            
            new_results = []
            new_tuples = []
            
            for rid in ranked_ids:
                if rid in lookup:
                    j_data = lookup[rid]
                    # Note: We don't need seen_content check here because 
                    # jobs_payload was already deduplicated before sending to LLM.
                    prob = j_data["nn_prob"]
                    
                    new_tuples.append((rid, j_data["title"], prob))
                    
                    # Extract skills for this job
                    job_skills = list(set(extract_skills_from_text(j_data["snippet"])))[:5]
                    
                    new_results.append(JobResult(
                        id=rid,
                        title=j_data["title"],
                        company="Techfest Partner",
                        location="Singapore",
                        salaryRange="$5k - $12k",
                        description=j_data["snippet"],
                        matchScore=round(prob * 100, 1),
                        skillsRequired=job_skills
                    ))
            
            # If we have results, override
            if new_results:
                # GUARANTEE: If LLM returned < 10, top up with original NN results
                if len(new_results) < 10:
                    seen_ids = {r.id for r in new_results}
                    seen_titles = {r.title.lower().strip() for r in new_results}
                    
                    for original_res in results_list:
                        if original_res.id not in seen_ids and original_res.title.lower().strip() not in seen_titles:
                            new_results.append(original_res)
                            new_tuples.append((original_res.id, original_res.title, original_res.matchScore / 100.0))
                            seen_ids.add(original_res.id)
                            seen_titles.add(original_res.title.lower().strip())
                        if len(new_results) >= 10:
                            break

                results_list = new_results
                top10_tuples = new_tuples
                valid_job_ids = [r.id for r in results_list]
                final_scores = [r.matchScore for r in results_list]
                return RecommendResponse(
                    top10=top10_tuples,
                    ranked_job_ids=valid_job_ids,
                    relative_matrix=compute_similarity_matrix(final_scores),
                    results=results_list,
                    device=DEVICE,
                    llm_used=True,
                    llm_notes=llm_res.get("notes", "")
                )
                
        except Exception as e:
            print(f"LLM Rerank failed: {e}")
            # Fallback to NN results (already in results_list)

    # Return NN results (either fallback or LLM disabled)
    final_scores = [r.matchScore for r in results_list]
    return RecommendResponse(
        top10=top10_tuples,
        ranked_job_ids=valid_job_ids,
        relative_matrix=compute_similarity_matrix(final_scores),
        results=results_list,
        device=DEVICE,
        llm_used=False,
        llm_notes=""
    )


class RoadmapRequest(BaseModel):
    job_role: str
    current_skills: List[str]

@app.post("/api/v1/roadmap")
def get_roadmap(req: RoadmapRequest):
    """
    Generates a personalized roadmap for a specific job role based on current skills.
    Uses LLM (DeepSeek or OpenAI) to annotate the roadmap.
    """
    try:
        # We can pass an OpenAI client if we want to share the pool, 
        # but roadmap.py handles its own client creation robustly now.
        result = generate_roadmap(req.current_skills, req.job_role)
        
        if result and "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
