# app.py
from __future__ import annotations

import json
import os
import re
import sys
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from two_tower import TwoTower
from rerank_model import Reranker

# Optional: LLM rerank (Stage B)
try:
    from openai import OpenAI  # pip install -U openai
except Exception:
    OpenAI = None


# ============================================================
# Config
# ============================================================
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 64
TOPK_RETRIEVE_DEFAULT = 200

JOBS_CSV = "data/jobs.csv"
CAND_FEAT_CSV = "data/candidate_features.csv"

TOWER_WEIGHTS = "two_tower.pt"
RERANK_WEIGHTS = "reranker.pt"

# LLM rerank
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TOPN = 10
LLM_SNIPPET_CHARS = 320


# ============================================================
# Feature extraction (MATCHES your extract_features.py)
# ============================================================
# ============================================================
# Feature extraction (MATCHES your extract_features.py)
# ============================================================
# Ensure backend module can be imported (same approach as your extract_features.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.skill_extractor import extract_skills_from_text, SKILL_PATTERNS  # noqa: E402
from backend.resume_parser import parse_resume # noqa: E402

# Canonical skill columns are defined by your shared skill dictionary
SKILLS = sorted(SKILL_PATTERNS.keys())


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()


def extract_years_experience(text: str) -> float:
    """
    Very rough: finds patterns like '3 years', '5+ years', '2 yrs'
    Returns max found, else 0.
    """
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
    """
    Ordinal encoding:
    4 = PhD/Doctorate
    3 = Master
    2 = Bachelor
    1 = Diploma/Poly
    0 = unknown
    """
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
    """
    Builds a feature vector for ONE user resume, using the SAME schema as candidate_features.csv:
      years_experience, edu_level, skill_<canonical_skill>...
    IMPORTANT: We respect feature_columns ordering from candidate_features.csv
    so it stays consistent with training.
    """
    txt = resume_text if isinstance(resume_text, str) else ""
    years = extract_years_experience(txt)
    edu = float(extract_edu_level(txt))
    skills_vec = skill_vector(txt)  # aligned with SKILLS sorted(SKILL_PATTERNS.keys())

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
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def upper_triangle_abs_diff(values_10: List[float]) -> List[List[float]]:
    p = np.array(values_10, dtype=np.float32)
    M = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        for j in range(i + 1, 10):
            M[i, j] = abs(float(p[i] - p[j]))
    return M.tolist()


def make_snippet(text: str, max_chars: int = 320) -> str:
    t = (text or "").replace("\n", " ").replace("\r", " ").strip()
    return t[:max_chars]


@torch.no_grad()
def encode_job_embeddings(tokenizer, tower, job_texts: List[str], device: str) -> np.ndarray:
    embs = []
    for i in range(0, len(job_texts), BATCH_SIZE):
        chunk = job_texts[i : i + BATCH_SIZE]
        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        e = tower.encode_job(tok)
        embs.append(e.detach().cpu().numpy())
    return np.vstack(embs)


@torch.no_grad()
def encode_candidate_embedding(
    tokenizer,
    tower,
    resume_text: str,
    cand_feat_vec: np.ndarray,
    device: str,
) -> torch.Tensor:
    tok = tokenizer(
        [resume_text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    tok["features"] = torch.from_numpy(cand_feat_vec).unsqueeze(0).to(device)
    e = tower.encode_candidate(tok)
    return e.squeeze(0)


@torch.no_grad()
def rerank_probs(reranker: Reranker, cand_emb: torch.Tensor, job_emb_topk: torch.Tensor) -> np.ndarray:
    """
    cand_emb: [D]
    job_emb_topk: [K, D]
    returns: [K] probabilities (or scores) from your reranker
    """
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)
    p = reranker(c, job_emb_topk)  # [K] or [K,1]
    p = p.squeeze(-1)
    return p.detach().cpu().numpy()


# ============================================================
# LLM reranker (Stage B)
# ============================================================
def llm_available() -> bool:
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY", "").strip())


def llm_rerank_topk(
    *,
    user_extra: str,
    jobs_payload: List[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    jobs_payload items:
      { job_id, title, snippet, nn_prob }
    Output dict:
      ranked_job_ids: [10]
      preference_scores: [10] floats [0,1]
      violations: [10] list-of-list[str]
      reasons: [10] list[str]
      notes: str
    """
    if not llm_available():
        raise RuntimeError("LLM rerank requested but OPENAI_API_KEY or openai SDK is not available.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    schema = {
        "name": "rerank_result",
        "schema": {
            "type": "object",
            "properties": {
                "ranked_job_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "preference_scores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "violations": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
                "reasons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": top_n,
                    "maxItems": top_n,
                },
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
        input=[
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            }
        ],
        text={"format": {"type": "json_schema", "json_schema": schema}},
    )

    return json.loads(resp.output_text)


# ============================================================
# FastAPI schemas
# ============================================================
class RecommendRequest(BaseModel):
    resume_text: str
    top_k_retrieve: int = Field(TOPK_RETRIEVE_DEFAULT, ge=10, le=5000)

    # If you want to bypass resume-derived features, pass an explicit vector.
    features_override: Optional[List[float]] = None

    # Stage B: optional free-form preferences/constraints
    extra_info: Optional[str] = None
    use_llm_rerank: bool = True


class RecommendResponse(BaseModel):
    # fit score = reranker probability (your “hiring probability proxy”)
    top10: List[Tuple[str, str, float]]
    relative_matrix: List[List[float]]
    device: str

    # present when llm_used=True
    llm_used: bool = False
    preference_scores: Optional[List[float]] = None
    violations: Optional[List[List[str]]] = None
    reasons: Optional[List[str]] = None
    llm_notes: Optional[str] = None


# ============================================================
# Global cache (loaded once)
# ============================================================
DEVICE = pick_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

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


# ============================================================
# Lifespan (startup / shutdown)
# ============================================================
# ============================================================
# Lifespan (startup / shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs_df, job_ids, job_titles, job_texts, job_snippets, job_emb
    global feature_columns, feat_dim, tower, reranker

    print("[startup] loading data and models...")

    if os.path.exists(JOBS_CSV):
        jobs_df = pd.read_csv(JOBS_CSV)
        job_ids = jobs_df["job_id"].astype(str).tolist()
        job_titles = (
            jobs_df["title"].astype(str).tolist()
            if "title" in jobs_df.columns
            else [""] * len(job_ids)
        )
        job_texts = jobs_df["job_text"].astype(str).tolist()
        job_snippets = [make_snippet(t, LLM_SNIPPET_CHARS) for t in job_texts]
    else:
        print(f"WARNING: {JOBS_CSV} not found. Running with empty job database.")
        jobs_df = pd.DataFrame() # dummy
        job_ids = []
        job_titles = []
        job_texts = []
        job_snippets = []

    if os.path.exists(CAND_FEAT_CSV):
        feat_df = pd.read_csv(CAND_FEAT_CSV)
        feature_columns = [c for c in feat_df.columns if c != "candidate_id"]
        feat_dim = len(feature_columns)
    else:
        print(f"WARNING: {CAND_FEAT_CSV} not found. Using default feature dimension 0.")
        feature_columns = []
        feat_dim = 0
        
    # --- Load Models (Robust) ---
    try:
        if os.path.exists(TOWER_WEIGHTS):
            tower = TwoTower(model_name=MODEL_NAME, emb_dim=256, cand_feat_dim=feat_dim).to(DEVICE)
            tower.load_state_dict(torch.load(TOWER_WEIGHTS, map_location=DEVICE, weights_only=True))
            tower.eval()
        else:
             print(f"WARNING: {TOWER_WEIGHTS} not found. TwoTower model will not be loaded.")
             tower = None
    except Exception as e:
        print(f"ERROR loading TwoTower: {e}")
        tower = None

    try:
        if os.path.exists(RERANK_WEIGHTS):
            reranker = Reranker(emb_dim=256, hidden=256, dropout=0.2).to(DEVICE)
            reranker.load_state_dict(torch.load(RERANK_WEIGHTS, map_location=DEVICE, weights_only=True))
            reranker.eval()
        else:
            print(f"WARNING: {RERANK_WEIGHTS} not found. Reranker model will not be loaded.")
            reranker = None
    except Exception as e:
        print(f"ERROR loading Reranker: {e}")
        reranker = None

    # Encode jobs if tower exists
    if tower is not None and len(job_texts) > 0:
        print("Encoding job embeddings...")
        job_emb = encode_job_embeddings(tokenizer, tower, job_texts, device=DEVICE)
        print(f"[startup] DEVICE={DEVICE} jobs={len(job_ids)} job_emb={job_emb.shape}")
    else:
        print("WARNING: Skipping job embedding generation (no tower or no jobs).")
        job_emb = None

    yield
    print("[shutdown] server stopping")


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="Techfest Job Recommender",
    lifespan=lifespan,
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs and use POST /recommend"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    # Option 1: File upload
    file: Optional[UploadFile] = File(None),
    # Option 2: JSON-like fields via Form (since we have a file, everything must be Form)
    resume_text: Optional[str] = Form(None),
    top_k_retrieve: int = Form(TOPK_RETRIEVE_DEFAULT),
    extra_info: Optional[str] = Form(None),
    use_llm_rerank: bool = Form(True),
    # For features_override, passing JSON string in Form is easiest, or list of queries
    # Simple fix: accept a JSON string for overrides if needed
    features_override_json: Optional[str] = Form(None)
):
    """
    Hybrid endpoint:
    - If `file` is uploaded (PDF/TXT), we parse it.
    - If `resume_text` is provided directly, we use that.
    
    Returns standard RecommendResponse.
    """
    if tower is None or reranker is None or job_emb is None:
        # For testing frontend connection without models, return a Mock response if permitted
        # Or just error out. Let's error out but with a clear message.
        # ALLOW_MOCK for demo purposes if env var set?
        if os.getenv("ALLOW_MOCK_RESPONSE", "0") == "1":
             return RecommendResponse(
                top10=[],
                relative_matrix=[],
                device=DEVICE,
                llm_used=False,
                llm_notes="Models not loaded. Mock response enabled."
            )
        raise HTTPException(status_code=503, detail="Models not loaded yet (missing .pt files).")

    # 1. Parse Input
    final_resume_text = ""
    
    if file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            parsed_data = parse_resume(tmp_path)
            if parsed_data:
                # Combine extracted fields into a rich text for embedding
                # This logic depends on what your model expects. 
                # If model expects raw text, use parsed_data['raw_text']
                final_resume_text = parsed_data.get('raw_text', '')
            else:
                final_resume_text = ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    elif resume_text:
        final_resume_text = resume_text
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'file' or 'resume_text'.")

    # 2. Features Override
    cand_feat_vec = None
    if features_override_json:
        try:
            overrides = json.loads(features_override_json)
            if isinstance(overrides, list) and len(overrides) == feat_dim:
                cand_feat_vec = np.array(overrides, dtype=np.float32)
        except:
             pass # ignore invalid json

    # 3. Default Features
    if cand_feat_vec is None:
        cand_feat_vec = build_feature_vector(final_resume_text, feature_columns)

    # --- candidate embedding ---
    cand_emb = encode_candidate_embedding(tokenizer, tower, final_resume_text, cand_feat_vec, DEVICE)

    # --- Stage A: retrieve top-K by tower similarity ---
    scores = job_emb @ cand_emb.detach().cpu().numpy()
    K = min(top_k_retrieve, len(scores))

    topk_idx = np.argpartition(scores, -K)[-K:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    # --- Stage A: reranker probability for the top-K ---
    job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(DEVICE).float()
    p_hired = rerank_probs(reranker, cand_emb, job_emb_topk)  # [K]

    def nn_only_top10() -> Tuple[List[Tuple[str, str, float]], List[float]]:
        order = np.argsort(p_hired)[::-1][:10]
        out: List[Tuple[str, str, float]] = []
        probs: List[float] = []
        for o in order:
            j = int(topk_idx[o])
            prob = float(p_hired[o])
            out.append((job_ids[j], job_titles[j], prob))
            probs.append(prob)
        return out, probs

    use_llm = bool(use_llm_rerank) and bool(extra_info and extra_info.strip())

    # If no extra_info, skip LLM and return NN-only immediately
    if not use_llm:
        top10, probs_10 = nn_only_top10()
        rel_matrix = upper_triangle_abs_diff(probs_10)
        return RecommendResponse(
            top10=top10,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=False,
        )

    # ============================================================
    # Stage B: LLM rerank the Stage-A shortlist
    # ============================================================
    jobs_payload: List[Dict[str, Any]] = []
    for pos, j in enumerate(topk_idx):
        jj = int(j)
        jobs_payload.append(
            {
                "job_id": job_ids[jj],
                "title": job_titles[jj],
                "snippet": job_snippets[jj],
                "nn_prob": float(p_hired[pos]),
            }
        )

    try:
        llm_result = llm_rerank_topk(
            user_extra=extra_info or "",
            jobs_payload=jobs_payload,
            top_n=LLM_TOPN,
        )

        ranked_ids: List[str] = llm_result["ranked_job_ids"]
        preference_scores: List[float] = [float(x) for x in llm_result["preference_scores"]]
        violations: List[List[str]] = llm_result["violations"]
        reasons: List[str] = llm_result["reasons"]
        notes: str = llm_result["notes"]

        by_id = {x["job_id"]: x for x in jobs_payload}

        top10: List[Tuple[str, str, float]] = []
        probs_10: List[float] = []

        for jid in ranked_ids:
            x = by_id.get(jid)
            if x is None:
                continue
            # Your requirement: fit score is probability output from your model's last layer
            fit_prob = float(x["nn_prob"])
            top10.append((x["job_id"], x["title"], fit_prob))
            probs_10.append(fit_prob)

        # Fallback if LLM output is weird/incomplete
        if len(top10) < 10:
            fallback_top10, fallback_probs = nn_only_top10()
            seen = {t[0] for t in top10}
            for t, p in zip(fallback_top10, fallback_probs):
                if t[0] in seen:
                    continue
                top10.append(t)
                probs_10.append(p)
                if len(top10) == 10:
                    break

        rel_matrix = upper_triangle_abs_diff(probs_10)

        return RecommendResponse(
            top10=top10,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=True,
            preference_scores=preference_scores,
            violations=violations,
            reasons=reasons,
            llm_notes=notes,
        )

    except Exception as e:
        # LLM failed -> NN-only fallback
        top10, probs_10 = nn_only_top10()
        rel_matrix = upper_triangle_abs_diff(probs_10)
        return RecommendResponse(
            top10=top10,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=False,
            llm_notes=f"LLM rerank failed; used NN-only fallback. Error={type(e).__name__}: {e}",
        )
