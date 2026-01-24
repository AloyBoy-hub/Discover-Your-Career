# app.py
from __future__ import annotations

import re
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from two_tower import TwoTower
from rerank_model import Reranker


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


# ============================================================
# Feature extraction (same logic as training)
# ============================================================
SKILLS = [
    "python","java","c++","javascript","typescript","go","sql",
    "pytorch","tensorflow","sklearn","scikit-learn","pandas","numpy",
    "llm","nlp",
    "aws","azure","gcp","docker","kubernetes","terraform","jenkins",
    "git","linux","excel",
]

def normalize(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""

def extract_years_experience(text: str) -> float:
    text = normalize(text)
    patterns = [
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+of\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)\s+experience",
        r"(\d{1,2})\s*\+?\s*(?:years|year|yrs|yr)",
    ]
    vals = []
    for p in patterns:
        for m in re.finditer(p, text):
            try:
                vals.append(float(m.group(1)))
            except:
                pass
    return float(max(vals)) if vals else 0.0

def extract_edu_level(text: str) -> float:
    text = normalize(text)
    if "phd" in text or "doctor" in text or "doctorate" in text: return 4.0
    if "master" in text or "msc" in text or "m.s." in text: return 3.0
    if "bachelor" in text or "bsc" in text or "b.s." in text or "undergraduate" in text: return 2.0
    if "diploma" in text or "polytechnic" in text: return 1.0
    return 0.0

def build_feature_vector(resume_text: str, feature_columns: List[str]) -> np.ndarray:
    t = normalize(resume_text)
    values = {
        "years_experience": extract_years_experience(t),
        "edu_level": extract_edu_level(t),
    }
    for s in SKILLS:
        values[f"skill_{s}"] = 1.0 if s in t else 0.0
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

def upper_triangle_abs_diff(probs_10: List[float]) -> List[List[float]]:
    p = np.array(probs_10, dtype=np.float32)
    M = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        for j in range(i + 1, 10):
            M[i, j] = abs(float(p[i] - p[j]))
    return M.tolist()

@torch.no_grad()
def encode_job_embeddings(tokenizer, tower, job_texts, device: str) -> np.ndarray:
    embs = []
    for i in range(0, len(job_texts), BATCH_SIZE):
        chunk = job_texts[i:i + BATCH_SIZE]
        tok = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        e = tower.encode_job(tok)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)

@torch.no_grad()
def encode_candidate_embedding(tokenizer, tower, resume_text: str, cand_feat_vec: np.ndarray, device: str) -> torch.Tensor:
    tok = tokenizer(
        [resume_text], padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    tok["features"] = torch.from_numpy(cand_feat_vec).unsqueeze(0).to(device)
    e = tower.encode_candidate(tok)
    return e.squeeze(0)

@torch.no_grad()
def rerank_probs(reranker: Reranker, cand_emb: torch.Tensor, job_emb_topk: torch.Tensor) -> np.ndarray:
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)
    p = reranker(c, job_emb_topk)
    return p.cpu().numpy()


# ============================================================
# FastAPI schemas
# ============================================================
class RecommendRequest(BaseModel):
    resume_text: str
    top_k_retrieve: int = Field(TOPK_RETRIEVE_DEFAULT, ge=10, le=5000)
    features_override: Optional[List[float]] = None

class RecommendResponse(BaseModel):
    top10: List[Tuple[str, str, float]]
    relative_matrix: List[List[float]]
    device: str


# ============================================================
# Global cache (loaded once)
# ============================================================
DEVICE = pick_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

jobs_df = None
job_ids: List[str] = []
job_titles: List[str] = []
job_texts: List[str] = []
job_emb: np.ndarray | None = None

feature_columns: List[str] = []
feat_dim: int = 0

tower: TwoTower | None = None
reranker: Reranker | None = None


# ============================================================
# Lifespan (startup / shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs_df, job_ids, job_titles, job_texts, job_emb
    global feature_columns, feat_dim, tower, reranker

    print("[startup] loading data and models...")

    jobs_df = pd.read_csv(JOBS_CSV)
    job_ids = jobs_df["job_id"].astype(str).tolist()
    job_titles = jobs_df["title"].astype(str).tolist() if "title" in jobs_df.columns else [""] * len(job_ids)
    job_texts = jobs_df["job_text"].astype(str).tolist()

    feat_df = pd.read_csv(CAND_FEAT_CSV)
    feature_columns = [c for c in feat_df.columns if c != "candidate_id"]
    feat_dim = len(feature_columns)

    tower = TwoTower(model_name=MODEL_NAME, emb_dim=256, cand_feat_dim=feat_dim).to(DEVICE)
    tower.load_state_dict(torch.load(TOWER_WEIGHTS, map_location=DEVICE, weights_only=True))
    tower.eval()

    reranker = Reranker(emb_dim=256, hidden=256, dropout=0.2).to(DEVICE)
    reranker.load_state_dict(torch.load(RERANK_WEIGHTS, map_location=DEVICE, weights_only=True))
    reranker.eval()

    job_emb = encode_job_embeddings(tokenizer, tower, job_texts, device=DEVICE)

    print(f"[startup] DEVICE={DEVICE} jobs={len(job_ids)} emb_dim={job_emb.shape}")

    yield

    print("[shutdown] server stopping")


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="Techfest Job Recommender",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs and use POST /recommend"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    assert tower is not None and reranker is not None and job_emb is not None

    if req.features_override is not None:
        if len(req.features_override) != feat_dim:
            raise ValueError("features_override has incorrect length")
        cand_feat_vec = np.array(req.features_override, dtype=np.float32)
    else:
        cand_feat_vec = build_feature_vector(req.resume_text, feature_columns)

    cand_emb = encode_candidate_embedding(
        tokenizer, tower, req.resume_text, cand_feat_vec, DEVICE
    )

    scores = job_emb @ cand_emb.detach().cpu().numpy()
    K = min(req.top_k_retrieve, len(scores))
    topk_idx = np.argpartition(scores, -K)[-K:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(DEVICE).float()
    p_hired = rerank_probs(reranker, cand_emb, job_emb_topk)

    order = np.argsort(p_hired)[::-1][:10]
    top10 = []
    probs_10 = []

    for o in order:
        j = int(topk_idx[o])
        prob = float(p_hired[o])
        top10.append((job_ids[j], job_titles[j], prob))
        probs_10.append(prob)

    rel_matrix = upper_triangle_abs_diff(probs_10)

    return RecommendResponse(
        top10=top10,
        relative_matrix=rel_matrix,
        device=DEVICE
    )
