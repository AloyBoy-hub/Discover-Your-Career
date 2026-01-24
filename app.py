# app.py
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from two_tower import TwoTower
from rerank_model import Reranker

# Optional: LLM rerank
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
# Feature extraction (same logic as training)
# ============================================================
SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "go", "sql",
    "pytorch", "tensorflow", "sklearn", "scikit-learn", "pandas", "numpy",
    "llm", "nlp",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "git", "linux", "excel",
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
            except Exception:
                pass
    return float(max(vals)) if vals else 0.0


def extract_edu_level(text: str) -> float:
    text = normalize(text)
    if "phd" in text or "doctor" in text or "doctorate" in text:
        return 4.0
    if "master" in text or "msc" in text or "m.s." in text:
        return 3.0
    if "bachelor" in text or "bsc" in text or "b.s." in text or "undergraduate" in text:
        return 2.0
    if "diploma" in text or "polytechnic" in text:
        return 1.0
    return 0.0


def build_feature_vector(resume_text: str, feature_columns: List[str]) -> np.ndarray:
    t = normalize(resume_text)
    values: Dict[str, float] = {
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
        chunk = job_texts[i:i + BATCH_SIZE]
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
    # cand_emb: [D] -> [K, D]
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)
    p = reranker(c, job_emb_topk)  # [K] or [K,1] depending on your model
    p = p.squeeze(-1)
    return p.detach().cpu().numpy()


# ============================================================
# LLM reranker (Stage B)
#   - Only runs when user provides req.extra_info
#   - Returns top10 job_ids + preference_scores + violations/reasons
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

    # Keep the prompt compact and JSON-friendly.
    # We ask the model to:
    # - respect hard constraints in user_extra (if any)
    # - rerank among provided jobs only
    # - output preference_scores (NOT hiring probability)
    payload = {
        "user_extra": user_extra,
        "jobs": jobs_payload,  # already reduced to title/snippet + nn prob
        "task": (
            f"Select and order the best {top_n} jobs for this candidate. "
            f"Use user_extra as preferences/constraints. "
            f"Do NOT invent new jobs. Choose only from jobs[]. "
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

    # Per docs: SDK convenience property aggregates text output into one string. :contentReference[oaicite:0]{index=0}
    result = json.loads(resp.output_text)
    return result


# ============================================================
# FastAPI schemas
# ============================================================
class RecommendRequest(BaseModel):
    resume_text: str
    top_k_retrieve: int = Field(TOPK_RETRIEVE_DEFAULT, ge=10, le=5000)
    features_override: Optional[List[float]] = None

    # Stage B inputs (optional, free-form)
    extra_info: Optional[str] = None
    use_llm_rerank: bool = True


class RecommendResponse(BaseModel):
    # fit_score here is the model probability from your reranker (your “hiring probability proxy”)
    top10: List[Tuple[str, str, float]]
    relative_matrix: List[List[float]]
    device: str

    # Stage B outputs (present when llm_used=True)
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs_df, job_ids, job_titles, job_texts, job_snippets, job_emb
    global feature_columns, feat_dim, tower, reranker

    print("[startup] loading data and models...")

    jobs_df = pd.read_csv(JOBS_CSV)
    job_ids = jobs_df["job_id"].astype(str).tolist()
    job_titles = (
        jobs_df["title"].astype(str).tolist()
        if "title" in jobs_df.columns
        else [""] * len(job_ids)
    )
    job_texts = jobs_df["job_text"].astype(str).tolist()
    job_snippets = [make_snippet(t, LLM_SNIPPET_CHARS) for t in job_texts]

    feat_df = pd.read_csv(CAND_FEAT_CSV)
    feature_columns = [c for c in feat_df.columns if c != "candidate_id"]
    feat_dim = len(feature_columns)

    tower = TwoTower(model_name=MODEL_NAME, emb_dim=256, cand_feat_dim=feat_dim).to(DEVICE)
    # Use weights_only=True to avoid pickle warning when possible
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
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs and use POST /recommend"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if tower is None or reranker is None or job_emb is None or jobs_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    # --- Candidate features ---
    if req.features_override is not None:
        if len(req.features_override) != feat_dim:
            raise HTTPException(status_code=400, detail=f"features_override length must be {feat_dim}")
        cand_feat_vec = np.array(req.features_override, dtype=np.float32)
    else:
        cand_feat_vec = build_feature_vector(req.resume_text, feature_columns)

    # --- Candidate embedding ---
    cand_emb = encode_candidate_embedding(tokenizer, tower, req.resume_text, cand_feat_vec, DEVICE)

    # --- Retrieve top-K by tower similarity ---
    # scores: [num_jobs]
    scores = job_emb @ cand_emb.detach().cpu().numpy()
    K = min(req.top_k_retrieve, len(scores))
    topk_idx = np.argpartition(scores, -K)[-K:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    # --- Rerank with your learned reranker -> "fit score" (your probability proxy) ---
    job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(DEVICE).float()
    p_hired = rerank_probs(reranker, cand_emb, job_emb_topk)  # [K]

    # Stage B decision
    use_llm = (
        bool(req.use_llm_rerank)
        and bool(req.extra_info and req.extra_info.strip())
    )

    # ============================================================
    # Stage A (always): NN-only top10
    # ============================================================
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
    # Stage B (optional): LLM rerank the topK list
    # ============================================================
    # Build compact payload: only title+snippet+NN prob
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
            user_extra=req.extra_info or "",
            jobs_payload=jobs_payload,
            top_n=LLM_TOPN,
        )

        ranked_ids: List[str] = llm_result["ranked_job_ids"]
        preference_scores: List[float] = [float(x) for x in llm_result["preference_scores"]]
        violations: List[List[str]] = llm_result["violations"]
        reasons: List[str] = llm_result["reasons"]
        notes: str = llm_result["notes"]

        # Map job_id -> job info from payload
        by_id = {x["job_id"]: x for x in jobs_payload}

        top10: List[Tuple[str, str, float]] = []
        probs_10: List[float] = []

        for jid in ranked_ids:
            x = by_id.get(jid)
            if x is None:
                continue
            # You said: fit score = model probability output (reranker)
            fit_prob = float(x["nn_prob"])
            top10.append((x["job_id"], x["title"], fit_prob))
            probs_10.append(fit_prob)

        # Safety fallback if LLM returns something odd
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
        # If LLM rerank fails, fall back to NN-only
        top10, probs_10 = nn_only_top10()
        rel_matrix = upper_triangle_abs_diff(probs_10)
        return RecommendResponse(
            top10=top10,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=False,
            llm_notes=f"LLM rerank failed; used NN-only fallback. Error={type(e).__name__}: {e}",
        )
