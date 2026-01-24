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
from deploy_utils import load_json, load_npy

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

# LLM rerank
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TOPN = 10
LLM_SNIPPET_CHARS = 320


# ============================================================
# Feature extraction (MATCHES your extract_features.py)
# ============================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.skill_extractor import extract_skills_from_text, SKILL_PATTERNS  # noqa: E402
from backend.resume_parser import parse_resume # noqa: E402

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
def encode_candidate_embedding(
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
def rerank_probs(reranker: Reranker, cand_emb: torch.Tensor, job_emb_topk: torch.Tensor) -> np.ndarray:
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)
    p = reranker(c, job_emb_topk)
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
        input=[{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
        text={"format": {"type": "json_schema", "json_schema": schema}},
    )

    return json.loads(resp.output_text)


# ============================================================
# FastAPI schemas (ALWAYS same shape)
# ============================================================
class RecommendRequest(BaseModel):
    resume_text: str
    top_k_retrieve: int = Field(TOPK_RETRIEVE_DEFAULT, ge=10, le=5000)
    features_override: Optional[List[float]] = None
    extra_info: Optional[str] = None
    use_llm_rerank: bool = True


class RecommendResponse(BaseModel):
    # Always present
    top10: List[Tuple[str, str, float]]                 # (job_id, title, nn_prob)
    ranked_job_ids: List[str]                           # length 10
    preference_scores: List[float]                      # length 10
    violations: List[List[str]]                         # length 10 (empty lists when no LLM)
    reasons: List[str]                                  # length 10 ("" when no LLM)
    relative_matrix: List[List[float]]
    device: str

    # Metadata
    llm_used: bool = False
    llm_notes: str = ""


# ============================================================
# Global cache
# ============================================================
DEVICE = pick_device()

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
# Lifespan
# ============================================================
# ============================================================
# Lifespan (startup / shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs_df, job_ids, job_titles, job_texts, job_snippets, job_emb
    global feature_columns, feat_dim, tower, reranker, tokenizer
    global CFG, MAX_LEN

    CFG = load_json(DEPLOY_CFG_PATH)
    MAX_LEN = int(CFG["max_len"])

    tok_dir = os.path.join(ART_DIR, CFG["tokenizer_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    feature_columns = CFG["feature_columns"]
    feat_dim = len(feature_columns)

    jobs_df = pd.read_csv(JOBS_CSV)
    job_ids = jobs_df["job_id"].astype(str).tolist()
    job_titles = jobs_df["title"].astype(str).tolist() if "title" in jobs_df.columns else [""] * len(job_ids)
    job_texts = jobs_df["job_text"].astype(str).tolist()
    job_snippets = [make_snippet(t, LLM_SNIPPET_CHARS) for t in job_texts]

    tower = TwoTower(
        model_name=CFG["model_name"],
        emb_dim=CFG["tower"]["emb_dim"],
        cand_feat_dim=CFG["tower"]["cand_feat_dim"],
    ).to(DEVICE)
    tower.load_state_dict(
        torch.load(os.path.join(ART_DIR, CFG["tower"]["weights"]), map_location=DEVICE, weights_only=True)
    )
    tower.eval()

    reranker = Reranker(
        emb_dim=CFG["reranker"]["emb_dim"],
        hidden=CFG["reranker"]["hidden"],
        dropout=CFG["reranker"]["dropout"],
    ).to(DEVICE)
    reranker.load_state_dict(
        torch.load(os.path.join(ART_DIR, CFG["reranker"]["weights"]), map_location=DEVICE, weights_only=True)
    )
    reranker.eval()

    job_emb = load_npy(os.path.join(ART_DIR, CFG["job_emb_path"]))
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
app = FastAPI(title="Techfest Job Recommender", lifespan=lifespan)

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
def recommend(req: RecommendRequest):
    if tower is None or reranker is None or job_emb is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    # Candidate feature vector
    if req.features_override is not None:
        if len(req.features_override) != feat_dim:
            raise HTTPException(status_code=400, detail=f"features_override length must be {feat_dim}")
        cand_feat_vec = np.array(req.features_override, dtype=np.float32)
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

    # Candidate embedding
    cand_emb = encode_candidate_embedding(tokenizer, tower, req.resume_text, cand_feat_vec, DEVICE, max_len=MAX_LEN)
    # --- candidate embedding ---
    cand_emb = encode_candidate_embedding(tokenizer, tower, final_resume_text, cand_feat_vec, DEVICE)

    # Retrieve top-K by dot product (TwoTower)
    scores = job_emb @ cand_emb.detach().cpu().numpy()
    K = min(top_k_retrieve, len(scores))

    topk_idx = np.argpartition(scores, -K)[-K:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    # Rerank top-K (Reranker) => probabilities
    job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(DEVICE).float()
    p_hired = rerank_probs(reranker, cand_emb, job_emb_topk)  # [K]

    # NN top-10 (always available)
    nn_order = np.argsort(p_hired)[::-1][:10]
    top10: List[Tuple[str, str, float]] = []
    ranked_job_ids: List[str] = []
    nn_scores_10: List[float] = []
    for o in nn_order:
        j = int(topk_idx[o])
        prob = float(p_hired[o])
        top10.append((job_ids[j], job_titles[j], prob))
        ranked_job_ids.append(job_ids[j])
        nn_scores_10.append(prob)

    # Default (no-LLM) fields — ALWAYS returned
    empty_violations = [[] for _ in range(10)]
    empty_reasons = ["" for _ in range(10)]
    rel_matrix = upper_triangle_abs_diff(nn_scores_10)

    # Decide if we will call LLM
    use_llm = bool(req.use_llm_rerank) and bool(req.extra_info and req.extra_info.strip())
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

    # If no LLM, return same shape with blanks + NN preference_scores
    if not use_llm:
        return RecommendResponse(
            top10=top10,
            ranked_job_ids=ranked_job_ids,
            preference_scores=nn_scores_10,   # NN-based “preference” scores
            violations=empty_violations,
            reasons=empty_reasons,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=False,
            llm_notes="",
        )

    # Stage B LLM rerank the Stage-A shortlist (top-K)
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
        llm_result = llm_rerank_topk(user_extra=req.extra_info or "", jobs_payload=jobs_payload, top_n=LLM_TOPN)
        llm_result = llm_rerank_topk(
            user_extra=extra_info or "",
            jobs_payload=jobs_payload,
            top_n=LLM_TOPN,
        )

        ranked_ids_llm: List[str] = llm_result["ranked_job_ids"]
        preference_scores_llm: List[float] = [float(x) for x in llm_result["preference_scores"]]
        violations_llm: List[List[str]] = llm_result["violations"]
        reasons_llm: List[str] = llm_result["reasons"]
        notes_llm: str = llm_result["notes"]

        # Make sure we only output jobs that exist in payload, and always length 10
        by_id = {x["job_id"]: x for x in jobs_payload}

        ranked_job_ids_out: List[str] = []
        top10_out: List[Tuple[str, str, float]] = []
        probs_out: List[float] = []
        pref_out: List[float] = []
        vio_out: List[List[str]] = []
        rea_out: List[str] = []

        for i, jid in enumerate(ranked_ids_llm[:10]):
            x = by_id.get(jid)
            if x is None:
                continue
            ranked_job_ids_out.append(jid)
            top10_out.append((x["job_id"], x["title"], float(x["nn_prob"])))  # still your NN fit prob
            probs_out.append(float(x["nn_prob"]))
            pref_out.append(float(preference_scores_llm[i]) if i < len(preference_scores_llm) else 0.0)
            vio_out.append(violations_llm[i] if i < len(violations_llm) else [])
            rea_out.append(reasons_llm[i] if i < len(reasons_llm) else "")

        # Fallback fill if LLM output was incomplete
        if len(ranked_job_ids_out) < 10:
            seen = set(ranked_job_ids_out)
            for (jid, title, prob), pref in zip(top10, nn_scores_10):
                if jid in seen:
                    continue
                ranked_job_ids_out.append(jid)
                top10_out.append((jid, title, prob))
                probs_out.append(prob)
                pref_out.append(pref)      # fallback to NN
                vio_out.append([])
                rea_out.append("")
                if len(ranked_job_ids_out) == 10:
                    break

        rel_matrix_llm = upper_triangle_abs_diff(probs_out)

        return RecommendResponse(
            top10=top10_out,
            ranked_job_ids=ranked_job_ids_out,
            preference_scores=pref_out,
            violations=vio_out,
            reasons=rea_out,
            relative_matrix=rel_matrix_llm,
            device=DEVICE,
            llm_used=True,
            llm_notes=notes_llm,
        )

    except Exception as e:
        # LLM failed -> same shape, NN fallback, blanks for LLM fields
        return RecommendResponse(
            top10=top10,
            ranked_job_ids=ranked_job_ids,
            preference_scores=nn_scores_10,
            violations=empty_violations,
            reasons=empty_reasons,
            relative_matrix=rel_matrix,
            device=DEVICE,
            llm_used=False,
            llm_notes=f"LLM rerank failed; NN-only fallback. Error={type(e).__name__}: {e}",
        )
