# infer.py
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import os
import sys
# Ensure root modules (backend, etc.) can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.two_tower import TwoTower
from backend.ml.reranker import Reranker

SKILLS = [
    "python","java","c++","javascript","typescript","go","sql",
    "pytorch","tensorflow","sklearn","scikit-learn","pandas","numpy",
    "llm","nlp",
    "aws","azure","gcp","docker","kubernetes","terraform","jenkins",
    "git","linux","excel",
]

def get_torch_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

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
            try: vals.append(float(m.group(1)))
            except: pass
    return float(max(vals)) if vals else 0.0

def extract_edu_level(text: str) -> float:
    text = normalize(text)
    if "phd" in text or "doctor" in text or "doctorate" in text: return 4.0
    if "master" in text or "msc" in text or "m.s." in text: return 3.0
    if "bachelor" in text or "bsc" in text or "b.s." in text or "undergraduate" in text: return 2.0
    if "diploma" in text or "polytechnic" in text: return 1.0
    return 0.0

def build_feature_vector(resume_text: str, feature_columns: list[str]) -> np.ndarray:
    t = normalize(resume_text)
    values = {}
    values["years_experience"] = extract_years_experience(t)
    values["edu_level"] = extract_edu_level(t)
    for s in SKILLS:
        values[f"skill_{s}"] = 1.0 if s in t else 0.0
    return np.array([values.get(col, 0.0) for col in feature_columns], dtype=np.float32)

def compute_similarity_matrix(probs_10: list[float]) -> list[list[float]]:
    p = np.array(probs_10, dtype=np.float32)
    M = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        for j in range(i+1, 10):
            M[i, j] = abs(float(p[i] - p[j]))
    return M.tolist()

@torch.no_grad()
def encode_job_embeddings(tokenizer, tower, job_texts, device, max_len=256, batch_size=64):
    embs = []
    for i in range(0, len(job_texts), batch_size):
        chunk = job_texts[i:i+batch_size]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        e = tower.encode_job(tok)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)

@torch.no_grad()
def encode_candidate_embedding(tokenizer, tower, resume_text, cand_feat_vec, device, max_len=256):
    tok = tokenizer([resume_text], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    tok["features"] = torch.from_numpy(cand_feat_vec).unsqueeze(0).to(device)  # [1,F]
    e = tower.encode_candidate(tok)  # [1,D]
    return e.squeeze(0)              # [D]

@torch.no_grad()
def rerank_probs(reranker, cand_emb, job_emb_topk, device):
    # cand_emb: [D], job_emb_topk: [K,D]
    c = cand_emb.unsqueeze(0).expand(job_emb_topk.size(0), -1)  # [K,D]
    p = reranker(c, job_emb_topk)  # [K] sigmoid
    return p.detach().cpu().numpy()

def recommend_for_one(resume_text: str, top_k_retrieve: int = 200):
    device = get_torch_device()
    print("DEVICE =", device)
    model_name = "distilbert-base-uncased"

    jobs = pd.read_csv("data/jobs.csv")
    job_ids = jobs["job_id"].tolist()
    job_titles = jobs["title"].tolist() if "title" in jobs.columns else [""] * len(jobs)
    job_texts = jobs["job_text"].tolist()

    feat_df = pd.read_csv("data/candidate_features.csv")
    feature_columns = [c for c in feat_df.columns if c != "candidate_id"]
    feat_dim = len(feature_columns)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load tower + reranker
    tower = TwoTower(model_name=model_name, emb_dim=256, cand_feat_dim=feat_dim).to(device)
    tower.load_state_dict(torch.load("two_tower.pt", map_location=device, weights_only=True))
    tower.eval()

    reranker = Reranker(emb_dim=256, hidden=256, dropout=0.2).to(device)
    reranker.load_state_dict(torch.load("reranker.pt", map_location=device, weights_only=True))
    reranker.eval()

    # Precompute / cache job embeddings (in production, cache this globally)
    job_emb = encode_job_embeddings(tokenizer, tower, job_texts, device=device)  # [N,D]

    # Candidate embedding
    cand_feat_vec = build_feature_vector(resume_text, feature_columns)  # [F]
    cand_emb = encode_candidate_embedding(tokenizer, tower, resume_text, cand_feat_vec, device=device)  # [D]

    # Retrieve top-K by cosine similarity
    scores = job_emb @ cand_emb.detach().cpu().numpy()  # [N]
    K = min(top_k_retrieve, len(scores))
    topk_idx = np.argpartition(scores, -K)[-K:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]  # sorted desc

    # Rerank: compute hire probabilities for these K
    job_emb_topk = torch.from_numpy(job_emb[topk_idx]).to(device).float()  # [K,D]
    p_hired = rerank_probs(reranker, cand_emb, job_emb_topk, device=device)  # [K]

    # Pick final top-10 by probability
    order = np.argsort(p_hired)[::-1][:10]
    top10 = []
    probs_10 = []
    for o in order:
        j = int(topk_idx[o])
        prob = float(p_hired[o])
        top10.append((job_ids[j], job_titles[j], prob))
        probs_10.append(prob)

    rel_matrix = compute_similarity_matrix(probs_10)

    return top10, rel_matrix

def main():
    resume_text = """Your resume text / LinkedIn text here..."""

    top10, rel = recommend_for_one(resume_text, top_k_retrieve=200)

    print("\nTop-10 recommended jobs (probability of hired):")
    for job_id, title, p in top10:
        print(job_id, f"{p:.4f}", title)

    print("\nUpper-triangle |p_i - p_j| matrix:")
    for row in rel:
        print(row)

if __name__ == "__main__":
    main()
