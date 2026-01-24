# Inference: Recommend top-K jobs for a new candidate
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer
from two_tower import TwoTower

@torch.no_grad()
def encode_texts(tokenizer, model, texts, tower="job", device="cpu", max_len=256, batch_size=64):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        if tower == "job":
            emb = model.encode_job(tok)
        else:
            emb = model.encode_candidate(tok)
        all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"

    # load jobs
    jobs = pd.read_csv("jobs.csv")
    job_ids = jobs["job_id"].tolist()
    job_texts = jobs["job_text"].tolist()

    # load model
    model = TwoTower(model_name=model_name, emb_dim=256).to(device)
    model.load_state_dict(torch.load("two_tower.pt", map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # precompute job embeddings (cache this in your app)
    job_emb = encode_texts(tokenizer, model, job_texts, tower="job", device=device)

    # candidate query (example)
    candidate_text = """Your resume text / LinkedIn text here..."""

    cand_emb = encode_texts(tokenizer, model, [candidate_text], tower="cand", device=device)  # [1, D]

    # cosine similarity since embeddings are normalized
    scores = (cand_emb @ job_emb.T).flatten()
    topk = scores.argsort()[::-1][:10]

    print("Top-10 recommended jobs:")
    for idx in topk:
        print(job_ids[idx], scores[idx], jobs.iloc[idx].get("title", ""))

if __name__ == "__main__":
    main()
