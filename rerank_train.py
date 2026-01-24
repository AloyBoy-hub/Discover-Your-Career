# rerank_train.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer

from two_tower import TwoTower
from rerank_model import Reranker

class RerankPairsDataset(Dataset):
    """
    Uses ALL pairs with labels (0/1). Includes candidate features if provided.
    """
    def __init__(self, pairs_df, cand_df, job_df, cand_feat_df=None):
        self.pairs = pairs_df.reset_index(drop=True)
        self.cand_text = dict(zip(cand_df["candidate_id"], cand_df["resume_text"]))
        self.job_text  = dict(zip(job_df["job_id"], job_df["job_text"]))

        self.use_feats = cand_feat_df is not None
        if self.use_feats:
            self.feat_cols = [c for c in cand_feat_df.columns if c != "candidate_id"]
            self.cand_feat = {}
            for _, r in cand_feat_df.iterrows():
                cid = r["candidate_id"]
                self.cand_feat[cid] = r[self.feat_cols].astype("float32").values

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        cid = row["candidate_id"]
        jid = row["job_id"]
        y = float(row["label"])
        item = {
            "cand_text": self.cand_text[cid],
            "job_text": self.job_text[jid],
            "label": y,
        }
        if self.use_feats:
            item["cand_feat"] = self.cand_feat[cid]
        return item

class RerankCollate:
    def __init__(self, model_name="distilbert-base-uncased", max_len=256, use_feats=False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.use_feats = use_feats

    def __call__(self, batch):
        cand_texts = [b["cand_text"] for b in batch]
        job_texts  = [b["job_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        cand = self.tok(cand_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        job  = self.tok(job_texts,  padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        if self.use_feats:
            feats = torch.from_numpy(np.stack([b["cand_feat"] for b in batch])).float()
            cand["features"] = feats

        return {"cand": cand, "job": job, "label": labels}

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    device = pick_device()
    print("DEVICE =", device)

    pairs_df = pd.read_csv("data/pairs.csv")         # candidate_id, job_id, label
    cand_df  = pd.read_csv("data/candidates.csv")    # candidate_id, resume_text
    job_df   = pd.read_csv("data/jobs.csv")          # job_id, job_text
    cand_feat_df = pd.read_csv("data/candidate_features.csv")
    feat_dim = cand_feat_df.shape[1] - 1

    model_name = "distilbert-base-uncased"

    # Load trained retriever (TwoTower)
    tower = TwoTower(model_name=model_name, emb_dim=256, cand_feat_dim=feat_dim).to(device)
    tower.load_state_dict(torch.load("two_tower.pt", map_location=device, weights_only=True))
    tower.eval()  # IMPORTANT: tower used only for embedding

    # Reranker model
    reranker = Reranker(emb_dim=256, hidden=256, dropout=0.2).to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=2e-4, weight_decay=0.01)

    ds = RerankPairsDataset(pairs_df, cand_df, job_df, cand_feat_df=cand_feat_df)
    collate = RerankCollate(model_name=model_name, max_len=256, use_feats=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate)

    reranker.train()
    for epoch in range(3):
        total = 0.0
        for step, batch in enumerate(dl):
            cand = {k: v.to(device) for k, v in batch["cand"].items()}
            job  = {k: v.to(device) for k, v in batch["job"].items()}
            y    = batch["label"].to(device)

            # Embed with tower (no grads)
            with torch.no_grad():
                c = tower.encode_candidate(cand)  # [B,D]
                j = tower.encode_job(job)         # [B,D]

            p = reranker(c, j)  # [B] sigmoid probs

            # BCE loss
            loss = F.binary_cross_entropy(p, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        print(f"epoch={epoch} avg_loss={total/len(dl):.4f}")

    torch.save(reranker.state_dict(), "reranker.pt")
    print("Saved to reranker.pt")

if __name__ == "__main__":
    main()
