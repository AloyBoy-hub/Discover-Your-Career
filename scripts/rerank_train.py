# rerank_train.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys

# Ensure root modules (two_tower, rerank_model, etc.) can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_tower import TwoTower
from rerank_model import Reranker
from deploy_utils import ensure_dir, save_json, save_npy, encode_job_embeddings


class RerankPairsDataset(Dataset):
    """
    Uses ALL pairs with labels (0/1).
    Requires:
      cand_df: candidate_id, resume_text
      job_df: job_id, job_text
      cand_feat_df: candidate_id + numeric features
    """
    def __init__(self, pairs_df, cand_df, job_df, cand_feat_df=None):
        self.pairs = pairs_df.reset_index(drop=True)

        cand_df = cand_df.copy()
        job_df = job_df.copy()
        cand_df["candidate_id"] = cand_df["candidate_id"].astype(str)
        job_df["job_id"] = job_df["job_id"].astype(str)
        self.pairs["candidate_id"] = self.pairs["candidate_id"].astype(str)
        self.pairs["job_id"] = self.pairs["job_id"].astype(str)

        self.cand_text = dict(zip(cand_df["candidate_id"], cand_df["resume_text"].astype(str)))
        self.job_text  = dict(zip(job_df["job_id"], job_df["job_text"].astype(str)))

        self.use_feats = cand_feat_df is not None
        self.feat_cols = []
        self.cand_feat = {}
        self.zero_feat = None

        if self.use_feats:
            cand_feat_df = cand_feat_df.copy()
            cand_feat_df["candidate_id"] = cand_feat_df["candidate_id"].astype(str)
            self.feat_cols = [c for c in cand_feat_df.columns if c != "candidate_id"]
            feat_dim = len(self.feat_cols)
            self.zero_feat = np.zeros((feat_dim,), dtype=np.float32)

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

        item = {"cand_text": self.cand_text[cid], "job_text": self.job_text[jid], "label": y}

        if self.use_feats:
            item["cand_feat"] = self.cand_feat.get(cid, self.zero_feat)

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

    pairs_df = pd.read_csv("data/pairs.csv")
    cand_df  = pd.read_csv("data/candidates.csv")              # <-- FIXED
    job_df   = pd.read_csv("data/jobs.csv")
    cand_feat_df = pd.read_csv("data/candidate_features.csv")  # <-- FIXED

    feature_columns = [c for c in cand_feat_df.columns if c != "candidate_id"]
    feat_dim = len(feature_columns)

    model_name = "distilbert-base-uncased"
    max_len = 256
    batch_size = 64

    emb_dim = 256
    rerank_hidden = 256
    rerank_dropout = 0.2

    # Load TwoTower
    tower = TwoTower(model_name=model_name, emb_dim=emb_dim, cand_feat_dim=feat_dim).to(device)
    tower.load_state_dict(torch.load("two_tower.pt", map_location=device, weights_only=True))
    tower.eval()

    # Train reranker
    reranker = Reranker(emb_dim=emb_dim, hidden=rerank_hidden, dropout=rerank_dropout).to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=2e-4, weight_decay=0.01)

    ds = RerankPairsDataset(pairs_df, cand_df, job_df, cand_feat_df=cand_feat_df)
    collate = RerankCollate(model_name=model_name, max_len=max_len, use_feats=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    reranker.train()
    reranker.train()
    # Reduced for speed
    for epoch in range(1):
        total = 0.0
        for step, batch in enumerate(dl):
            cand = {k: v.to(device) for k, v in batch["cand"].items()}
            job  = {k: v.to(device) for k, v in batch["job"].items()}
            y    = batch["label"].to(device)

            with torch.no_grad():
                c = tower.encode_candidate(cand)
                j = tower.encode_job(job)

            p = reranker(c, j)
            loss = F.binary_cross_entropy(p, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            # Speed hack
            if step >= 5:
                break

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        print(f"epoch={epoch} avg_loss={total/len(dl):.4f}")

    # ----------------------------
    # Save deployment bundle
    # ----------------------------
    ART_DIR = "artifacts"
    ensure_dir(ART_DIR)

    torch.save(reranker.state_dict(), os.path.join(ART_DIR, "reranker.pt"))
    torch.save(tower.state_dict(), os.path.join(ART_DIR, "two_tower.pt"))

    tok_dir = os.path.join(ART_DIR, "tokenizer")
    ensure_dir(tok_dir)
    AutoTokenizer.from_pretrained(model_name, use_fast=True).save_pretrained(tok_dir)

    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    job_texts = job_df["job_text"].astype(str).tolist()
    job_emb = encode_job_embeddings(
        tokenizer,
        tower,
        job_texts,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
    )
    save_npy(os.path.join(ART_DIR, "job_emb.npy"), job_emb)

    deploy_cfg = {
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "feature_columns": feature_columns,
        "tower": {"emb_dim": emb_dim, "cand_feat_dim": feat_dim, "weights": "two_tower.pt"},
        "reranker": {"emb_dim": emb_dim, "hidden": rerank_hidden, "dropout": rerank_dropout, "weights": "reranker.pt"},
        "tokenizer_dir": "tokenizer",
        "job_emb_path": "job_emb.npy",
    }
    save_json(os.path.join(ART_DIR, "deploy_config.json"), deploy_cfg)

    print(f"Saved deployment bundle to: {ART_DIR}/")

if __name__ == "__main__":
    main()
