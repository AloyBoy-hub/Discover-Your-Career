# turn CSVs into batched tensors
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PairDataset(Dataset):
    """
    Uses ONLY positive pairs for contrastive training (in-batch negatives handle the rest).
    Also attaches candidate feature vector if provided.
    """
    def __init__(self, pairs_df: pd.DataFrame, cand_df: pd.DataFrame, job_df: pd.DataFrame, cand_feat_df: pd.DataFrame | None = None):
        self.pairs = pairs_df[pairs_df["label"] == 1].reset_index(drop=True)

        self.cand_text = dict(zip(cand_df["candidate_id"], cand_df["resume_text"]))
        self.job_text  = dict(zip(job_df["job_id"], job_df["job_text"]))

        self.use_feats = cand_feat_df is not None
        if self.use_feats:
            self.feat_cols = [c for c in cand_feat_df.columns if c != "candidate_id"]
            # dict: candidate_id -> np.array(float32)
            self.cand_feat = {}
            for _, r in cand_feat_df.iterrows():
                cid = r["candidate_id"]
                self.cand_feat[cid] = r[self.feat_cols].astype("float32").values

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        cid = row["candidate_id"]
        jid = row["job_id"]
        item = {
            "candidate_id": cid,
            "job_id": jid,
            "cand_text": self.cand_text[cid],
            "job_text": self.job_text[jid],
        }
        if self.use_feats:
            item["cand_feat"] = self.cand_feat[cid]
        return item

class TwoTowerCollate:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 256, use_feats: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.use_feats = use_feats

    def __call__(self, batch):
        cand_texts = [x["cand_text"] for x in batch]
        job_texts  = [x["job_text"] for x in batch]

        cand = self.tok(
            cand_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        job = self.tok(
            job_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        if self.use_feats:
            feats = torch.from_numpy(np.stack([x["cand_feat"] for x in batch])).float()
            cand["features"] = feats  # <-- added

        return {"cand": cand, "job": job}
