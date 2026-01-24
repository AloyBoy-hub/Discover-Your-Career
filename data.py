# turn CSVs into batched tensors
# data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional


class PairDataset(Dataset):
    """
    Uses ONLY positive pairs for contrastive training (in-batch negatives handle the rest).
    Attaches candidate feature vector if provided.

    Requires:
      cand_df: candidate_id, resume_text
      job_df:  job_id, job_text
      cand_feat_df: candidate_id + numeric feature columns (fixed-length)
    """
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        cand_df: pd.DataFrame,
        job_df: pd.DataFrame,
        cand_feat_df: Optional[pd.DataFrame] = None,
    ):
        self.pairs = pairs_df[pairs_df["label"] == 1].reset_index(drop=True)

        cand_df = cand_df.copy()
        job_df = job_df.copy()
        cand_df["candidate_id"] = cand_df["candidate_id"].astype(str)
        job_df["job_id"] = job_df["job_id"].astype(str)
        self.pairs["candidate_id"] = self.pairs["candidate_id"].astype(str)
        self.pairs["job_id"] = self.pairs["job_id"].astype(str)

        self.cand_text = dict(zip(cand_df["candidate_id"], cand_df["resume_text"].astype(str)))
        self.job_text = dict(zip(job_df["job_id"], job_df["job_text"].astype(str)))

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
            item["cand_feat"] = self.cand_feat.get(cid, self.zero_feat)

        return item


class TwoTowerCollate:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 256, use_feats: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.use_feats = use_feats

    def __call__(self, batch):
        cand_texts = [x["cand_text"] for x in batch]
        job_texts = [x["job_text"] for x in batch]

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
            cand["features"] = feats

        return {"cand": cand, "job": job}
