# Data + Collate (tokenize candidate/ job text separatelu)
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PairDataset(Dataset):
    """
    Uses ONLY positive pairs for contrastive training (in-batch negatives handle the rest).
    """
    def __init__(self, pairs_df: pd.DataFrame, cand_df: pd.DataFrame, job_df: pd.DataFrame):
        self.pairs = pairs_df[pairs_df["label"] == 1].reset_index(drop=True)

        # quick lookups
        self.cand_text = dict(zip(cand_df["candidate_id"], cand_df["resume_text"]))
        self.job_text = dict(zip(job_df["job_id"], job_df["job_text"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        cid = row["candidate_id"]
        jid = row["job_id"]
        return {
            "candidate_id": cid,
            "job_id": jid,
            "cand_text": self.cand_text[cid],
            "job_text": self.job_text[jid],
        }

class TwoTowerCollate:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 256):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

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

        return {"cand": cand, "job": job}
