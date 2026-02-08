# deploy_utils.py
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


@torch.no_grad()
def encode_job_embeddings(
    tokenizer,
    tower,
    job_texts: List[str],
    device: str,
    batch_size: int = 64,
    max_len: int = 256,
) -> np.ndarray:
    embs = []
    for i in range(0, len(job_texts), batch_size):
        chunk = job_texts[i : i + batch_size]
        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        e = tower.encode_job(tok)
        embs.append(e.detach().cpu().numpy())
    return np.vstack(embs)
