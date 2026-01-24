# Two-tower model (PyTorch + Transformers)
# two_tower.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B,1]
    return summed / counts

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.proj(pooled)

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TwoTower(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", emb_dim: int = 256, cand_feat_dim: int = 0):
        super().__init__()
        self.cand_text = TextEncoder(model_name=model_name, out_dim=emb_dim)
        self.job_text  = TextEncoder(model_name=model_name, out_dim=emb_dim)

        # Freeze transformer backbones for speed
        for p in self.cand_text.backbone.parameters():
            p.requires_grad = False
        for p in self.job_text.backbone.parameters():
            p.requires_grad = False

        self.cand_feat_dim = cand_feat_dim
        if cand_feat_dim > 0:
            self.cand_feat = FeatureEncoder(in_dim=cand_feat_dim, out_dim=64)
            self.fuse = nn.Sequential(
                nn.Linear(emb_dim + 64, emb_dim),
                nn.Tanh(),
                nn.LayerNorm(emb_dim),
            )

    def encode_candidate(self, cand_batch: dict) -> torch.Tensor:
        t = self.cand_text(cand_batch["input_ids"], cand_batch["attention_mask"])
        if self.cand_feat_dim > 0:
            f = self.cand_feat(cand_batch["features"])
            t = self.fuse(torch.cat([t, f], dim=-1))
        return F.normalize(t, p=2, dim=-1)

    def encode_job(self, job_batch: dict) -> torch.Tensor:
        t = self.job_text(job_batch["input_ids"], job_batch["attention_mask"])
        return F.normalize(t, p=2, dim=-1)
