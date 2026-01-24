# Two-tower model (PyTorch + Transformers)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over tokens, ignoring padding.
    last_hidden_state: [B, T, H]
    attention_mask: [B, T]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
    return summed / counts

class TextEncoder(nn.Module):
    """
    A text encoder that outputs a normalized embedding vector.
    """
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
        pooled = mean_pool(out.last_hidden_state, attention_mask)  # [B, H]
        emb = self.proj(pooled)                                    # [B, D]
        emb = F.normalize(emb, p=2, dim=-1)                        # cosine space
        return emb

class TwoTower(nn.Module):
    """
    Two independent towers: candidate encoder and job encoder.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", emb_dim: int = 256):
        super().__init__()
        self.cand_encoder = TextEncoder(model_name=model_name, out_dim=emb_dim)
        self.job_encoder = TextEncoder(model_name=model_name, out_dim=emb_dim)

    def encode_candidate(self, cand_batch: dict) -> torch.Tensor:
        return self.cand_encoder(cand_batch["input_ids"], cand_batch["attention_mask"])

    def encode_job(self, job_batch: dict) -> torch.Tensor:
        return self.job_encoder(job_batch["input_ids"], job_batch["attention_mask"])
