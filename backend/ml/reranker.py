# rerank_model.py
import torch
import torch.nn as nn

class Reranker(nn.Module):
    def __init__(self, emb_dim: int = 256, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        in_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand_emb: torch.Tensor, job_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cand_emb, job_emb, torch.abs(cand_emb - job_emb), cand_emb * job_emb], dim=-1)
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)
