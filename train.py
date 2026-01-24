# Training loop (InfoNCE/ in-batch negatives)
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from two_tower import TwoTower
from techfest.data import PairDataset, TwoTowerCollate

def info_nce_loss(cand_emb: torch.Tensor, job_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    cand_emb: [B, D]
    job_emb:  [B, D]
    Similarity matrix S = cand_emb @ job_emb.T
    Correct matches are diagonal.
    """
    logits = cand_emb @ job_emb.T  # [B, B]
    logits = logits / temperature

    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)         # candidate -> correct job
    loss_j = F.cross_entropy(logits.T, labels)       # job -> correct candidate (symmetric)
    return (loss_i + loss_j) / 2.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load your data ----
    pairs_df = pd.read_csv("pairs.csv")         # candidate_id, job_id, label
    cand_df  = pd.read_csv("candidates.csv")    # candidate_id, resume_text, ...
    job_df   = pd.read_csv("jobs.csv")          # job_id, job_text, ...

    model_name = "distilbert-base-uncased"
    model = TwoTower(model_name=model_name, emb_dim=256).to(device)

    ds = PairDataset(pairs_df, cand_df, job_df)
    collate = TwoTowerCollate(model_name=model_name, max_len=256)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    model.train()
    for epoch in range(3):
        total = 0.0
        for step, batch in enumerate(dl):
            cand = {k: v.to(device) for k, v in batch["cand"].items()}
            job  = {k: v.to(device) for k, v in batch["job"].items()}

            cand_emb = model.encode_candidate(cand)  # [B, D]
            job_emb  = model.encode_job(job)         # [B, D]

            loss = info_nce_loss(cand_emb, job_emb, temperature=0.07)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        print(f"epoch={epoch} avg_loss={total/len(dl):.4f}")

    torch.save(model.state_dict(), "two_tower.pt")
    print("Saved to two_tower.pt")

if __name__ == "__main__":
    main()
