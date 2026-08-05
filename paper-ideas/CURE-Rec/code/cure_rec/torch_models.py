"""Optional PyTorch BPR-MF with Adam, item bias, and mixed hard negatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cure_rec.models import EvaluationCase, LeaveOneOutSplit, RankingMetrics, evaluate_cases

try:  # Keep the core package usable without PyTorch.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised by availability guard
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class TorchBPRConfig:
    embedding_dim: int = 128
    batch_size: int = 4096
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    max_epochs: int = 200
    evaluation_every: int = 2
    early_stopping_patience: int = 15
    hard_candidates: int = 20
    negative_strategy: str = "hard_mixture"  # uniform | popularity_mixture | hard_mixture
    seed: int = 42


def torch_available() -> bool:
    return torch is not None


def choose_device() -> str:
    if torch is None:
        raise ImportError("Install optional PyTorch support: python -m pip install -e '.[dev,torch]'")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if nn is not None:
    class _BPRMFWithBias(nn.Module):
        def __init__(self, n_users: int, n_items: int, dim: int):
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, dim)
            self.item_embedding = nn.Embedding(n_items, dim)
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
            nn.init.zeros_(self.item_bias.weight)

        def score_indexed(self, users, items):
            return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1) + self.item_bias(items).squeeze(-1)
else:  # pragma: no cover
    _BPRMFWithBias = object


class TorchBPRMFWithBias:
    """PyTorch Adam BPR baseline with shared-candidate evaluation support."""

    name = "torch_bpr_mf_bias"

    def __init__(self, config: TorchBPRConfig = TorchBPRConfig()):
        self.config = config
        self.loss_history: list[dict[str, float]] = []
        self.validation_history: list[dict[str, float]] = []

    def fit(self, interactions: pd.DataFrame, validation_split: LeaveOneOutSplit | None = None, max_eval_users: int = 1_000):
        if torch is None:
            raise ImportError("PyTorch is unavailable. Install with: python -m pip install -e '.[dev,torch]'")
        positives = interactions[interactions["response"] > 0][["user_id", "item_id"]].drop_duplicates()
        self.user_ids = np.sort(interactions["user_id"].astype(int).unique())
        self.item_ids = np.sort(interactions["item_id"].astype(int).unique())
        self.user_to_index = {u: k for k, u in enumerate(self.user_ids)}
        self.item_to_index = {i: k for k, i in enumerate(self.item_ids)}
        pairs = np.asarray([(self.user_to_index[int(r.user_id)], self.item_to_index[int(r.item_id)]) for r in positives.itertuples(index=False)], dtype=np.int64)
        self.user_seen = {k: set() for k in range(len(self.user_ids))}
        for u, i in pairs: self.user_seen[int(u)].add(int(i))
        pairs = pairs[np.asarray([len(self.user_seen[int(u)]) < len(self.item_ids) for u, _ in pairs])]
        if not len(pairs): return self

        torch.manual_seed(self.config.seed)
        np_rng = np.random.default_rng(self.config.seed)
        self.device = choose_device()
        self.model = _BPRMFWithBias(len(self.user_ids), len(self.item_ids), self.config.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        popularity = np.bincount(pairs[:, 1], minlength=len(self.item_ids)).astype(float) + 1.0
        popularity /= popularity.sum()
        pop_tensor = torch.tensor(popularity, dtype=torch.float32, device=self.device)
        best_metric, best_state, stale = -np.inf, None, 0
        updates = 0

        def sample_unseen(users_np: np.ndarray, strategy: str):
            if strategy == "pop":
                candidates = torch.multinomial(pop_tensor, len(users_np), replacement=True).cpu().numpy()
            else:
                candidates = np_rng.integers(len(self.item_ids), size=len(users_np))
            invalid = np.asarray([c in self.user_seen[int(u)] for u, c in zip(users_np, candidates)])
            while invalid.any():
                candidates[invalid] = np_rng.integers(len(self.item_ids), size=int(invalid.sum()))
                invalid = np.asarray([c in self.user_seen[int(u)] for u, c in zip(users_np, candidates)])
            return candidates

        for epoch in range(1, self.config.max_epochs + 1):
            perm = np_rng.permutation(len(pairs))
            epoch_losses = []
            for begin in range(0, len(pairs), self.config.batch_size):
                batch = pairs[perm[begin:begin + self.config.batch_size]]
                users_np, pos_np = batch[:, 0], batch[:, 1]
                mix = np_rng.random(len(batch))
                neg_np = sample_unseen(users_np, "uniform")
                hard_mask = np.zeros(len(batch), dtype=bool)
                if self.config.negative_strategy == "popularity_mixture":
                    pop_mask = mix >= 0.3
                    neg_np[pop_mask] = sample_unseen(users_np[pop_mask], "pop")
                elif self.config.negative_strategy == "hard_mixture":
                    pop_mask = (mix >= 0.3) & (mix < 0.7)
                    neg_np[pop_mask] = sample_unseen(users_np[pop_mask], "pop")
                    hard_mask = mix >= 0.7
                elif self.config.negative_strategy != "uniform":
                    raise ValueError(f"Unknown negative strategy: {self.config.negative_strategy}")
                if hard_mask.any():
                    hard_users = users_np[hard_mask]
                    candidate_matrix = np.stack([sample_unseen(hard_users, "uniform") for _ in range(self.config.hard_candidates)], axis=1)
                    with torch.no_grad():
                        u = torch.tensor(hard_users, dtype=torch.long, device=self.device)
                        cand = torch.tensor(candidate_matrix, dtype=torch.long, device=self.device)
                        scores = self.model.score_indexed(u[:, None].expand_as(cand), cand)
                        top = torch.topk(scores, k=min(5, scores.shape[1]), dim=1).indices
                        choose = torch.randint(top.shape[1], (len(hard_users),), device=self.device)
                        neg_np[hard_mask] = cand[torch.arange(len(hard_users), device=self.device), top[torch.arange(len(hard_users), device=self.device), choose]].cpu().numpy()
                users = torch.tensor(users_np, dtype=torch.long, device=self.device)
                positives_t = torch.tensor(pos_np, dtype=torch.long, device=self.device)
                negatives_t = torch.tensor(neg_np, dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                pos_score = self.model.score_indexed(users, positives_t)
                neg_score = self.model.score_indexed(users, negatives_t)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite BPR loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                updates += len(batch)
                epoch_losses.append(float(loss.detach().cpu()))
            self.loss_history.append({"epoch": float(epoch), "updates": float(updates), "bpr_loss": float(np.mean(epoch_losses))})
            if validation_split is not None and epoch % self.config.evaluation_every == 0:
                metric = evaluate_torch_leave_one_out(self, validation_split, use_validation=True, max_users=max_eval_users).ndcg_at_k
                self.validation_history.append({"epoch": float(epoch), "validation_ndcg_at_10": metric})
                if metric > best_metric + 1e-8:
                    best_metric = metric
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    self.best_validation_epoch = epoch
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.config.early_stopping_patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.restored_checkpoint_epoch = self.best_validation_epoch
        else:
            self.best_validation_epoch = None
            self.restored_checkpoint_epoch = None
        self.updates_completed = updates
        return self

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        user = self.user_to_index.get(int(user_id))
        if user is None: return np.zeros(len(items))
        item_idx = np.asarray([self.item_to_index.get(int(i), -1) for i in items])
        result = np.full(len(items), -np.inf)
        valid = item_idx >= 0
        with torch.no_grad():
            users = torch.full((int(valid.sum()),), user, dtype=torch.long, device=self.device)
            vals = torch.tensor(item_idx[valid], dtype=torch.long, device=self.device)
            result[valid] = self.model.score_indexed(users, vals).detach().cpu().numpy()
        return result


def evaluate_torch_leave_one_out(model: TorchBPRMFWithBias, split: LeaveOneOutSplit, *, use_validation: bool = False, max_users: int = 1_000, k: int = 10) -> RankingMetrics:
    from cure_rec.models import build_shared_candidates
    cases, cold = build_shared_candidates(split, use_validation=use_validation, max_users=max_users)
    return evaluate_cases(model, cases, cold, k=k)
