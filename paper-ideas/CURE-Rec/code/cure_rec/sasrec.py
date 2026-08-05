"""Optional Torch SASRec with the shared CURE-Rec ranking evaluator.

The model is a sequential *external robustness comparator*. It consumes only
chronological training interactions and keeps the same warm-item candidate sets,
cold-target accounting, and evaluator audit used by popularity and BPR-MF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cure_rec.models import LeaveOneOutSplit, RankingMetrics, evaluate_leave_one_out
from cure_rec.torch_models import choose_device, torch_available

try:  # The package remains usable when torch is not installed.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - guarded by torch_available
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class SASRecConfig:
    embedding_dim: int = 128
    max_sequence_length: int = 50
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.20
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 120
    evaluation_every: int = 2
    early_stopping_patience: int = 12
    negative_strategy: str = "popularity_mixture"  # uniform | popularity_mixture
    seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads:
            raise ValueError("SASRec embedding_dim must be divisible by num_heads")
        if self.max_sequence_length < 2:
            raise ValueError("SASRec max_sequence_length must be at least 2")
        if self.num_layers < 1 or self.batch_size < 1:
            raise ValueError("SASRec requires positive layer and batch counts")
        if self.negative_strategy not in {"uniform", "popularity_mixture"}:
            raise ValueError("SASRec negative_strategy must be uniform or popularity_mixture")


if nn is not None:
    class _SASRec(nn.Module):
        def __init__(self, n_items: int, config: SASRecConfig):
            super().__init__()
            # Index zero is reserved for left padding; real items start at one.
            self.item_embedding = nn.Embedding(n_items + 1, config.embedding_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
            self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
            self.layer_norm = nn.LayerNorm(config.embedding_dim)
            self.dropout = nn.Dropout(config.dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embedding_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            nn.init.normal_(self.position_embedding.weight, std=0.02)
            nn.init.zeros_(self.item_bias.weight)
            with torch.no_grad():
                self.item_embedding.weight[0].zero_()

        def encode(self, sequences):
            length = sequences.shape[1]
            positions = torch.arange(length, device=sequences.device).unsqueeze(0)
            embedded = self.item_embedding(sequences) + self.position_embedding(positions)
            embedded = self.dropout(self.layer_norm(embedded))
            # True means blocked for PyTorch's attention mask. The same causal mask
            # is used in training and full-catalog evaluation.
            causal = torch.triu(torch.ones((length, length), device=sequences.device, dtype=torch.bool), diagonal=1)
            padding = sequences.eq(0)
            encoded = self.encoder(embedded, mask=causal, src_key_padding_mask=padding)
            valid_lengths = sequences.ne(0).sum(dim=1).clamp(min=1) - 1
            return encoded[torch.arange(len(sequences), device=sequences.device), valid_lengths]

        def score_items(self, representations, items):
            return representations @ self.item_embedding(items).T + self.item_bias(items).squeeze(-1)
else:  # pragma: no cover
    _SASRec = object


class TorchSASRec:
    """Causal self-attentive sequential recommender trained with BPR pairs."""

    name = "torch_sasrec"

    def __init__(self, config: SASRecConfig = SASRecConfig()):
        self.config = config
        self.loss_history: list[dict[str, float]] = []
        self.validation_history: list[dict[str, float]] = []

    @staticmethod
    def _ordered_sequences(interactions: pd.DataFrame) -> dict[int, list[int]]:
        positive = interactions[interactions["response"] > 0].copy()
        required = {"user_id", "item_id", "timestamp"}
        missing = required.difference(positive.columns)
        if missing:
            raise ValueError(f"SASRec requires chronological columns; missing {sorted(missing)}")
        if positive["timestamp"].isna().any():
            positive = positive.dropna(subset=["timestamp"])
        ordered = positive.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
        return {
            int(user): [int(item) for item in frame["item_id"].tolist()]
            for user, frame in ordered.groupby("user_id", sort=False)
        }

    def _pad_prefixes(self, example_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = self.config.max_sequence_length
        sequences = np.zeros((len(example_indices), length), dtype=np.int64)
        positive = np.empty(len(example_indices), dtype=np.int64)
        users = np.empty(len(example_indices), dtype=np.int64)
        for destination, example_index in enumerate(example_indices):
            sequence_key, target_position = self.examples[int(example_index)]
            values = self.sequence_values[sequence_key]
            prefix = values[max(0, target_position - length):target_position]
            sequences[destination, -len(prefix):] = prefix
            positive[destination] = values[target_position]
            users[destination] = sequence_key
        return sequences, positive, users

    def _sample_negative(self, users: np.ndarray, rng: np.random.Generator, *, popularity: np.ndarray) -> np.ndarray:
        if len(users) == 0:
            return np.empty(0, dtype=np.int64)
        if self.config.negative_strategy == "popularity_mixture":
            use_popularity = rng.random(len(users)) >= 0.3
            sampled = rng.integers(1, self.n_items + 1, size=len(users), dtype=np.int64)
            count = int(use_popularity.sum())
            if count:
                sampled[use_popularity] = rng.choice(np.arange(1, self.n_items + 1), size=count, replace=True, p=popularity)
        else:
            sampled = rng.integers(1, self.n_items + 1, size=len(users), dtype=np.int64)
        invalid = np.asarray([item in self.user_seen[int(user)] for user, item in zip(users, sampled, strict=True)])
        while invalid.any():
            sampled[invalid] = rng.integers(1, self.n_items + 1, size=int(invalid.sum()), dtype=np.int64)
            invalid = np.asarray([item in self.user_seen[int(user)] for user, item in zip(users, sampled, strict=True)])
        return sampled

    def fit(self, interactions: pd.DataFrame, validation_split: LeaveOneOutSplit | None = None, max_eval_users: int = 1_000):
        if not torch_available():
            raise ImportError("PyTorch is unavailable. Install with: python -m pip install -e '.[dev,torch]'")
        raw_sequences = self._ordered_sequences(interactions)
        self.user_ids = np.sort(interactions["user_id"].astype(int).unique())
        self.item_ids = np.sort(interactions["item_id"].astype(int).unique())
        self.item_to_index = {item: index + 1 for index, item in enumerate(self.item_ids)}
        self.index_to_item = {index + 1: item for index, item in enumerate(self.item_ids)}
        self.n_items = len(self.item_ids)
        self.sequence_values: dict[int, np.ndarray] = {
            user: np.asarray([self.item_to_index[item] for item in sequence if item in self.item_to_index], dtype=np.int64)
            for user, sequence in raw_sequences.items()
        }
        self.user_seen = {user: set(values.tolist()) for user, values in self.sequence_values.items()}
        self.examples = [(user, position) for user, values in self.sequence_values.items() for position in range(1, len(values))]
        if not self.examples:
            raise ValueError("SASRec needs at least two chronological training positives for one user")

        torch.manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        self.device = choose_device()
        self.model = _SASRec(self.n_items, self.config).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        counts = np.bincount(np.concatenate(list(self.sequence_values.values())), minlength=self.n_items + 1)[1:].astype(float) + 1.0
        popularity = counts / counts.sum()
        best_metric, best_state, stale, updates = -np.inf, None, 0, 0

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            order = rng.permutation(len(self.examples))
            losses = []
            for begin in range(0, len(order), self.config.batch_size):
                sequence, positive, users = self._pad_prefixes(order[begin:begin + self.config.batch_size])
                negative = self._sample_negative(users, rng, popularity=popularity)
                sequence_t = torch.tensor(sequence, dtype=torch.long, device=self.device)
                positive_t = torch.tensor(positive, dtype=torch.long, device=self.device)
                negative_t = torch.tensor(negative, dtype=torch.long, device=self.device)
                representation = self.model.encode(sequence_t)
                pos_score = (representation * self.model.item_embedding(positive_t)).sum(dim=-1) + self.model.item_bias(positive_t).squeeze(-1)
                neg_score = (representation * self.model.item_embedding(negative_t)).sum(dim=-1) + self.model.item_bias(negative_t).squeeze(-1)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite SASRec loss at epoch {epoch}")
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                updates += len(sequence); losses.append(float(loss.detach().cpu()))
            self.loss_history.append({"epoch": float(epoch), "updates": float(updates), "bpr_loss": float(np.mean(losses))})
            if validation_split is not None and epoch % self.config.evaluation_every == 0:
                metric = evaluate_leave_one_out(self, validation_split, use_validation=True, max_users=max_eval_users).ndcg_at_k
                self.validation_history.append({"epoch": float(epoch), "validation_ndcg_at_10": metric})
                if metric > best_metric + 1e-8:
                    best_metric = metric
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    self.best_validation_epoch = epoch; stale = 0
                else:
                    stale += 1
                    if stale >= self.config.early_stopping_patience:
                        break
        if best_state is None:
            self.best_validation_epoch = None; self.restored_checkpoint_epoch = None
        else:
            self.model.load_state_dict(best_state)
            self.restored_checkpoint_epoch = self.best_validation_epoch
        self.updates_completed = updates
        return self

    def _sequence_for_user(self, user_id: int) -> np.ndarray | None:
        values = self.sequence_values.get(int(user_id))
        if values is None or not len(values):
            return None
        sequence = np.zeros((1, self.config.max_sequence_length), dtype=np.int64)
        tail = values[-self.config.max_sequence_length:]
        sequence[0, -len(tail):] = tail
        return sequence

    def prepare_evaluation(self, user_ids: list[int]) -> None:
        """Batch sequential encodings for a shared-candidate evaluation pass."""
        self._evaluation_cache: dict[int, object] = {}
        unique = [user for user in dict.fromkeys(int(user) for user in user_ids) if self._sequence_for_user(user) is not None]
        if not unique:
            return
        self.model.eval()
        chunk_size = max(1, min(2048, self.config.batch_size))
        with torch.no_grad():
            for begin in range(0, len(unique), chunk_size):
                users = unique[begin:begin + chunk_size]
                sequences = np.concatenate([self._sequence_for_user(user) for user in users], axis=0)
                representation = self.model.encode(torch.tensor(sequences, dtype=torch.long, device=self.device))
                for offset, user in enumerate(users):
                    self._evaluation_cache[user] = representation[offset:offset + 1]

    def finish_evaluation(self) -> None:
        self._evaluation_cache = {}

    def score(self, user_id: int, items: np.ndarray) -> np.ndarray:
        sequence = self._sequence_for_user(user_id)
        output = np.full(len(items), -np.inf, dtype=float)
        if sequence is None:
            return output
        indices = np.asarray([self.item_to_index.get(int(item), -1) for item in items], dtype=np.int64)
        valid = indices > 0
        if not valid.any():
            return output
        self.model.eval()
        with torch.no_grad():
            representation = getattr(self, "_evaluation_cache", {}).get(int(user_id))
            if representation is None:
                representation = self.model.encode(torch.tensor(sequence, dtype=torch.long, device=self.device))
            candidates = torch.tensor(indices[valid], dtype=torch.long, device=self.device)
            output[valid] = self.model.score_items(representation, candidates).detach().cpu().numpy()[0]
        return output


def evaluate_sasrec_leave_one_out(model: TorchSASRec, split: LeaveOneOutSplit, *, use_validation: bool = False, max_users: int = 1_000) -> RankingMetrics:
    """Named convenience wrapper; evaluator semantics remain shared across models."""
    return evaluate_leave_one_out(model, split, use_validation=use_validation, max_users=max_users)
