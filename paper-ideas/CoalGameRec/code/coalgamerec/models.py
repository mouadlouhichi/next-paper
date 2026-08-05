from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    dim: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 4096
    n_neg: int = 4
    seed: int = 42
    device: str = "auto"  # auto, cpu, mps, cuda


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BPRMF(torch.nn.Module):
    """Small frozen-backbone recommender for Mac-local runs.

    This is not the official HCCF port. It is a self-contained backbone used by
    the notebook to exercise the full data -> attribution -> reranking pipeline.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)
        torch.nn.init.normal_(self.user_emb.weight, std=0.02)
        torch.nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)

    @torch.no_grad()
    def full_scores(self, batch_users: np.ndarray, device: torch.device | None = None, chunk_items: int | None = None) -> np.ndarray:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        users_t = torch.as_tensor(batch_users, dtype=torch.long, device=device)
        U = self.user_emb(users_t)
        I = self.item_emb.weight
        if chunk_items is None:
            return (U @ I.T).detach().cpu().numpy()
        outs = []
        for s in range(0, I.shape[0], chunk_items):
            outs.append((U @ I[s : s + chunk_items].T).detach().cpu().numpy())
        return np.concatenate(outs, axis=1)


def _sample_negatives(train_csr: sparse.csr_matrix, users: np.ndarray, n_items: int, rng: np.random.Generator, item_probs: np.ndarray) -> np.ndarray:
    neg = np.empty_like(users)
    for idx, u in enumerate(users):
        seen = set(train_csr[u].indices)
        while True:
            j = int(rng.choice(n_items, p=item_probs))
            if j not in seen:
                neg[idx] = j
                break
    return neg


def train_bprmf(train_df: pd.DataFrame, n_users: int, n_items: int, cfg: TrainConfig, verbose: bool = True) -> BPRMF:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = BPRMF(n_users, n_items, cfg.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_csr = sparse.csr_matrix((np.ones(len(train_df)), (train_df.user, train_df.item)), shape=(n_users, n_items)).tocsr()
    item_deg = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float64)
    item_probs = item_deg + 1e-6
    item_probs /= item_probs.sum()
    users_all = train_df.user.values.astype(np.int64)
    pos_all = train_df.item.values.astype(np.int64)
    n = len(train_df)
    iterator = range(cfg.epochs)
    if verbose:
        iterator = tqdm(iterator, desc="train BPR-MF")
    for _ in iterator:
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            u = users_all[idx]
            pos = pos_all[idx]
            # one negative per positive per step; repeat epochs/steps for n_neg effect
            losses = []
            opt.zero_grad(set_to_none=True)
            for _k in range(cfg.n_neg):
                neg = _sample_negatives(train_csr, u, n_items, rng, item_probs)
                ut = torch.as_tensor(u, dtype=torch.long, device=device)
                pt = torch.as_tensor(pos, dtype=torch.long, device=device)
                nt = torch.as_tensor(neg, dtype=torch.long, device=device)
                loss = -torch.nn.functional.logsigmoid(model(ut, pt) - model(ut, nt)).mean() / cfg.n_neg
                loss.backward()
                losses.append(float(loss.detach().cpu()))
            opt.step()
    return model


def cache_full_scores(model: BPRMF, n_users: int, batch_size: int = 512, chunk_items: int | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    outs = []
    for start in tqdm(range(0, n_users, batch_size), desc="cache base scores"):
        users = np.arange(start, min(n_users, start + batch_size), dtype=np.int64)
        outs.append(model.full_scores(users, device=device, chunk_items=chunk_items))
    return np.vstack(outs).astype(np.float32)

class LightGCN(torch.nn.Module):
    """LightGCN backbone with differentiable index_add propagation.

    This implementation avoids torch sparse kernels so it can run on Apple MPS.
    It is intended as the practical graph-recommender backbone for prospective
    experiments before the validated HCCF port is available.
    """

    def __init__(self, n_users: int, n_items: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_nodes = int(n_users + n_items)
        self.n_layers = int(n_layers)
        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)
        torch.nn.init.normal_(self.user_emb.weight, std=0.02)
        torch.nn.init.normal_(self.item_emb.weight, std=0.02)
        self.register_buffer("edge_src", edge_index[0].long())
        self.register_buffer("edge_dst", edge_index[1].long())
        self.register_buffer("edge_weight", edge_weight.float())

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_e]
        src, dst, w = self.edge_src, self.edge_dst, self.edge_weight
        for _ in range(self.n_layers):
            out = torch.zeros_like(all_e)
            msg = all_e[src] * w.unsqueeze(1)
            out.index_add_(0, dst, msg)
            all_e = out
            embs.append(all_e)
        final = torch.stack(embs, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        U, I = self.propagate()
        return (U[users] * I[items]).sum(-1)

    @torch.no_grad()
    def final_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        U, I = self.propagate()
        return U.detach().cpu().numpy().astype(np.float32), I.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def full_scores(self, batch_users: np.ndarray, device: torch.device | None = None, chunk_items: int | None = None) -> np.ndarray:
        self.eval()
        U_all, I_all = self.propagate()
        users_t = torch.as_tensor(batch_users, dtype=torch.long, device=U_all.device)
        U = U_all[users_t]
        if chunk_items is None:
            return (U @ I_all.T).detach().cpu().numpy()
        outs = []
        for s in range(0, I_all.shape[0], chunk_items):
            outs.append((U @ I_all[s : s + chunk_items].T).detach().cpu().numpy())
        return np.concatenate(outs, axis=1)


def build_lightgcn_graph(train_df: pd.DataFrame, n_users: int, n_items: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    users = train_df.user.values.astype(np.int64)
    items = train_df.item.values.astype(np.int64) + int(n_users)
    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    deg = np.bincount(np.concatenate([src, dst]), minlength=n_users + n_items).astype(np.float32)
    deg[deg == 0] = 1.0
    w = 1.0 / np.sqrt(deg[src] * deg[dst])
    edge_index = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(w, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def train_lightgcn(train_df: pd.DataFrame, n_users: int, n_items: int, cfg: TrainConfig, n_layers: int = 2, verbose: bool = True) -> LightGCN:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    edge_index, edge_weight = build_lightgcn_graph(train_df, n_users, n_items, device)
    model = LightGCN(n_users, n_items, edge_index, edge_weight, cfg.dim, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_csr = sparse.csr_matrix((np.ones(len(train_df)), (train_df.user, train_df.item)), shape=(n_users, n_items)).tocsr()
    item_deg = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float64)
    item_probs = item_deg + 1e-6
    item_probs /= item_probs.sum()
    users_all = train_df.user.values.astype(np.int64)
    pos_all = train_df.item.values.astype(np.int64)
    n = len(train_df)
    iterator = range(cfg.epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"train LightGCN-L{n_layers}")
    for _ in iterator:
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            u = users_all[idx]
            pos = pos_all[idx]
            opt.zero_grad(set_to_none=True)
            for _k in range(cfg.n_neg):
                neg = _sample_negatives(train_csr, u, n_items, rng, item_probs)
                ut = torch.as_tensor(u, dtype=torch.long, device=device)
                pt = torch.as_tensor(pos, dtype=torch.long, device=device)
                nt = torch.as_tensor(neg, dtype=torch.long, device=device)
                loss = -torch.nn.functional.logsigmoid(model(ut, pt) - model(ut, nt)).mean() / cfg.n_neg
                loss.backward()
            opt.step()
    return model


def get_item_embeddings(model) -> np.ndarray | None:
    if hasattr(model, "final_embeddings"):
        return model.final_embeddings()[1]
    if hasattr(model, "item_emb"):
        return model.item_emb.weight.detach().cpu().numpy().astype(np.float32)
    return None
