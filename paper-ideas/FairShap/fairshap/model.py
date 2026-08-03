"""
model.py — hypergraph recommender backbone with Shapley-guided propagation
reweighting (FairShap Sec. 4.4).
"""
from __future__ import annotations
from typing import Dict, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn


class HypergraphGNN(nn.Module):
    """Two-stage hypergraph encoder via sparse incidence matrix ops."""

    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self._A = None
        self._A_w = None

    def encode(self, users_items, item_users, item_weight=None):
        dev = self.item_emb.weight.device
        I = self.item_emb.weight
        if self._A is None:
            rows, cols, vals = [], [], []
            w_vec = item_weight if item_weight is not None else None
            for u, its in users_items.items():
                if not its:
                    continue
                for i in its:
                    rows.append(u); cols.append(i)
                    vals.append(w_vec[i] if w_vec is not None else 1.0)
            idx = torch.tensor([rows, cols], device=dev)
            val = torch.tensor(vals, dtype=torch.float, device=dev)
            self._A = torch.sparse_coo_tensor(idx, val, (self.n_users, self.n_items))
            self._A_w = (item_weight is not None)
        A = self._A
        user_sum = torch.sparse.mm(A, I)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
        user_profile = user_sum / deg[:, None]
        AT = A.transpose(0, 1)
        item_sum = torch.sparse.mm(AT, user_profile)
        item_deg = torch.sparse.sum(AT, dim=1).to_dense().clamp_min(1.0)
        return (item_sum / item_deg[:, None] + I) * 0.5


def build_item_users(train_users, users_items):
    iu = {}
    for u in train_users:
        for i in users_items.get(u, []):
            iu.setdefault(i, []).append(u)
    return iu


def train_hypergraph(users_items, n_items, n_users, dim=32, epochs=20, lr=0.01,
                     batch_size=4096, seed=0, device="cpu", item_weight=None,
                     verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HypergraphGNN(n_users, n_items, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_user_list = list(users_items.keys())
    pos_u, pos_i = [], []
    for u in train_user_list:
        for i in users_items[u]:
            pos_u.append(u); pos_i.append(i)
    pos_u = torch.tensor(pos_u); pos_i = torch.tensor(pos_i)
    item_users = build_item_users(train_user_list, users_items)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(pos_u)); total = 0.0
        for start in range(0, len(pos_u), batch_size):
            idx = perm[start:start + batch_size]
            bu = pos_u[idx]; bi = pos_i[idx]
            bj = torch.randint(0, n_items, (len(idx),))
            I = model.encode(users_items, item_users, item_weight)
            u_emb = model.user_emb(bu)
            pos = (u_emb * I[bi]).sum(1); neg = (u_emb * I[bj]).sum(1)
            loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={total/(len(pos_u)//batch_size+1):.4f}")
    model.eval()
    with torch.no_grad():
        I = model.encode(users_items, item_users, item_weight)
    Q = I.detach().cpu().numpy()
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q)


def train_hypergraph_with_fair_loss(users_items, n_items, n_users, dim=32,
                                    epochs=20, lr=0.01, batch_size=4096, seed=0,
                                    device="cpu", lam_fair=0.1, lam_reg=0.001,
                                    popularity=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HypergraphGNN(n_users, n_items, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_user_list = list(users_items.keys())
    pos_u, pos_i = [], []
    for u in train_user_list:
        for i in users_items[u]:
            pos_u.append(u); pos_i.append(i)
    pos_u = torch.tensor(pos_u); pos_i = torch.tensor(pos_i)
    item_users = build_item_users(train_user_list, users_items)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(pos_u)); total = 0.0
        for start in range(0, len(pos_u), batch_size):
            idx = perm[start:start + batch_size]
            bu = pos_u[idx]; bi = pos_i[idx]
            bj = torch.randint(0, n_items, (len(idx),))
            I = model.encode(users_items, item_users, None)
            u_emb = model.user_emb(bu)
            pos = (u_emb * I[bi]).sum(1); neg = (u_emb * I[bj]).sum(1)
            loss_bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
            if popularity is not None:
                loss_fair = torch.var(I[bi].norm(dim=1)) + torch.var(I[bj].norm(dim=1))
            else:
                loss_fair = torch.zeros((), device=device)
            loss_reg = 0.5 * (model.user_emb.weight.norm()**2 + model.item_emb.weight.norm()**2)
            loss = loss_bpr + lam_fair * loss_fair + lam_reg * loss_reg
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={total/(len(pos_u)//batch_size+1):.4f}")
    model.eval()
    with torch.no_grad():
        I = model.encode(users_items, item_users, None)
    Q = I.detach().cpu().numpy()
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q)
