"""
lightgcn.py — a LightGCN encoder for the CAVI recommender backbone.

LightGCN (He et al., 2020) learns user and item embeddings by message passing
over the user-item interaction graph. Crucially for CAVI, the ITEM embeddings it
produces are highly discriminative (they encode higher-order connectivity), so
which interactions are kept/pruned/reweighted in a user's profile genuinely
changes the profile and the ranking — unlike BPR item-factors, where every
interaction contributes ~1/N and single-intervention effects are tiny.

Interface: train LightGCN to produce item embeddings Q (n_items x d), then reuse
the same CAVI history-conditioned evaluation (profile = weighted aggregate of the
item embeddings of the active interactions).

Pure PyTorch, CPU-capable; on your M4 it will use MPS automatically.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 32, n_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.register_buffer("norm_adj", None)
        self.dropout = dropout

    def _build_norm_adj(self, edges_u: np.ndarray, edges_i: np.ndarray, n_nodes: int):
        """Symmetric normalized adjacency over user+item nodes (block matrix)."""
        import scipy.sparse as sp
        idx = np.concatenate([edges_u, edges_i + self.n_users])
        data = np.ones(len(idx))
        adj = sp.coo_matrix((data, (idx, idx)), shape=(n_nodes, n_nodes))
        # split into user-item off-diagonal (LightGCN only uses U-I blocks)
        n = self.n_users
        m = self.n_items
        adj_ui = sp.coo_matrix((np.ones(len(edges_u)), (edges_u, edges_i)),
                               shape=(n, m)).astype(np.float32)
        # build [0 A; A^T 0]
        A = sp.bmat([[None, adj_ui], [adj_ui.T, None]], format="coo").tocsc()
        deg = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        d_inv_sqrt[deg == 0] = 0
        d_mat = sp.diags(d_inv_sqrt)
        norm = (d_mat @ A @ d_mat).tocoo()
        row = torch.LongTensor(norm.row)
        col = torch.LongTensor(norm.col)
        val = torch.FloatTensor(norm.data)
        idx_t = torch.stack([row, col])
        shape = torch.Size((n_nodes, n_nodes))
        return torch.sparse_coo_tensor(idx_t, val, shape)

    def fit_norm_adj(self, edges_u, edges_i):
        n_nodes = self.n_users + self.n_items
        self.register_buffer("norm_adj", self._build_norm_adj(edges_u, edges_i, n_nodes))

    def forward(self, return_item_emb: bool = False):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = [ego]
        emb = ego
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            all_emb.append(emb)
        final = torch.mean(torch.stack(all_emb), dim=0)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        if return_item_emb:
            return user_final, item_final
        return user_final, item_final


def train_lightgcn(edges_u: np.ndarray, edges_i: np.ndarray, n_users: int,
                   n_items: int, dim: int = 32, n_layers: int = 2,
                   epochs: int = 30, lr: float = 0.01, batch_size: int = 4096,
                   neg_per_pos: int = 1, seed: int = 0, device: str = "cpu",
                   verbose: bool = True) -> np.ndarray:
    """
    Train LightGCN via BPR loss. Returns L2-normalized ITEM embeddings (n_items x d).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LightGCN(n_users, n_items, dim, n_layers).to(device)
    model.fit_norm_adj(edges_u, edges_i)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_edges = len(edges_u)
    # pos pairs
    pos_u = torch.LongTensor(edges_u)
    pos_i = torch.LongTensor(edges_i)
    all_items = torch.arange(n_items)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_edges)
        total_loss = 0.0
        for start in range(0, n_edges, batch_size):
            idx = perm[start:start + batch_size]
            bu = pos_u[idx]; bi = pos_i[idx]
            bj = torch.randint(0, n_items, (len(idx),))
            u_e, i_e = model()
            u = u_e[bu]; i_pos = i_e[bi]; i_neg = i_e[bj]
            pos = (u * i_pos).sum(1)
            neg = (u * i_neg).sum(1)
            loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={total_loss/(len(range(0,n_edges,batch_size))):.4f}")
    model.eval()
    with torch.no_grad():
        _, item_final = model()
    Q = item_final.detach().cpu().numpy()
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return np.ascontiguousarray(Q)


def main():
    """Quick smoke test on synthetic data."""
    rng = np.random.default_rng(0)
    n_u, n_i = 200, 300
    edges_u = rng.integers(0, n_u, 1000)
    edges_i = rng.integers(0, n_i, 1000)
    Q = train_lightgcn(edges_u, edges_i, n_u, n_i, epochs=3, batch_size=256,
                       verbose=True)
    print("Q shape:", Q.shape, "norm range:", Q.min(), Q.max())


if __name__ == "__main__":
    main()
