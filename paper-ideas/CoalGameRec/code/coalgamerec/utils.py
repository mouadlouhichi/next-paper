from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def as_1d_float(x: Any, dtype=np.float32) -> np.ndarray:
    """Convert dense/sparse/matrix-like output to a 1D numeric ndarray.

    SciPy sparse matrix multiplication sometimes returns a sparse matrix; NumPy
    matrix operations may return np.matrix. Calling np.asarray directly on a
    sparse matrix creates a 0-D object array, which later raises errors such as
    "float() argument must be a string or a real number, not 'csr_matrix'".
    This helper centralizes the safe conversion.
    """
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=dtype).ravel()


def stable_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.nanmean(x[finite]))
    sd = float(np.nanstd(x[finite]))
    if not np.isfinite(sd) or sd <= eps:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - mu) / (sd + eps)
    out[~finite] = 0.0
    return out.astype(np.float32)


def sparse_fingerprint(mat: sparse.spmatrix) -> str:
    """Deterministic hash of sparse matrix structure and values."""
    csr = mat.tocsr()
    h = hashlib.sha256()
    h.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    h.update(csr.indptr.astype(np.int64, copy=False).tobytes())
    h.update(csr.indices.astype(np.int64, copy=False).tobytes())
    h.update(csr.data.astype(np.float32, copy=False).tobytes())
    return h.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
