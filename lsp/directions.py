from __future__ import annotations
import numpy as np

def thin_svd_subspace(task_differences: np.ndarray, rank: int):
    """Return leading left singular vectors of task x output-dimension differences."""
    x = np.asarray(task_differences, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("Expected non-empty [tasks, d_out] difference matrix.")
    # Do not center: the paper specifies SVD of the per-task difference matrix.
    # The paper describes leading left singular vectors in output space; for
    # X=[tasks,d_out], use right singular vectors as output directions.
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    r = min(rank, vh.shape[0])
    basis = vh[:r].T.astype(np.float32)
    return basis, s[:r].astype(np.float32)

def projection_update(weight: np.ndarray, basis: np.ndarray, dose: float):
    """Dense update -dose * P(P^T W), with W [d_out,d_in], P [d_out,r]."""
    w = np.asarray(weight, dtype=np.float32)
    p = np.asarray(basis, dtype=np.float32)
    if w.ndim != 2 or p.ndim != 2 or w.shape[0] != p.shape[0]:
        raise ValueError("Shape mismatch: W must be [d_out,d_in], P [d_out,r].")
    return -float(dose) * (p @ (p.T @ w))

def factorize_update(delta: np.ndarray, rank: int, alpha: float | None = None,
                     dose: float = 1.0, tolerance: float = 1e-5):
    """Factor delta ~= dose*(alpha/r)*B@A; returns A,B and relative error.
    Use SVD truncation. Set alpha=rank so dose is the effective multiplier.
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("Delta must be a matrix.")
    r = min(rank, min(d.shape))
    if alpha is None:
        alpha = r
    # Factors encode delta / dose because runtime applies dose externally.
    target = d if dose == 0 else d / dose
    u, s, vh = np.linalg.svd(target, full_matrices=False)
    root = np.sqrt(s[:r])
    b = (u[:, :r] * root).astype(np.float32)
    a = (root[:, None] * vh[:r, :]).astype(np.float32)
    recon = (alpha / r) * (b @ a) if r else np.zeros_like(d)
    denom = max(np.linalg.norm(target), 1e-12)
    err = float(np.linalg.norm(recon - target) / denom)
    if err > tolerance:
        raise ValueError(f"LoRA reconstruction error {err:.3g} exceeds {tolerance}; increase rank.")
    return a, b, err

def lora_dense_delta(a, b, alpha, rank, dose=1.0):
    return float(dose) * (float(alpha) / int(rank)) * (np.asarray(b) @ np.asarray(a))

def footprint(rank: int, d_in: int, d_out: int):
    return int(rank) * (int(d_in) + int(d_out))
