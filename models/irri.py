from pathlib import Path
import json, time
import numpy as np

def fit_predict(Ktrain, Kpred, y, Dtrain, Dpred, weights, lam):
    K = np.asarray(Ktrain, dtype=float)
    P = np.asarray(Kpred, dtype=float)
    y = np.asarray(y, dtype=float)
    D = np.asarray(Dtrain, dtype=float)
    DP = np.asarray(Dpred, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(y)
    if n > 651 or n < 1 or K.shape != (n, n) or (D.shape[0] != n) or (P.shape[1] != n) or (DP.shape != (P.shape[0], D.shape[1])) or (w.shape != (n,)):
        raise ValueError('Frozen dimension cap or dimensions violated')
    if not all((np.all(np.isfinite(a)) for a in (K, P, y, D, DP, w))) or np.any(w <= 0) or (not np.isfinite(lam)) or (lam <= 0):
        raise ValueError('Nonfinite or invalid inputs')
    if np.max(np.abs(K - K.T)) > 1e-10 * max(1.0, np.max(np.abs(K))):
        raise ValueError('Asymmetric kernel')
    if np.linalg.matrix_rank(D) != D.shape[1]:
        raise ValueError('Training nuisance design rank deficient')
    eigen = np.linalg.eigvalsh(K)
    if eigen[0] < -64 * np.finfo(float).eps * n * max(1.0, float(np.max(np.abs(eigen)))):
        raise ValueError('Non-PSD kernel')
    A = (K + K.T) / 2
    A[np.diag_indices(n)] += lam * np.sum(w) / w
    L = np.linalg.cholesky(A)
    solved = np.linalg.solve(L.T, np.linalg.solve(L, np.column_stack([y, D])))
    iy, ID = (solved[:, 0], solved[:, 1:])
    beta = np.linalg.solve(D.T @ ID, D.T @ iy)
    alpha = iy - ID @ beta
    pred = P @ alpha + DP @ beta
    normal = float(np.linalg.norm(A @ alpha + D @ beta - y) / max(np.linalg.norm(y), 1.0))
    constraint = float(np.linalg.norm(D.T @ alpha) / max(np.linalg.norm(alpha) * np.linalg.norm(D), 1.0))
    if not np.all(np.isfinite(pred)) or max(normal, constraint) > 1e-08:
        raise RuntimeError('Numerical residual failed')
    return (pred, {'normal_relative_residual': normal, 'nuisance_constraint_relative_residual': constraint, 'n_train': n, 'lambda': lam})

def primal_oracle(X, Xp, y, D, DP, w, lam):
    W = w[:, None]
    A = np.block([[X.T @ (W * X) + lam * np.sum(w) * np.eye(X.shape[1]), X.T @ (W * D)], [D.T @ (W * X), D.T @ (W * D)]])
    rhs = np.r_[X.T @ (w * y), D.T @ (w * y)]
    coef = np.linalg.solve(A, rhs)
    return Xp @ coef[:X.shape[1]] + DP @ coef[X.shape[1]:]
