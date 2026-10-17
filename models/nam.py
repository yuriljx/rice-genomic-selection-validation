from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
DEFAULT_LAMBDAS = (0.0001, 0.001, 0.01, 0.1)
MAX_TRAIN = 2000
MAX_KERNEL_ELEMENTS = 8000000
RESIDUAL_LIMIT = 1e-08
SYMMETRY_LIMIT = 1e-10

class NumericalError(RuntimeError):
    pass

def _finite_array(value, ndim, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != ndim or not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must be a finite {ndim}-dimensional array')
    return out

def make_weights(n_source, k_target, *, strategy='pooled', eta=None):
    if int(n_source) != n_source or int(k_target) != k_target:
        raise ValueError('sample counts must be integers')
    n_source, k_target = (int(n_source), int(k_target))
    if min(n_source, k_target) < 0 or n_source + k_target < 1:
        raise ValueError('sample counts must be nonnegative with positive total')
    if strategy == 'pooled':
        return np.full(n_source + k_target, 1.0 / (n_source + k_target))
    if strategy == 'target_only':
        if k_target == 0:
            raise ValueError('target-only requires target labels')
        return np.r_[np.zeros(n_source), np.full(k_target, 1.0 / k_target)]
    if strategy != 'weighted':
        raise ValueError('strategy must be pooled, weighted, or target_only')
    if k_target == 0:
        if eta not in (None, 0, 0.0):
            raise ValueError('k=0 requires eta=0')
        return np.full(n_source, 1.0 / n_source)
    if eta is None or not np.isfinite(eta) or (not 0.0 <= eta <= 1.0):
        raise ValueError('eta must be finite and between 0 and 1')
    if n_source == 0:
        if eta != 1.0:
            raise ValueError('without source rows eta must equal 1')
        return np.full(k_target, 1.0 / k_target)
    return np.r_[np.full(n_source, (1.0 - eta) / n_source), np.full(k_target, eta / k_target)]

def _features(za, zb):
    za = _finite_array(za, 2, 'Za')
    same = zb is None
    zb = za if same else _finite_array(zb, 2, 'Zb')
    if za.shape[1] == 0 or za.shape[1] != zb.shape[1]:
        raise ValueError('feature dimensions must agree and be nonzero')
    if za.shape[0] * zb.shape[0] > MAX_KERNEL_ELEMENTS:
        raise ValueError('requested kernel exceeds bounded element limit')
    if same and za.shape[0] > MAX_TRAIN:
        raise ValueError('self kernel exceeds bounded training size')
    return (za, zb)

def linear_kernel(za, zb=None):
    za, zb = _features(za, zb)
    kernel = za @ zb.T / za.shape[1]
    if not np.all(np.isfinite(kernel)):
        raise NumericalError('nonfinite linear kernel')
    return kernel

def rbf_kernel(za, zb=None, *, h=1.0):
    if not np.isfinite(h) or h <= 0:
        raise ValueError('h must be finite and positive')
    za, zb = _features(za, zb)
    na, nb = (np.einsum('ij,ij->i', za, za), np.einsum('ij,ij->i', zb, zb))
    distance = -2.0 * (za @ zb.T)
    distance += na[:, None]
    distance += nb[None, :]
    if not np.all(np.isfinite(distance)):
        raise NumericalError('nonfinite squared distances')
    distance_scale = max(float(np.max(na, initial=0)), float(np.max(nb, initial=0)), np.finfo(float).tiny)
    tolerance = 128 * np.finfo(float).eps * za.shape[1] * distance_scale
    if float(np.min(distance, initial=0)) < -tolerance:
        raise NumericalError('squared distance below roundoff tolerance')
    np.maximum(distance, 0.0, out=distance)
    distance /= -2.0 * h * h
    np.exp(distance, out=distance)
    if not np.all(np.isfinite(distance)):
        raise NumericalError('nonfinite RBF kernel')
    return distance

def forward_substitute(lower, rhs):
    vector = np.ndim(rhs) == 1
    out = np.array(rhs, dtype=np.float64, copy=True)
    if vector:
        out = out[:, None]
    for i in range(lower.shape[0]):
        out[i] -= lower[i, :i] @ out[:i]
        out[i] /= lower[i, i]
    return out[:, 0] if vector else out

def backward_substitute_transpose(lower, rhs):
    vector = np.ndim(rhs) == 1
    out = np.array(rhs, dtype=np.float64, copy=True)
    if vector:
        out = out[:, None]
    for i in range(lower.shape[0] - 1, -1, -1):
        out[i] -= lower[i + 1:, i] @ out[i + 1:]
        out[i] /= lower[i, i]
    return out[:, 0] if vector else out

def cholesky_solve(lower, rhs):
    return backward_substitute_transpose(lower, forward_substitute(lower, rhs))

def _prepare(ktrain, kpred, y, weights, lambdas):
    kt = _finite_array(ktrain, 2, 'Ktrain')
    kp = _finite_array(kpred, 2, 'Kpred')
    y = _finite_array(y, 1, 'y')
    w = _finite_array(weights, 1, 'weights')
    n = len(y)
    if not 1 <= n <= MAX_TRAIN or kt.shape != (n, n) or kp.shape[1] != n or (w.shape != (n,)):
        raise ValueError('incompatible dimensions or training size outside 1..2000')
    if kp.size > MAX_KERNEL_ELEMENTS:
        raise ValueError('prediction kernel exceeds bounded element limit')
    if np.any(w < 0) or not np.isclose(float(w.sum()), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError('weights must be nonnegative and sum to 1; no silent renormalization')
    lambdas = tuple((float(x) for x in lambdas))
    if not lambdas or len(set(lambdas)) != len(lambdas) or any((not np.isfinite(x) or x <= 0 for x in lambdas)):
        raise ValueError('lambdas must be distinct, finite, and positive')
    symmetry_error = float(np.linalg.norm(kt - kt.T)) / max(float(np.linalg.norm(kt)), np.finfo(float).tiny)
    if symmetry_error > SYMMETRY_LIMIT:
        raise ValueError(f'Ktrain is asymmetric: relative error {symmetry_error}')
    kt = 0.5 * (kt + kt.T)
    d = np.sqrt(w)
    g0 = d[:, None] * kt * d[None, :]
    h = d * y
    return (kp, d, g0, h, lambdas, symmetry_error)

def _residual(g0, d, h, c, intercept, lam, cancellation_scale):
    residual = g0 @ c + lam * c + d * intercept - h
    absolute = float(np.linalg.norm(residual))
    relative = absolute / max(float(np.linalg.norm(h)), np.finfo(float).tiny)
    constraint = abs(float(d @ c))
    constraint_relative = constraint / max(cancellation_scale, np.finfo(float).tiny)
    values = (absolute, relative, constraint, constraint_relative)
    if not all((np.isfinite(x) for x in values)) or max(relative, constraint_relative) > RESIDUAL_LIMIT:
        raise NumericalError(f'normal-equation residual failed for lambda={lam}: {values}')
    return {'equation_absolute_residual': absolute, 'equation_relative_residual': relative, 'intercept_constraint_absolute_residual': constraint, 'intercept_constraint_relative_residual': constraint_relative}

def fit_predict_path(ktrain, kpred, y, weights, lambdas=DEFAULT_LAMBDAS, *, method='eigh'):
    start = time.perf_counter()
    kp, d, g0, h, lambdas, symmetry_error = _prepare(ktrain, kpred, y, weights, lambdas)
    if method not in ('eigh', 'cholesky'):
        raise ValueError('method must be eigh or cholesky')
    diag = {'method': method, 'n_train': len(d), 'n_predict': kp.shape[0], 'weight_sum': float(np.sum(np.asarray(weights))), 'symmetry_relative_error': symmetry_error, 'validation_seconds': time.perf_counter() - start, 'factor_seconds': 0.0, 'solve_seconds': 0.0, 'prediction_seconds': 0.0, 'residual_check_seconds': 0.0, 'factorization_count': 0, 'rhs_solved_count': 2 * len(lambdas), 'triangular_substitution_sweeps': 0, 'per_lambda': {}, 'jitter': 0.0, 'rank_truncation': False, 'eigenvalues_clipped': False}
    predictions, intercepts = ({}, {})
    if method == 'eigh':
        stamp = time.perf_counter()
        e, q = np.linalg.eigh(g0)
        diag['factor_seconds'] = time.perf_counter() - stamp
        diag['factorization_count'] = 1
        spectral_scale = max(float(np.max(np.abs(e))), np.finfo(float).tiny)
        psd_tolerance = 64 * len(d) * np.finfo(float).eps * spectral_scale
        diag.update({'weighted_kernel_min_eigenvalue': float(e[0]), 'weighted_kernel_max_eigenvalue': float(e[-1]), 'psd_roundoff_tolerance': psd_tolerance})
        if not np.all(np.isfinite(e)) or e[0] < -psd_tolerance:
            raise NumericalError('weighted kernel is not PSD within roundoff tolerance')
        stamp = time.perf_counter()
        qd, qh = (q.T @ d, q.T @ h)
        diag['solve_seconds'] += time.perf_counter() - stamp
    for lam in lambdas:
        key = str(float(lam))
        stamp = time.perf_counter()
        condition = None
        if method == 'eigh':
            shifted = e + lam
            if np.any(shifted <= 0) or not np.all(np.isfinite(shifted)):
                raise NumericalError('regularized spectrum is not strictly positive')
            vq, uq = (qd / shifted, qh / shifted)
            denominator = float(qd @ vq)
            numerator = float(qd @ uq)
            if not np.isfinite(denominator) or denominator <= 0:
                raise NumericalError('invalid intercept Schur denominator')
            intercept = numerator / denominator
            c = q @ (uq - intercept * vq)
            cancellation_scale = float(np.linalg.norm(d)) * (float(np.linalg.norm(uq)) + abs(intercept) * float(np.linalg.norm(vq)))
            condition = float(shifted[-1] / shifted[0])
        else:
            system = g0.copy()
            system[np.diag_indices_from(system)] += lam
            factor_stamp = time.perf_counter()
            lower = np.linalg.cholesky(system)
            factor_elapsed = time.perf_counter() - factor_stamp
            diag['factor_seconds'] += factor_elapsed
            diag['factorization_count'] += 1
            uv = cholesky_solve(lower, np.column_stack((h, d)))
            u, v = (uv[:, 0], uv[:, 1])
            denominator = float(d @ v)
            if not np.isfinite(denominator) or denominator <= 0:
                raise NumericalError('invalid intercept Schur denominator')
            intercept = float(d @ u) / denominator
            c = u - intercept * v
            cancellation_scale = float(np.linalg.norm(d)) * (float(np.linalg.norm(u)) + abs(intercept) * float(np.linalg.norm(v)))
            diag['triangular_substitution_sweeps'] += 2
        elapsed = time.perf_counter() - stamp
        if method == 'cholesky':
            elapsed -= factor_elapsed
        diag['solve_seconds'] += elapsed
        stamp = time.perf_counter()
        per_lambda = _residual(g0, d, h, c, intercept, lam, cancellation_scale)
        diag['residual_check_seconds'] += time.perf_counter() - stamp
        per_lambda.update({'shifted_condition_number': condition, 'intercept_schur_denominator': denominator})
        diag['per_lambda'][key] = per_lambda
        stamp = time.perf_counter()
        pred = intercept + kp @ (d * c)
        if not np.isfinite(intercept) or not np.all(np.isfinite(pred)):
            raise NumericalError('nonfinite predictions or intercept')
        diag['prediction_seconds'] += time.perf_counter() - stamp
        predictions[key], intercepts[key] = (pred, float(intercept))
    diag['total_seconds'] = time.perf_counter() - start
    return {'predictions': predictions, 'intercepts': intercepts, 'diagnostics': diag}

def fit_predict(ktrain, kpred, y, weights, lam, *, return_info=False):
    result = fit_predict_path(ktrain, kpred, y, weights, (lam,), method='cholesky')
    key = str(float(lam))
    if return_info:
        return {'predictions': result['predictions'][key], 'intercept': result['intercepts'][key], 'diagnostics': result['diagnostics']}
    return result['predictions'][key]
