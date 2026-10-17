from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env] = '1'
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
ARMS = ('ridge0', 'rkhs0', 'weighted25', 'weighted100', 'pool100', 'pool_rkhs100', 'source_tuned_targetonly100')
ETAS = (0.05, 0.2, 0.5)
H2_MULTIPLIERS = (0.5, 1.0, 2.0)
TIE_SEED = 2026100503
PATH_METHOD = 'cholesky'

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()

def _codes(raw):
    raw = np.asarray(raw)
    _require(raw.ndim == 2 and raw.shape[1] > 0 and (raw.shape[1] <= 2048) and np.issubdtype(raw.dtype, np.integer), 'Raw codes must be a bounded integer marker matrix')
    _require(np.all((raw >= 0) & (raw <= 3)), 'Genotype code outside frozen 0..3 contract')
    return raw

def fit_preprocessor(source_core_codes, *, strict=True):
    raw = _codes(source_core_codes)
    if strict:
        _require(raw.shape[0] == 1200, 'Preprocessing requires frozen 1200-row NAM source core')
    _require(raw.shape[0] > 1, 'Empty source core')
    dosage = np.array([0.0, 2.0, 1.0, np.nan])[raw]
    _require(np.all(np.any(np.isfinite(dosage), axis=0)), 'All-missing source core marker')
    mean = np.nanmean(dosage, axis=0)
    filled = np.where(np.isnan(dosage), mean, dosage)
    variance = np.mean((filled - mean) ** 2, axis=0)
    active = variance > 1e-12
    _require(np.any(active), 'Degenerate source-only marker panel')
    scale = np.sqrt(variance)
    state = {'mean': mean, 'scale': scale, 'active_mask': active, 'p_active': int(active.sum()), 'n_markers': raw.shape[1], 'core_rows': raw.shape[0], 'source_core_codes_sha256': _array_sha(raw), 'preprocessing_role': 'frozen_NAM_source_core_only'}
    z = transform_codes(raw, state)
    gram = linear_kernel(z)
    norm = np.einsum('ij,ij->i', z, z) / z.shape[1]
    distance = np.maximum(norm[:, None] + norm[None, :] - 2 * gram, 0)
    base_h2 = float(np.median(distance[np.triu_indices(len(z), 1)]))
    _require(np.isfinite(base_h2) and base_h2 > 0, 'Degenerate source-core bandwidth')
    state['base_h2'] = base_h2
    for value in (mean, scale, active):
        value.setflags(write=False)
    return state

def transform_codes(raw_codes, state):
    raw = _codes(raw_codes)
    _require(state.get('preprocessing_role') == 'frozen_NAM_source_core_only', 'Unknown preprocessing role')
    mean, scale, active = (state['mean'], state['scale'], state['active_mask'])
    _require(raw.shape[1] == state['n_markers'] and int(np.sum(active)) == state['p_active'], 'Marker order/dimension differs from frozen source state')
    dosage = np.array([0.0, 2.0, 1.0, np.nan])[raw]
    z = (np.where(np.isnan(dosage), mean, dosage)[:, active] - mean[active]) / scale[active]
    _require(np.isfinite(z).all(), 'Nonfinite transformed genotypes')
    return z

def normalized_rbf_kernel(za, zb=None, *, base_h2, multiplier):
    _require(np.isfinite(base_h2) and base_h2 > 0 and (multiplier in H2_MULTIPLIERS), 'Invalid frozen source bandwidth')
    za = np.asarray(za)
    return rbf_kernel(za, zb, h=np.sqrt(za.shape[1] * base_h2 * multiplier))

def arm_grid(arm):
    _require(arm in ARMS, 'Unknown model arm')
    if arm in ('rkhs0', 'pool_rkhs100'):
        return [{'lambda': lam, 'h2_multiplier': h} for h in H2_MULTIPLIERS for lam in DEFAULT_LAMBDAS]
    if arm in ('weighted25', 'weighted100'):
        return [{'lambda': lam, 'eta': eta} for eta in ETAS for lam in DEFAULT_LAMBDAS]
    return [{'lambda': lam} for lam in DEFAULT_LAMBDAS]

def _config_key(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))

def _choose(candidates):
    _require(candidates and all((np.isfinite(c['mean_utility']) and len(c['fold_utilities']) == 2 and np.isfinite(c['fold_utilities']).all() for c in candidates)), 'Invalid source candidate path')
    top = max((c['mean_utility'] for c in candidates))
    tolerance = 1e-12 * max(1.0, abs(top))

    def priority(c):
        cfg = c['config']
        return (-cfg['lambda'], cfg.get('eta', 0.0), {1.0: 0, 0.5: 1, 2.0: 2}.get(cfg.get('h2_multiplier', 1.0), 3), _config_key(cfg))
    return min((c for c in candidates if top - c['mean_utility'] <= tolerance), key=priority)['config']

def _utility(pred, y, ids):
    pred, y = (np.asarray(pred), np.asarray(y))
    _require(pred.shape == y.shape == (len(ids),) and len(ids) > 0 and (len(set(ids)) == len(ids)) and np.isfinite(pred).all() and np.isfinite(y).all(), 'Invalid source validation endpoint arrays')
    ties = [hashlib.sha256(f'selection-v1|{TIE_SEED}|{x}'.encode()).hexdigest() for x in ids]
    selected = sorted(range(len(ids)), key=lambda i: (-float(pred[i]), ties[i], ids[i]))[:int(np.ceil(0.1 * len(ids)))]
    return float(np.mean(y[selected]) - np.mean(y))

def validate_source_plan(source_plan, source_families, n_rows):
    families = list(source_families)
    _require(len(families) == n_rows, 'Source family registry dimension mismatch')
    if 'source_banks' in source_plan:
        banks = {k: list(source_plan['source_banks'].get(str(k), source_plan['source_banks'].get(k, []))) for k in (0, 25, 100)}
    else:
        common, extras = (list(source_plan['common_source_indices']), list(source_plan['extra100_indices']))
        banks = {0: common + extras, 25: common + extras[:75], 100: common}
    folds = source_plan['pseudo_folds']
    _require(len(folds) == 2 and len({f['family'] for f in folds}) == 2, 'Exactly two different source pseudo families required')
    for k, count in ((0, 2000), (25, 1975), (100, 1900)):
        bank = banks[k]
        _require(len(bank) == count and len(set(bank)) == count and all((type(i) is int and 0 <= i < n_rows for i in bank)), 'Frozen source bank size/index mismatch')
    _require(set(banks[100]).issubset(banks[25]) and set(banks[25]).issubset(banks[0]), 'Non-nested information banks')
    core = list(source_plan['core_indices'])
    pseudo = {f['family'] for f in folds}
    _require(len(core) == len(set(core)) == 1200 and set(core).issubset(banks[100]) and (not any((families[i] in pseudo for i in core))), 'Source core isolation failure')
    counts = {fam: sum((families[i] == fam for i in core)) for fam in set((families[i] for i in core))}
    _require(len(counts) == 13 and sorted(counts.values()) == [92] * 9 + [93] * 4, 'Frozen 13-family core92/93 allocation mismatch')
    for fold in folds:
        adapt, val = (list(fold['adapt100_indices']), list(fold['validation250_indices']))
        _require(len(adapt) == len(set(adapt)) == 100 and len(val) == len(set(val)) == 250 and (not set(adapt) & set(val)), 'Pseudo100/250 isolation failure')
        _require(set(adapt + val).issubset(banks[100]) and all((families[i] == fold['family'] for i in adapt + val)), 'Pseudo reservoir family/index failure')
        for k, count in ((0, 1650), (25, 1625), (100, 1550)):
            donors = [i for i in banks[k] if families[i] != fold['family']]
            _require(len(donors) == count and (not set(donors) & set(adapt + val)), 'Whole pseudo-family source holdout failure')
    return banks

def tune_source(z_source, y_source, source_ids, source_families, source_plan, base_h2, *, emit=None):
    z_source = _finite_array(z_source, 2, 'z_source')
    n = len(z_source)
    ids = list(source_ids)
    _require(len(ids) == n and len(set(ids)) == n and all((isinstance(x, str) and x for x in ids)), 'Invalid source ID registry')
    banks = validate_source_plan(source_plan, source_families, n)
    pool = set(banks[0])
    if isinstance(y_source, dict):
        _require(set(y_source) == pool, 'Source label mapping must expose exactly the assigned2000 pool')
        labels = np.full(n, np.nan)
        for i, value in y_source.items():
            labels[i] = float(value)
    else:
        labels = np.asarray(y_source, dtype=float)
        _require(labels.shape == (n,) and np.all(np.isnan(labels[[i for i in range(n) if i not in pool]])), 'Source vector exposes labels outside assigned budget')
    _require(np.isfinite(labels[banks[0]]).all(), 'Nonfinite assigned source labels')
    candidates = {arm: {} for arm in ARMS}
    fit_count = 0
    factors = 0
    used = {arm: set() for arm in ARMS}
    for fold in source_plan['pseudo_folds']:
        val = list(fold['validation250_indices'])
        adapt = list(fold['adapt100_indices'])
        cache = {}
        for arm in ARMS:
            k = 0 if arm in ('ridge0', 'rkhs0') else 25 if arm == 'weighted25' else 100
            donors = [] if arm == 'source_tuned_targetonly100' else [i for i in banks[k] if source_families[i] != fold['family']]
            train = donors + adapt[:k]
            _require(len(train) == (100 if arm == 'source_tuned_targetonly100' else 1650) and (not set(train) & set(val)), 'Inner training/validation leakage')
            used[arm].update(train + val)
            modifiers = H2_MULTIPLIERS if arm in ('rkhs0', 'pool_rkhs100') else ETAS if arm in ('weighted25', 'weighted100') else (None,)
            for modifier in modifiers:
                kernel_kind = 'rbf' if arm in ('rkhs0', 'pool_rkhs100') else 'linear'
                kernel_key = (tuple(train), kernel_kind, modifier if kernel_kind == 'rbf' else None)
                if kernel_key not in cache:
                    zt, zv = (z_source[train], z_source[val])
                    cache[kernel_key] = (normalized_rbf_kernel(zt, base_h2=base_h2, multiplier=modifier), normalized_rbf_kernel(zv, zt, base_h2=base_h2, multiplier=modifier)) if kernel_kind == 'rbf' else (linear_kernel(zt), linear_kernel(zv, zt))
                kt, kp = cache[kernel_key]
                weighted = arm in ('weighted25', 'weighted100')
                w = make_weights(len(donors), k, strategy='weighted' if weighted else 'pooled', eta=modifier if weighted else None)
                if emit:
                    emit({'event': 'fits_started', 'amount': 4, 'arm': arm, 'pseudo_family': fold['family']})
                fit_count += 4
                path = fit_predict_path(kt, kp, labels[train], w, DEFAULT_LAMBDAS, method=PATH_METHOD)
                factors += path['diagnostics']['factorization_count']
                for lam in DEFAULT_LAMBDAS:
                    config = {'lambda': lam}
                    if kernel_kind == 'rbf':
                        config['h2_multiplier'] = modifier
                    if weighted:
                        config['eta'] = modifier
                    key = _config_key(config)
                    candidate = candidates[arm].setdefault(key, {'config': config, 'fold_utilities': []})
                    candidate['fold_utilities'].append(_utility(path['predictions'][str(float(lam))], labels[val], [ids[i] for i in val]))
                if emit:
                    emit({'event': 'path_completed', 'arm': arm, 'pseudo_family': fold['family']})
        del cache
    _require(fit_count == 120, 'Fixed source fit count violated')
    selected = {}
    candidate_lists = {}
    actual_labels = {}
    for arm in ARMS:
        values = list(candidates[arm].values())
        _require(len(values) == len(arm_grid(arm)), 'Missing source grid candidates')
        for item in values:
            item['mean_utility'] = float(np.mean(item['fold_utilities']))
        selected[arm] = dict(_choose(values))
        candidate_lists[arm] = values
        final_source = set() if arm == 'source_tuned_targetonly100' else set(banks[0 if arm in ('ridge0', 'rkhs0') else 25 if arm == 'weighted25' else 100])
        actual_labels[arm] = {'source_tune_unique': len(used[arm]), 'source_tune_and_final_unique': len(used[arm] | final_source), 'allocated_site_label_upper_bound': 2000, 'target_final_labels': 0 if arm in ('ridge0', 'rkhs0') else 25 if arm == 'weighted25' else 100, 'actual_unique_model_and_tuning_labels': 800 if arm == 'source_tuned_targetonly100' else 2000}
        _require(len(used[arm] | final_source) == (700 if arm == 'source_tuned_targetonly100' else 2000 if arm in ('ridge0', 'rkhs0') else 1975 if arm == 'weighted25' else 1900), 'Source information ledger mismatch')
    return {'selected_configs': selected, 'candidates': candidate_lists, 'fit_count': fit_count, 'actual_labels_by_arm': actual_labels, 'diagnostics': {'factorization_count': factors, 'path_method': PATH_METHOD, 'target_labels_used_for_tuning': 0, 'source_only_tuning': True}}

def predict_arm(arm, z_train, y_train, z_score, config, base_h2, n_source, n_target, *, emit=None, strict=True):
    _require(arm in ARMS and config in arm_grid(arm), 'Config not in frozen source grid')
    expected = (2000, 0) if arm in ('ridge0', 'rkhs0') else (1975, 25) if arm == 'weighted25' else (0, 100) if arm == 'source_tuned_targetonly100' else (1900, 100)
    if strict:
        _require((n_source, n_target) == expected, 'Final arm label budget mismatch')
    z_train, z_score = (_finite_array(z_train, 2, 'z_train'), _finite_array(z_score, 2, 'z_score'))
    _require(len(z_train) == n_source + n_target and z_train.shape[1] == z_score.shape[1], 'Final arrays/label count mismatch')
    if arm in ('rkhs0', 'pool_rkhs100'):
        kt = normalized_rbf_kernel(z_train, base_h2=base_h2, multiplier=config['h2_multiplier'])
        kp = normalized_rbf_kernel(z_score, z_train, base_h2=base_h2, multiplier=config['h2_multiplier'])
    else:
        kt, kp = (linear_kernel(z_train), linear_kernel(z_score, z_train))
    weighted = arm in ('weighted25', 'weighted100')
    w = make_weights(n_source, n_target, strategy='weighted' if weighted else 'pooled', eta=config.get('eta'))
    if emit:
        emit({'event': 'fits_started', 'amount': 1, 'arm': arm})
    result = fit_predict(kt, kp, y_train, w, config['lambda'], return_info=True)
    return {'predictions': result['predictions'], 'diagnostics': result['diagnostics'], 'fit_count': 1, 'label_ledger': {'allocated_site_label_upper_bound': 2000, 'actual_unique_model_and_tuning_labels': 800 if arm == 'source_tuned_targetonly100' else 2000, 'final_fit_source_labels': n_source, 'final_fit_target_labels': n_target, 'score_truth_used': False}}

def prepare_source_cache(kss, ys, source_weights, lam):
    kss = _finite_array(kss, 2, 'kss')
    ys = _finite_array(ys, 1, 'ys')
    w = _finite_array(source_weights, 1, 'source_weights')
    _require(kss.shape == (len(ys), len(ys)) and w.shape == ys.shape and (0 < len(ys) <= MAX_TRAIN), 'Invalid source cache dimensions')
    _require(np.all(w >= 0) and 0 < float(w.sum()) <= 1 + 1e-12 and np.isfinite(lam) and (lam > 0), 'Invalid exact source cache weights/lambda')
    _require(np.linalg.norm(kss - kss.T) <= SYMMETRY_LIMIT * max(np.linalg.norm(kss), np.finfo(float).tiny), 'Asymmetric source kernel')
    kss = np.array(0.5 * (kss + kss.T), copy=True)
    d = np.sqrt(w)
    a = d[:, None] * kss * d[None, :]
    a[np.diag_indices_from(a)] += lam
    lower = np.linalg.cholesky(a)
    h = d * ys
    uv = cholesky_solve(lower, np.column_stack((h, d)))
    cache = {'kss': kss, 'ys': ys.copy(), 'weights': w.copy(), 'lambda': float(lam), 'd': d, 'system': a, 'lower': lower, 'u': uv[:, 0], 'v': uv[:, 1]}
    for value in cache.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    cache['identity_sha256'] = hashlib.sha256((_array_sha(kss) + _array_sha(ys) + _array_sha(w) + str(float(lam))).encode()).hexdigest()
    return cache

def block_update_predict(cache, kst, ktt, kpreds, kpredt, yt, target_weights, *, expected_source_weights=None, expected_lambda=None):
    w = cache['weights']
    lam = cache['lambda']
    ds = cache['d']
    u, v = (cache['u'], cache['v'])
    if expected_source_weights is not None:
        _require(np.array_equal(w, np.asarray(expected_source_weights)), 'Changed source weights invalidate block cache')
    if expected_lambda is not None:
        _require(float(expected_lambda) == lam, 'Changed lambda invalidates block cache')
    ident = hashlib.sha256((_array_sha(cache['kss']) + _array_sha(cache['ys']) + _array_sha(w) + str(lam)).encode()).hexdigest()
    _require(ident == cache['identity_sha256'], 'Source cache identity mutated')
    kst, ktt = (_finite_array(kst, 2, 'kst'), _finite_array(ktt, 2, 'ktt'))
    ps, pt = (_finite_array(kpreds, 2, 'kpreds'), _finite_array(kpredt, 2, 'kpredt'))
    yt, wt = (_finite_array(yt, 1, 'yt'), _finite_array(target_weights, 1, 'target_weights'))
    k = len(yt)
    _require(kst.shape == (len(ds), k) and ktt.shape == (k, k) and (wt.shape == (k,)) and (ps.shape[1] == len(ds)) and (pt.shape == (len(ps), k)) and (len(ds) + k <= 2000), 'Incompatible block dimensions')
    _require(np.all(wt >= 0) and np.isclose(float(w.sum() + wt.sum()), 1.0, rtol=1e-10, atol=1e-12), 'Full block weights must sum1 without rescaling source')
    if k:
        _require(np.linalg.norm(ktt - ktt.T) <= SYMMETRY_LIMIT * max(np.linalg.norm(ktt), np.finfo(float).tiny), 'Asymmetric target kernel')
    dt = np.sqrt(wt)
    b = ds[:, None] * kst * dt[None, :]
    t = dt[:, None] * ktt * dt[None, :]
    if k:
        t[np.diag_indices_from(t)] += lam
    hs, ht = (ds * cache['ys'], dt * yt)
    e = float(ds @ v)
    q = float(ds @ u)
    if k:
        vb = cholesky_solve(cache['lower'], b)
        c = 0.5 * (t - b.T @ vb + (t - b.T @ vb).T)
        lower = np.linalg.cholesky(c)
        g = dt - b.T @ v
        r = ht - b.T @ u
        solved = cholesky_solve(lower, np.column_stack((r, g)))
        ur, vg = (solved[:, 0], solved[:, 1])
        denominator = e + float(g @ vg)
        intercept = (q + float(g @ ur)) / denominator
        ct = ur - vg * intercept
        cs = u - vb @ ct - v * intercept
        cancel = np.linalg.norm(ds) * (np.linalg.norm(u) + np.linalg.norm(vb @ ct) + abs(intercept) * np.linalg.norm(v)) + np.linalg.norm(dt) * (np.linalg.norm(ur) + abs(intercept) * np.linalg.norm(vg))
    else:
        denominator = e
        intercept = q / e
        cs = u - v * intercept
        ct = np.zeros(0)
        cancel = np.linalg.norm(ds) * (np.linalg.norm(u) + abs(intercept) * np.linalg.norm(v))
    _require(np.isfinite(denominator) and denominator > 0, 'Invalid block intercept denominator')
    residual = np.r_[cache['system'] @ cs + b @ ct + ds * intercept - hs, b.T @ cs + t @ ct + dt * intercept - ht]
    relative = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(np.r_[hs, ht])), np.finfo(float).tiny)
    constraint = abs(float(ds @ cs + dt @ ct)) / max(float(cancel), np.finfo(float).tiny)
    _require(np.isfinite(relative) and np.isfinite(constraint) and (max(relative, constraint) <= RESIDUAL_LIMIT), 'Block normal equations failed')
    pred = intercept + ps @ (ds * cs) + pt @ (dt * ct)
    _require(np.isfinite(pred).all(), 'Nonfinite block predictions')
    return {'predictions': pred, 'intercept': float(intercept), 'diagnostics': {'equation_relative_residual': relative, 'intercept_constraint_relative_residual': constraint, 'source_cache_identity_sha256': ident, 'target_factorization_count': int(k > 0), 'source_refactorization_count': 0}}
