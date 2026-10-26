"""Exact X-only windows and bounded chronological selections."""
import numpy as np


def evenly_spaced_starts(lower, upper, window, count):
    if not all(type(v) is int for v in (lower, upper, window, count)) or lower < 0 or window < 1 or count < 1:
        raise ValueError('Invalid selection arguments')
    if (upper - lower) // window < count:
        raise ValueError('Insufficient nonoverlapping windows')
    starts = np.linspace(lower, upper - window, count, dtype=np.int64)
    if count > 1 and np.min(np.diff(starts)) < window:
        raise ValueError('Overlapping selection')
    return starts.tolist()


def window_batch(ds, starts):
    blocks = [ds.read_rows(int(i), int(i) + ds.window_size) for i in starts]
    values = np.stack([b['values'] for b in blocks]); status = np.stack([b['status'] for b in blocks])
    features = np.concatenate([values[..., None], np.eye(5, dtype=np.float32)[status]], -1)
    return dict(X=features, values=values, status=status,
        source_start=np.array([b['source_rows'][0] for b in blocks]),
        source_stop=np.array([b['source_rows'][-1] + 1 for b in blocks]),
        timestamp_end=np.array([b['timestamps'][-1] for b in blocks]))


def fit_threshold(scores, quantile=.99):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all() or not 0 < quantile < 1:
        raise ValueError('Invalid calibration scores/quantile')
    return float(np.quantile(scores, quantile, method='higher'))
