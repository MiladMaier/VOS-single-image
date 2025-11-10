"""
Lightweight calibration utilities shim

This module provides a minimal get_calibration_error(preds, gts)
implementation so the repository can run when the external
`calibration` package isn't installed. It computes a standard
expected calibration error (ECE) by binning confidences and
measuring |accuracy - confidence| per bin.

Note: This is a compatibility shim. For research-grade calibration
metrics you may want to install or use the original `calibration`
library the project expects.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


def _as_numpy(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def get_calibration_error(predicted_scores, gt_labels, n_bins: int = 15) -> float:
    """Compute an Expected Calibration Error (ECE)-style score.

    Args:
        predicted_scores: numpy array of shape (N,) confidences or (N, C)
            class probability vectors.
        gt_labels: numpy array of shape (N,) with integer class labels or
            one-hot arrays of shape (N, C). If gt_labels are 0/1 indicators
            they are treated as correctness values directly.
        n_bins: number of histogram bins to use.

    Returns:
        scalar float ECE in [0,1]
    """
    preds = _as_numpy(predicted_scores)
    gts = _as_numpy(gt_labels)

    if preds.ndim == 1:
        confidences = preds.astype(float)
        # If gts are binary indicators (0/1), treat them as correctness
        unique = np.unique(gts)
        if set(unique.tolist()).issubset({0, 1}):
            correctness = gts.astype(float)
        else:
            # Not much we can do: assume gt labels where 1==positive
            correctness = (gts == 1).astype(float)
    else:
        # preds are class probabilities
        confidences = preds.max(axis=1).astype(float)
        pred_labels = preds.argmax(axis=1)
        if gts.ndim > 1:
            true_labels = gts.argmax(axis=1)
        else:
            true_labels = gts
        correctness = (pred_labels == true_labels).astype(float)

    # Clip confidences to [0,1]
    confidences = np.clip(confidences, 0.0, 1.0)

    # Compute ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(confidences.shape[0]) if confidences.shape[0] > 0 else 1.0
    ece = 0.0
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        # include left edge except for the first bin
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        avg_conf = confidences[mask].mean()
        acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - acc)

    return float(ece)


__all__ = ["get_calibration_error"]
