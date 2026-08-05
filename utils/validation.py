"""
Validation metrics for evaluating fused product performance.
"""

import numpy as np


def rmse(pred, obs):
    """
    Root Mean Square Error.
    """
    valid = ~np.isnan(pred) & ~np.isnan(obs)
    if np.sum(valid) == 0:
        return np.nan
    return np.sqrt(np.nanmean((pred[valid] - obs[valid]) ** 2))


def mae(pred, obs):
    """
    Mean Absolute Error.
    """
    valid = ~np.isnan(pred) & ~np.isnan(obs)
    if np.sum(valid) == 0:
        return np.nan
    return np.nanmean(np.abs(pred[valid] - obs[valid]))


def bias(pred, obs):
    """
    Mean bias (pred - obs).
    """
    valid = ~np.isnan(pred) & ~np.isnan(obs)
    if np.sum(valid) == 0:
        return np.nan
    return np.nanmean(pred[valid] - obs[valid])


def compute_metrics(pred, obs):
    """
    Compute all standard metrics (RMSE, MAE, Bias) at once.

    Returns
    -------
    dict
        Dictionary with keys 'rmse', 'mae', 'bias'.
    """
    return {
        'rmse': rmse(pred, obs),
        'mae': mae(pred, obs),
        'bias': bias(pred, obs)
    }


def pod(hits, misses):
    """
    Probability of Detection: POD = H / (H + M)
    """
    denominator = hits + misses
    if denominator == 0:
        return np.nan
    return hits / denominator


def far(hits, false_alarms):
    """
    False Alarm Ratio: FAR = F / (H + F)
    """
    denominator = hits + false_alarms
    if denominator == 0:
        return np.nan
    return false_alarms / denominator


def csi(hits, misses, false_alarms):
    """
    Critical Success Index: CSI = H / (H + M + F)
    """
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan
    return hits / denominator
