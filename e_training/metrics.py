import numpy as np

def _as1d(x):
    x = np.asarray(x)
    if x.ndim > 1: x = x.reshape(-1)
    return x.astype(np.float64, copy=False)

def rmse(y, yhat):
    y = _as1d(y); yhat = _as1d(yhat)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))

def mae(y, yhat):
    y = _as1d(y); yhat = _as1d(yhat)
    return float(np.mean(np.abs(yhat - y)))

def smape(y, yhat):
    y = _as1d(y); yhat = _as1d(yhat)
    denom = (np.abs(y) + np.abs(yhat)) + 1e-12
    return float(200.0 * np.mean(np.abs(yhat - y) / denom))

def r2(y, yhat):
    y = _as1d(y); yhat = _as1d(yhat)
    ss_res = np.sum((yhat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)

METRICS = {"rmse": rmse, "mae": mae, "smape": smape, "r2": r2}