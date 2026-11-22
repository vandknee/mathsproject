"""
regression.py  (Module 3: Correlation and Regression, Least Square Method)
Simple linear regression only. All methods implemented with NumPy.
"""
import numpy as np


def pearson_r(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx @ dy) / np.sqrt((dx @ dx) * (dy @ dy)))


def correlation_matrix(cols):
    return np.array([[pearson_r(a, b) for b in cols] for a in cols])


def regression_lines(x, y):
    """Least-squares lines of regression.
       y on x:  y - y_bar = b_yx (x - x_bar),  b_yx = Sxy / Sxx
       x on y:  x - x_bar = b_xy (y - y_bar),  b_xy = Sxy / Syy
    """
    dx, dy = x - x.mean(), y - y.mean()
    sxy, sxx, syy = dx @ dy, dx @ dx, dy @ dy
    b_yx, b_xy = sxy / sxx, sxy / syy
    a_yx = y.mean() - b_yx * x.mean()          # y = a_yx + b_yx x
    return {"a": float(a_yx), "b": float(b_yx), "b_yx": float(b_yx), "b_xy": float(b_xy),
            "x_bar": float(x.mean()), "y_bar": float(y.mean()),
            "r": float(sxy / np.sqrt(sxx * syy)),
            "r_from_coeffs": float(np.sign(sxy) * np.sqrt(b_yx * b_xy))}


def predict(line, x):
    return line["a"] + line["b"] * np.asarray(x, float)


def metrics(y, y_hat):
    e = y - y_hat
    sse, sst = float(e @ e), float(((y - y.mean()) ** 2).sum())
    return {"r2": 1 - sse / sst, "rmse": float(np.sqrt(sse / len(y))), "mae": float(np.abs(e).mean())}
