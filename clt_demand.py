"""
clt_demand.py  (Module 3: Central Limit Theorem)
Daily total load is the SUM of many independent, skewed order weights, so by the
CLT it is approximately normal. Plan capacity for a 95% service level: mean + z*sd.
"""
import numpy as np
from scipy import stats
import config as C


def plan_demand(hist):
    daily = hist.groupby("day").agg(orders=("weight_kg", "size"), total_kg=("weight_kg", "sum"))
    w = hist.weight_kg.to_numpy()
    out = {"n_days": len(daily), "n_orders": len(w), "order_weight_mean": float(w.mean()),
           "order_skew": float(stats.skew(w)), "daily_skew": float(stats.skew(daily.total_kg)),
           "shapiro_order_p": float(stats.shapiro(w[:5000]).pvalue),
           "shapiro_daily_p": float(stats.shapiro(daily.total_kg).pvalue)}
    for col, key in [("total_kg", "kg"), ("orders", "stops")]:
        x = daily[col].to_numpy(float)
        mu, s, n = x.mean(), x.std(ddof=1), len(x)
        half = 1.96 * s / np.sqrt(n)
        out[f"{key}_mean"], out[f"{key}_std"] = float(mu), float(s)
        out[f"{key}_ci"] = [float(mu - half), float(mu + half)]
        out[f"{key}_plan"] = float(np.ceil(mu + C.SERVICE_LEVEL_Z * s))
    return out, daily
