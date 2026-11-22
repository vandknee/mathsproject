"""
main.py - Delivery Time Prediction using Regression (MFAIDS PBL)
Run:  py main.py   (Windows)   or   python main.py

  Step 1  Data (80% train / 20% test split)
  Step 2  Central Limit Theorem: daily demand plan          (Module 3)
  Step 3  Correlation analysis                              (Module 3)
  Step 4  Simple linear regression, least squares           (Module 3)
  Step 5  Separate regression line for bikes and vans       (Module 3)
  Step 6  Evaluation on the test set
  Step 7  Predicted delivery time -> deliveries per shift
  Step 8  Fleet planning: LP by the Graphical Method        (Module 4)
"""
import os
import json
import numpy as np
import pandas as pd

import config as C
import data_generator
import regression as R
import clt_demand
import lp_graphical as LP
import plots as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def line(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    line("STEP 1  Data")
    rec, hist = data_generator.generate()
    idx = np.random.default_rng(C.SEED).permutation(len(rec))
    cut = int(0.8 * len(rec))
    train, test = rec.iloc[idx[:cut]], rec.iloc[idx[cut:]]
    print(f"Delivery records: {len(rec)} ({len(train)} train, {len(test)} test)")
    print(f"Order history   : {len(hist)} orders over {hist.day.nunique()} days")
    res["data"] = {"n": len(rec), "n_train": len(train), "n_test": len(test),
                   "mean_time": float(rec.time_min.mean()), "sd_time": float(rec.time_min.std()),
                   "mean_dist": float(rec.distance_km.mean()), "max_dist": float(rec.distance_km.max()),
                   "vehicles": rec.vehicle.value_counts().to_dict(),
                   "rain_share": float((rec.weather == "Rain").mean())}

    line("STEP 2  Central Limit Theorem - daily demand plan")
    clt, daily = clt_demand.plan_demand(hist)
    for k, v in clt.items():
        print(f"  {k:18s} {v}")
    res["clt"] = clt
    P.fig_clt(hist, daily, clt, os.path.join(OUT, "fig1_clt.png"))

    line("STEP 3  Correlation - choosing the predictor")
    labels = ["Distance", "Traffic", "Rain", "Time"]
    cols = [train.distance_km.to_numpy(float), train.traffic.to_numpy(float),
            (train.weather == "Rain").to_numpy(float), train.time_min.to_numpy(float)]
    M = R.correlation_matrix(cols)
    print(pd.DataFrame(M, index=labels, columns=labels).round(3).to_string())
    res["corr_with_time"] = {labels[j]: float(M[-1, j]) for j in range(3)}

    line("STEP 4  Simple linear regression of time on distance (all deliveries)")
    x, y = train.distance_km.to_numpy(float), train.time_min.to_numpy(float)
    overall = R.regression_lines(x, y)
    m_all = R.metrics(test.time_min.to_numpy(float), R.predict(overall, test.distance_km))
    print(overall, m_all)
    res["overall"] = {**overall, "test": m_all}

    line("STEP 5  Separate regression line for each vehicle")
    lines, per_test, per_train, n_v = {}, {}, {}, {}
    for v in ["Bike", "Van"]:
        tr, te = train[train.vehicle == v], test[test.vehicle == v]
        L = R.regression_lines(tr.distance_km.to_numpy(float), tr.time_min.to_numpy(float))
        lines[v] = L
        per_train[v] = R.metrics(tr.time_min.to_numpy(float), R.predict(L, tr.distance_km))
        per_test[v] = R.metrics(te.time_min.to_numpy(float), R.predict(L, te.distance_km))
        n_v[v] = (len(tr), len(te))
        print(f"  {v}: t = {L['a']:.3f} + {L['b']:.3f} d   r = {L['r']:.3f}  "
              f"b_yx = {L['b_yx']:.4f}, b_xy = {L['b_xy']:.4f}, sqrt(b_yx*b_xy) = {L['r_from_coeffs']:.3f}")
        print(f"        train {per_train[v]}\n        test  {per_test[v]}  (n = {n_v[v]})")
    # combined test performance of the two vehicle lines
    yhat = np.where(test.vehicle == "Bike", R.predict(lines["Bike"], test.distance_km),
                    R.predict(lines["Van"], test.distance_km))
    yte = test.time_min.to_numpy(float)
    m_comb = R.metrics(yte, yhat)
    base = R.metrics(yte, np.full_like(yte, y.mean()))
    within = float(np.mean(np.abs(yte - yhat) <= 3))
    print(f"Combined test {m_comb}; baseline {base}; within 3 min {within:.3f}")
    res["lines"] = lines
    res["eval"] = {"train": per_train, "test": per_test, "n": n_v, "combined": m_comb,
                   "baseline": base, "within3": within}
    P.fig_correlation(M, labels, train, lines, os.path.join(OUT, "fig2_regression.png"))
    P.fig_evaluation(yte, yhat, os.path.join(OUT, "fig3_evaluation.png"))

    line("STEP 6/7  Predicted delivery time and deliveries per shift")
    d_bar = float(rec.distance_km.mean())
    stops, minutes = {}, {}
    for v in ["Bike", "Van"]:
        minutes[v] = float(R.predict(lines[v], d_bar))
        stops[v] = int(C.SHIFT_MINUTES // minutes[v])
        print(f"  {v:5s}: {minutes[v]:.2f} min per delivery -> {stops[v]} deliveries per shift")
    eta = []
    for v in ["Bike", "Van"]:
        for d in [1.0, 2.0, 4.0]:
            eta.append({"vehicle": v, "distance_km": d, "predicted_min": round(float(R.predict(lines[v], d)), 1)})
    print(pd.DataFrame(eta).to_string(index=False))
    res["avg_distance"], res["minutes"], res["stops"], res["eta"] = d_bar, minutes, stops, eta

    line("STEP 8  Fleet planning - LP by the Graphical Method")
    c, A, bb, s = LP.build(stops, clt["kg_plan"], clt["stops_plan"])
    g = LP.graphical_method(c, A, bb, s)
    for p, val in zip(g["corners"], g["values"]):
        print(f"  corner ({p[0]:6.3f}, {p[1]:6.3f})  Z = Rs {val:9.2f}")
    print(f"Optimum x_B = {g['optimum'][0]:.3f}, x_V = {g['optimum'][1]:.3f}, Z = Rs {g['Z']:.2f}; "
          f"binding constraints: {g['binding']}")
    whole, wcost = LP.best_whole_fleet(c, A, bb, s)
    print(f"Best whole fleet: {whole.astype(int)} -> Rs {wcost:.0f}")
    res["lp"] = {"A": A.tolist(), "b": bb.tolist(), "corners": [list(map(float, p)) for p in g["corners"]],
                 "values": g["values"], "optimum": list(map(float, g["optimum"])), "Z": g["Z"],
                 "binding": g["binding"], "whole": whole.tolist(), "whole_cost": wcost}
    P.fig_graphical(c, A, bb, g, whole, os.path.join(OUT, "fig4_graphical_lp.png"))

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    line("DONE - figures and results saved in outputs/")


if __name__ == "__main__":
    main()
