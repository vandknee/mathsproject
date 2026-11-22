"""plots.py - figures used in the report (saved in outputs/)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#1f4e79", "#e07b39", "#3a7d44", "#b23a48", "#7f7f7f"


def fig_clt(hist, daily, clt, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(hist.weight_kg.clip(upper=30), bins=50, color=GREY, edgecolor="white")
    ax[0].set(title=f"(a) Individual order weights (skewness = {clt['order_skew']:.2f})",
              xlabel="Order weight (kg, values above 30 grouped at 30)", ylabel="Number of orders")
    x = daily.total_kg
    ax[1].hist(x, bins=15, density=True, color=BLUE, alpha=0.75, edgecolor="white")
    g = np.linspace(x.min() - 50, x.max() + 50, 300)
    ax[1].plot(g, stats.norm.pdf(g, clt["kg_mean"], clt["kg_std"]), color=ORANGE, lw=2, label="Normal curve")
    ax[1].axvline(clt["kg_plan"], color=RED, ls="--", lw=1.5, label=f"Plan (95%) = {clt['kg_plan']:.0f} kg")
    ax[1].set(title=f"(b) Daily total weight (skewness = {clt['daily_skew']:.2f})",
              xlabel="Total weight per day (kg)", ylabel="Density")
    ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(path); plt.close(fig)


def fig_correlation(M, labels, train, lines, path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.45})
    im = ax[0].imshow(M, cmap="Blues", vmin=-0.2, vmax=1)
    ax[0].set_xticks(range(len(labels)), labels)
    ax[0].set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                       color="white" if M[i, j] > 0.6 else "black")
    for s in ax[0].spines.values():
        s.set_visible(False)
    ax[0].set_title("(a) Pearson correlation matrix")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for v, col in [("Bike", BLUE), ("Van", ORANGE)]:
        d = train[train.vehicle == v]
        ax[1].scatter(d.distance_km, d.time_min, s=7, alpha=0.35, color=col)
        g = np.linspace(0, train.distance_km.max(), 100)
        L = lines[v]
        ax[1].plot(g, L["a"] + L["b"] * g, color=col, lw=2.2,
                   label=f"{v}: t = {L['a']:.2f} + {L['b']:.2f} d")
    ax[1].set(xlabel="Distance d (km)", ylabel="Delivery time t (min)",
              title="(b) Regression lines of time on distance")
    ax[1].legend(frameon=False)
    fig.savefig(path); plt.close(fig)


def fig_evaluation(y, y_hat, path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.3})
    lim = [0, max(y.max(), y_hat.max()) + 2]
    ax[0].scatter(y, y_hat, s=10, alpha=0.6, color=BLUE)
    ax[0].plot(lim, lim, color=ORANGE, lw=1.5, label="Perfect prediction")
    ax[0].set(xlim=lim, ylim=lim, xlabel="Actual time (min)", ylabel="Predicted time (min)",
              title="(a) Test set: actual vs predicted")
    ax[0].legend(frameon=False)
    ax[1].scatter(y_hat, y - y_hat, s=10, alpha=0.6, color=BLUE)
    ax[1].axhline(0, color=ORANGE, lw=1.5)
    ax[1].set(xlabel="Predicted time (min)", ylabel="Residual (min)", title="(b) Residuals vs predicted")
    fig.savefig(path); plt.close(fig)


def fig_graphical(c, A, b, g, whole, path):
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    xmax, ymax = 13, 13
    xs = np.linspace(0, xmax, 400)
    names = [f"Load: {A[0,0]:.0f}x$_B$ + {A[0,1]:.0f}x$_V$ ≥ {b[0]:.0f}",
             f"Stops: {A[1,0]:.0f}x$_B$ + {A[1,1]:.0f}x$_V$ ≥ {b[1]:.0f}",
             f"Drivers: x$_B$ + x$_V$ ≤ {b[2]:.0f}"]
    for i, col in enumerate([BLUE, GREEN, RED]):
        ax.plot(xs, (b[i] - A[i, 0] * xs) / A[i, 1], color=col, lw=1.6, label=names[i])
    P = np.array(g["corners"])
    ax.fill(P[:, 0], P[:, 1], color=ORANGE, alpha=0.25, label="Feasible region")
    on_axis = sorted([p[0] for p in g["corners"] if abs(p[1]) < 1e-9])
    for p, val in zip(g["corners"], g["values"]):
        ax.plot(*p, "o", color="black", ms=4)
        txt = f"({p[0]:.2f}, {p[1]:.2f})\nZ = ₹{val:,.0f}"
        if abs(p[1]) < 1e-9:
            k = on_axis.index(p[0])
            ax.annotate(txt, p, textcoords="offset points", xytext=(-22, 34) if k == 0 else (-14, 70),
                        fontsize=7.5, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
        elif np.allclose(p, g["optimum"]):
            ax.annotate(txt, p, textcoords="offset points", xytext=(-92, 16), fontsize=7.5)
        else:
            ax.annotate(txt, p, textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.plot(xs, (g["Z"] - c[0] * xs) / c[1], color=GREY, ls="--", lw=1.2,
            label=f"Iso-cost line Z = ₹{g['Z']:,.0f}")
    ax.plot(*g["optimum"], "*", color=RED, ms=14, zorder=5, label="Optimal corner point")
    ax.plot(*whole, "D", color=GREEN, ms=7, zorder=5, label=f"Best whole fleet ({whole[0]:.0f}, {whole[1]:.0f})")
    ax.set(xlim=(0, xmax), ylim=(0, ymax), xlabel="x$_B$ (number of bikes)",
           ylabel="x$_V$ (number of vans)", title="Graphical method: bikes vs vans")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(path); plt.close(fig)
