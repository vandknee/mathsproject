"""
lp_graphical.py  (Module 4: Linear Programming - Graphical Method)

    Minimise  Z = 600 x_B + 1800 x_V
    s.t.      160 x_B + 600 x_V >= W_plan     (load, kg)
              s_B x_B + s_V x_V >= S_plan     (stops; s_B, s_V from regression)
              x_B + x_V <= 12                 (drivers)
              x_B, x_V >= 0
"""
import itertools
import numpy as np
import config as C


def build(stops, W_plan, S_plan):
    c = np.array([C.VEHICLES[v]["daily_cost"] for v in ["Bike", "Van"]], float)
    A = np.array([[C.VEHICLES["Bike"]["capacity_kg"], C.VEHICLES["Van"]["capacity_kg"]],
                  [stops["Bike"], stops["Van"]], [1, 1]], float)
    b = np.array([W_plan, S_plan, C.MAX_DRIVERS], float)
    return c, A, b, [">=", ">=", "<="]


def feasible(p, A, b, senses, tol=1e-7):
    if np.any(p < -tol):
        return False
    for i, s in enumerate(senses):
        lhs = A[i] @ p
        if (s == ">=" and lhs < b[i] - tol) or (s == "<=" and lhs > b[i] + tol):
            return False
    return True


def graphical_method(c, A, b, senses):
    """Intersect every pair of boundary lines (including the axes), keep the
    feasible intersections (corner points) and evaluate Z at each one."""
    lines = [(A[i], b[i]) for i in range(len(b))] + [(np.array([1.0, 0]), 0.0),
                                                      (np.array([0, 1.0]), 0.0)]
    corners = []
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        M = np.vstack([a1, a2])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, [b1, b2])
        p = np.where(np.abs(p) < 1e-9, 0.0, p)
        if feasible(p, A, b, senses) and not any(np.allclose(p, q) for q in corners):
            corners.append(p)
    ctr = np.mean(corners, axis=0)
    corners.sort(key=lambda p: np.arctan2(p[1] - ctr[1], p[0] - ctr[0]))
    values = [float(c @ p) for p in corners]
    k = int(np.argmin(values))
    binding = [i for i in range(len(b)) if abs(A[i] @ corners[k] - b[i]) < 1e-6]
    return {"corners": corners, "values": values, "optimum": corners[k], "Z": values[k],
            "binding": binding}


def best_whole_fleet(c, A, b, senses):
    """Vehicles must be whole: check the integer (lattice) points of the feasible region."""
    best, cost = None, np.inf
    for xb in range(C.MAX_DRIVERS + 1):
        for xv in range(C.MAX_DRIVERS + 1):
            p = np.array([xb, xv], float)
            if feasible(p, A, b, senses) and c @ p < cost:
                best, cost = p, float(c @ p)
    return best, cost
