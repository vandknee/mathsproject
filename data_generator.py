"""
data_generator.py
Creates two synthetic (but realistic) datasets for a delivery hub at Sector 62, Noida:
  data/delivery_records.csv : 1,000 past deliveries (for regression)
  data/order_history.csv    : every order of the last 90 days (for the CLT)
The rest of the code works unchanged on real data in the same format.
"""
import os
import numpy as np
import pandas as pd
import config as C

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def delivery_records(rng, n=C.N_RECORDS):
    vehicle = rng.choice(["Bike", "Van"], n, p=[0.6, 0.4])
    distance = np.round(rng.gamma(2.0, 0.9, n), 2)               # km
    traffic = rng.choice([1, 2, 3], n, p=[0.3, 0.45, 0.25])      # 1 low, 3 high
    weather = rng.choice(["Clear", "Rain"], n, p=[0.8, 0.2])
    van = (vehicle == "Van").astype(float)
    rain = (weather == "Rain").astype(float)
    # Underlying process (unknown to the model)
    time = 3.0 + 3.2 * distance + 2.5 * traffic + 3.0 * rain + 2.0 * van + rng.normal(0, 2.0, n)
    time = np.round(np.clip(time, 2, None), 1)
    return pd.DataFrame({"delivery_id": [f"D{i+1:04d}" for i in range(n)], "vehicle": vehicle,
                         "distance_km": distance, "traffic": traffic, "weather": weather,
                         "time_min": time})


def order_history(rng):
    rows = []
    for day in range(1, C.HIST_DAYS + 1):
        n = rng.poisson(C.MEAN_DAILY_ORDERS)
        w = rng.lognormal(C.ORDER_WEIGHT_MU, C.ORDER_WEIGHT_SIGMA, n)
        rows += [(day, round(x, 2)) for x in w]
    return pd.DataFrame(rows, columns=["day", "weight_kg"])


def generate(seed=C.SEED):
    rng = np.random.default_rng(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    rec, hist = delivery_records(rng), order_history(rng)
    rec.to_csv(os.path.join(DATA_DIR, "delivery_records.csv"), index=False)
    hist.to_csv(os.path.join(DATA_DIR, "order_history.csv"), index=False)
    return rec, hist


if __name__ == "__main__":
    r, h = generate()
    print(f"delivery_records.csv: {len(r)} rows;  order_history.csv: {len(h)} orders")
