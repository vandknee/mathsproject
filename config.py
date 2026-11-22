"""config.py - settings for the Delivery Time Prediction project."""
SEED = 42

# Delivery records (for regression)
N_RECORDS = 1000

# Order history (for the Central Limit Theorem)
HIST_DAYS = 90
MEAN_DAILY_ORDERS = 240
ORDER_WEIGHT_MU = 1.692       # log-normal parameters -> mean about 6.5 kg
ORDER_WEIGHT_SIGMA = 0.6
SERVICE_LEVEL_Z = 1.645       # 95% one-sided service level

# Vehicles (for the LP graphical method)
VEHICLES = {
    "Bike": {"daily_cost": 600, "capacity_kg": 160},
    "Van":  {"daily_cost": 1800, "capacity_kg": 600},
}
SHIFT_MINUTES = 480           # one 8-hour shift
MAX_DRIVERS = 12
