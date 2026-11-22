# Delivery Time Prediction using Regression
MFAIDS (25B11MA213) Project Based Learning

Predicts delivery time from distance (separate regression lines for bikes and vans), then
uses the predictions to plan the cheapest daily fleet of bikes and vans.

| Step | Syllabus topic | File |
|---|---|---|
| Daily demand to plan for | Central Limit Theorem (Module 3) | `clt_demand.py` |
| What affects delivery time | Correlation (Module 3) | `regression.py` |
| Time vs distance (bikes and vans) | Lines of regression, least square method (Module 3) | `regression.py` |
| Cheapest bike/van fleet | LP - Graphical Method (Module 4) | `lp_graphical.py` |

## How to run (Windows)
```
py -m pip install -r requirements.txt
py main.py
```
(Use `python` instead of `py` on Mac/Linux.) Figures and results are saved in `outputs/`.

## Key results
- Distance r = 0.81, traffic r = 0.33 with delivery time
- Bike line: t = 8.742 + 3.135 d (r = 0.808);  Van line: t = 10.285 + 3.228 d (r = 0.842)
- Bike and van lines on test data: R^2 = 0.705, RMSE = 2.93 min, MAE = 2.37 min
- Deliveries per shift: bike 33, van 29
- CLT plan (95%): 1,802 kg and 267 deliveries per day
- Graphical method: optimum 7.12 bikes + 1.11 vans = Rs 6,260/day
- Best whole fleet: 8 bikes + 1 van = Rs 6,600/day
