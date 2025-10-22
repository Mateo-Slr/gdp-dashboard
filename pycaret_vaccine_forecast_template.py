
# -*- coding: utf-8 -*-
"""
PyCaret Time Series - Vaccination Coverage Forecast (Template)
Author: ChatGPT
Usage:
  1) pip install "pycaret[time_series]" --upgrade
  2) python pycaret_vaccine_forecast_template.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Paths ----------
DATA_PATH = Path("couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv") 
OUTPUT_DIR = Path("pycaret_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH, parse_dates=["ds"])

# Choose one series to start (example: "Grippe 65 ans et plus")
SERIE_NAME = "Grippe 65 ans et plus"
one = df[df["serie"] == SERIE_NAME].sort_values("ds").reset_index(drop=True)

# Keep only date and target
one = one[["ds", "couverture_pct"]].rename(columns={"ds": "index", "couverture_pct": "y"})
one = one.set_index("index")

# ---------- PyCaret setup ----------
from pycaret.time_series import setup, compare_models, plot_model, predict_model, finalize_model, tune_model, save_model, load_model

exp = setup(
    data=one,
    target="y",
    fold=3,                 # time-series cross-validation (rolling origin)
    fh=1,                   # forecast horizon: 1 period ahead (next year)
    seasonal_period="auto", # let PyCaret infer seasonality
    n_jobs=-1,
    session_id=42,
    ignore_features=None,
    numeric_imputation_target="linear",    # interpolate missing y
    verbose=True
)

# ---------- Model selection ----------
best = compare_models(sort="MASE")   # or "sMAPE" / "MAE"
print("Best model:", best)

# Optional: hyperparameter tuning
best_tuned = tune_model(best, optimize="MASE")

# ---------- Finalize and forecast next year ----------
final_best = finalize_model(best_tuned)
future = predict_model(final_best, fh=1)   # one-step ahead
print("Forecast next year:", future)

# Save artifacts
save_model(final_best, OUTPUT_DIR / f"model_{SERIE_NAME.replace(' ', '_')}")
future.to_csv(OUTPUT_DIR / f"forecast_{SERIE_NAME.replace(' ', '_')}.csv", index=False)

# ---------- OPTIONAL: scale to all series ----------
# This loop will train a quick model per series and forecast 1 step ahead. Adjust as needed.
all_forecasts = []

for serie, g in df.groupby("serie"):
    ts = g.sort_values("ds")[["ds", "couverture_pct"]].rename(columns={"ds":"index", "couverture_pct":"y"}).set_index("index")
    exp_i = setup(data=ts, target="y", fold=3, fh=1, seasonal_period="auto", n_jobs=-1, session_id=42, verbose=False, numeric_imputation_target="linear")
    best_i = compare_models(sort="MASE", turbo=True)  # turbo=True for speed
    final_i = finalize_model(best_i)
    fc_i = predict_model(final_i, fh=1)
    fc_i["serie"] = serie
    all_forecasts.append(fc_i.assign(ds=ts.index.max() + pd.offsets.YearEnd(1)))

all_fc = pd.concat(all_forecasts, ignore_index=True)
all_fc.to_csv(OUTPUT_DIR / "all_series_forecasts.csv", index=False)

print("Done. Forecasts saved to:", OUTPUT_DIR)
