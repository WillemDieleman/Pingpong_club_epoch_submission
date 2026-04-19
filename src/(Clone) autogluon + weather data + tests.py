# Databricks notebook source
# MAGIC %md
# MAGIC # Energy Consumption Prediction: Exploration & Local Testing
# MAGIC
# MAGIC Use this notebook for **data exploration** and **local model testing**.
# MAGIC Changes here do **not** affect your submission — when you are happy with your
# MAGIC model, copy the `EnergyConsumptionModel` class back to the **submission notebook**.
# MAGIC
# MAGIC ## Workflow
# MAGIC 1. Explore the data below.
# MAGIC 2. Iterate on your model in this notebook using a train/test split (train: before 2025-12-01, test: Dec 2025 – Feb 2026).
# MAGIC 3. When satisfied, copy your `EnergyConsumptionModel` class (and its imports) to the submission notebook.
# MAGIC 4. Run the Submit cell in the submission notebook.

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

!pip install "autogluon.timeseries==1.5.0" --quiet

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F, Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC
# MAGIC Row-level security ensures you only see data **up to end of November 2025** interactively.
# MAGIC
# MAGIC **Tip:** The dataset is large. Keep data as PySpark DataFrames. If you want to
# MAGIC explore in pandas, use `df.limit(N).toPandas()` on a small sample.

# COMMAND ----------

df = spark.table("datathon.shared.client_consumption")
# display(df)

# COMMAND ----------

# DBTITLE 1,Aggregate to a simple pandas DataFrame for AutoGluon prototyping
# ---------------------------------------------------------------
# Aggregate raw rows → one row per 15-min interval PER REGION
# item_id = community_code (18 regions)
# ---------------------------------------------------------------
agg_df = (
    df
    .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
    .groupBy("datetime_15min", "community_code")
    .agg(
        F.sum("active_kw").alias("active_kw"),
        F.countDistinct("client_id").alias("n_clients"),
    )
    .orderBy("datetime_15min", "community_code")
)

# Safe to .toPandas() — ~32K intervals × 18 regions ≈ ~576K rows
ts_pdf = agg_df.toPandas()
ts_pdf["datetime_15min"] = pd.to_datetime(ts_pdf["datetime_15min"])
ts_pdf = ts_pdf.sort_values(["community_code", "datetime_15min"]).reset_index(drop=True)
ts_pdf = ts_pdf.rename(columns={"community_code": "item_id"})
ts_pdf["avg_kw_per_client"] = ts_pdf["active_kw"] / ts_pdf["n_clients"]

print(f"Shape: {ts_pdf.shape}")
print(f"Regions: {sorted(ts_pdf['item_id'].unique())}")
print(f"Date range: {ts_pdf['datetime_15min'].min()} → {ts_pdf['datetime_15min'].max()}")
print(f"Memory: {ts_pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")
display(ts_pdf.head(10))

# COMMAND ----------

# DBTITLE 1,Load weather forecasts and merge with consumption data
# ---------------------------------------------------------------
# Load weather forecasts + regional holidays, merge into ts_pdf
# as known covariates for AutoGluon
# ---------------------------------------------------------------

# ========================  WEATHER  ========================
# Map Spanish autonomous community names → community_code
ADMIN_TO_CC = {
    "Andalusia": "AN",  "Aragon": "AR",       "Asturias": "AS",
    "Cantabria": "CB",  "Castille-Leon": "CL", "Castille-La Mancha": "CM",
    "Canary Islands": "CN", "Catalonia": "CT", "Extremadura": "EX",
    "Galicia": "GA",    "Murcia": "MC",        "Madrid": "MD",
    "Navarre": "NC",    "Basque Country": "PV", "La Rioja": "RI",
    "Valencia": "VC",
    # Balearic Islands excluded — not in consumption data
    # CE (Ceuta), ML (Melilla) — no weather stations available
}

# Day-1 forecast columns (available when predicting 1 day ahead)
WEATHER_RAW_COLS = [
    "temperature_2m_previous_day1",
    "apparent_temperature_previous_day1",
    "relative_humidity_2m_previous_day1",
    "cloud_cover_previous_day1",
    "wind_speed_10m_previous_day1",
    "rain_previous_day1",
    "surface_pressure_previous_day1",
]

WEATHER_RENAME = {
    "temperature_2m_previous_day1":       "temp_forecast",
    "apparent_temperature_previous_day1": "apparent_temp_forecast",
    "relative_humidity_2m_previous_day1": "humidity_forecast",
    "cloud_cover_previous_day1":          "cloud_cover_forecast",
    "wind_speed_10m_previous_day1":       "wind_speed_forecast",
    "rain_previous_day1":                 "rain_forecast",
    "surface_pressure_previous_day1":     "pressure_forecast",
}
WEATHER_COLS = list(WEATHER_RENAME.values())

# --- Load & map ---
weather_spark = spark.table("datathon.pingpong_club_epoch.weather_hourly_2")
weather_pdf = weather_spark.select("admin_name", "date", *WEATHER_RAW_COLS).toPandas()
weather_pdf["community_code"] = weather_pdf["admin_name"].map(ADMIN_TO_CC)
weather_pdf = weather_pdf.dropna(subset=["community_code"])
weather_pdf["date"] = pd.to_datetime(weather_pdf["date"])

# --- Average across cities within each region per hour ---
weather_agg = (
    weather_pdf
    .groupby(["community_code", "date"])[WEATHER_RAW_COLS]
    .mean()
    .reset_index()
    .rename(columns=WEATHER_RENAME)
)

# --- Resample hourly → 15-min (forward-fill within each hour) ---
weather_15min_parts = []
for cc, grp in weather_agg.groupby("community_code"):
    grp = grp.set_index("date").sort_index()
    grp_15 = grp[WEATHER_COLS].resample("15min").ffill()
    grp_15["item_id"] = cc
    weather_15min_parts.append(grp_15.reset_index().rename(columns={"date": "datetime_15min"}))

weather_15min = pd.concat(weather_15min_parts, ignore_index=True)

# --- Merge weather into consumption data ---
ts_pdf = ts_pdf.merge(weather_15min, on=["item_id", "datetime_15min"], how="left")

# ========================  HOLIDAYS  ========================
HOLIDAY_COLS = ["is_holiday", "is_bridge"]

holidays_pdf = spark.table("datathon.pingpong_club_epoch.holidays_regional").toPandas()
holidays_pdf["date"] = pd.to_datetime(holidays_pdf["date"])
# Convert booleans → int (0/1) for AutoGluon numeric covariates
holidays_pdf["is_holiday"] = holidays_pdf["is_holiday"].astype(int)
holidays_pdf["is_bridge"]  = holidays_pdf["is_bridge"].astype(int)
holidays_pdf = holidays_pdf[["date", "community_code", "is_holiday", "is_bridge"]]

# Add a date column to ts_pdf for the join
ts_pdf["date"] = ts_pdf["datetime_15min"].dt.normalize()
ts_pdf = ts_pdf.merge(
    holidays_pdf,
    left_on=["item_id", "date"],
    right_on=["community_code", "date"],
    how="left",
)
ts_pdf.drop(columns=["community_code", "date"], inplace=True)

# Fill missing holidays (regions/dates not in the table) with 0
ts_pdf["is_holiday"] = ts_pdf["is_holiday"].fillna(0).astype(int)
ts_pdf["is_bridge"]  = ts_pdf["is_bridge"].fillna(0).astype(int)

# ========================  SUMMARY  ========================
KNOWN_COV_COLS = WEATHER_COLS + HOLIDAY_COLS  # all known covariates

n_weather_nulls = ts_pdf[WEATHER_COLS].isnull().any(axis=1).sum()
n_holidays = ts_pdf["is_holiday"].sum()
n_bridges  = ts_pdf["is_bridge"].sum()

print(f"Known covariate columns: {KNOWN_COV_COLS}")
print(f"ts_pdf shape: {ts_pdf.shape}")
print(f"Rows with missing weather: {n_weather_nulls} / {len(ts_pdf)} ({100*n_weather_nulls/len(ts_pdf):.1f}%)")
print(f"Holiday intervals: {n_holidays:,}  |  Bridge intervals: {n_bridges:,}")
display(ts_pdf.head(5))

# COMMAND ----------

# DBTITLE 1,MLflow experiment setup
import mlflow
import time
import random

# ---------------------------------------------------------------
# Random run name generator (adjective-noun, like Docker containers)
# ---------------------------------------------------------------
_ADJECTIVES = [
    "bold", "calm", "dark", "eager", "fair", "glad", "keen", "loud",
    "mild", "neat", "odd", "pale", "quick", "rare", "slim", "tall",
    "vast", "warm", "zany", "bright", "crisp", "dense", "fiery",
    "grand", "hardy", "icy", "jolly", "lush", "noble", "proud",
    "rapid", "sharp", "swift", "vivid", "witty", "agile", "brisk",
]
_NOUNS = [
    "aurora", "blaze", "comet", "delta", "ember", "frost", "gale",
    "haze", "iris", "jade", "kite", "lark", "moon", "nova", "opal",
    "peak", "quasar", "ridge", "spark", "tide", "vale", "wave",
    "zenith", "bolt", "cedar", "dawn", "flare", "grove", "hawk",
    "orbit", "pulse", "reef", "storm", "torch", "vortex", "zephyr",
]

def random_run_name():
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{random.randint(10, 99)}"

# ---------------------------------------------------------------
# Set up MLflow experiment — must use /Workspace/ path in Git repos
# ---------------------------------------------------------------
# EXPERIMENT_NAME = "/Workspace/Users/limmied2004@gmail.com/autogluon_forecasting"
# mlflow.set_experiment(EXPERIMENT_NAME)

# End any lingering active run (safe to call even if none exists)
mlflow.end_run()

# Start a new run with a memorable random name
RUN_NAME = random_run_name()
run = mlflow.start_run(run_name=RUN_NAME)
print(f"Run name:     {RUN_NAME}")
print(f"Run ID:       {run.info.run_id}")
print(f"Artifact URI: {run.info.artifact_uri}")

# COMMAND ----------

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

MODEL_DIR = f"/Workspace/datathon/pingpong_club_epoch/AutogluonModels/{RUN_NAME}"
print(f"Model will be saved to: {MODEL_DIR}")

target = "active_kw" #active_kw, avg_kw_per_client

# Weather + holiday columns → known covariates (available for the prediction horizon)
known_cov_names = KNOWN_COV_COLS  # from the covariate processing cell

# --- November holdout: train on Jan–Oct, test on Nov ---
HOLDOUT_START = pd.Timestamp("2025-11-01")

predictor = TimeSeriesPredictor(
    path=MODEL_DIR,
    prediction_length=96, #15 minutes * 96 = 1 day
    target=target, #active_kw, avg_kw_per_client
    eval_metric="MAE", 
    freq="15min",
    known_covariates_names=known_cov_names,
)

# Drop the non-target column, keep weather + holidays + n_clients
if target == "active_kw":
    input_data = ts_pdf.drop(columns=["avg_kw_per_client"])
else:
    input_data = ts_pdf.drop(columns=["active_kw"])

# Split: train on pre-November, hold out November for testing
train_input = input_data[input_data["datetime_15min"] < HOLDOUT_START].copy()
test_input = input_data[input_data["datetime_15min"] >= HOLDOUT_START].copy()

train_data = TimeSeriesDataFrame.from_data_frame(
    train_input,
    id_column="item_id",
    timestamp_column="datetime_15min"
)

print(f"Target: {target}")
print(f"Known covariates: {known_cov_names}")
print(f"\nHoldout split at: {HOLDOUT_START}")
print(f"Train shape: {train_data.shape}  |  {train_input['datetime_15min'].min()} → {train_input['datetime_15min'].max()}")
print(f"Test rows:  {len(test_input):,}  |  {test_input['datetime_15min'].min()} → {test_input['datetime_15min'].max()}")
print(f"Columns: {list(train_data.columns)}")
print(train_data)

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Fit AutoGluon with MLflow parameter logging
# ---------------------------------------------------------------
# Log training parameters & fit the predictor
# ---------------------------------------------------------------
FIT_PARAMS = {
    "prediction_length": 96,
    "target": target,
    "eval_metric": "MAE",
    "freq": "15min",
    "presets": "best_quality",
    "time_limit": 300,
    "train_rows": len(train_data),
    "n_time_series": train_data.num_items,

}
mlflow.log_params(FIT_PARAMS)

# if predictor.is_fit:
#     print("Predictor already fitted — skipping retraining, logging existing results.")
# else:
start_time = time.time()
predictor.fit(
    train_data,
    presets=FIT_PARAMS["presets"],
    verbosity=2,
    time_limit=FIT_PARAMS["time_limit"],
    excluded_model_types=["TemporalFusionTransformer"],
)
training_duration = time.time() - start_time
mlflow.log_metric("training_duration_sec", training_duration)
print(f"\nTraining completed in {training_duration:.1f}s")

# COMMAND ----------

# DBTITLE 1,Log leaderboard, validation metrics, and end MLflow run
# ---------------------------------------------------------------
# Log leaderboard + model metadata
# ---------------------------------------------------------------
leaderboard = predictor.leaderboard()
leaderboard_path = MODEL_DIR + "/leaderboard.csv"
leaderboard.to_csv(leaderboard_path, index=False)
mlflow.log_artifact(leaderboard_path, artifact_path="autogluon")

n_items = train_data.num_items
mlflow.log_param("n_regions", n_items)
mlflow.log_param("known_covariates", str(known_cov_names))
mlflow.log_param("holdout_start", str(HOLDOUT_START))

# Log each model's per-region average MAE
for _, row in leaderboard.iterrows():
    model_name = row["model"].replace(" ", "_").replace("[", "_").replace("]", "")
    mlflow.log_metric(f"val_avg_region_MAE_{model_name}", abs(row["score_val"]))
    mlflow.log_metric(f"val_sum_region_MAE_{model_name}", abs(row["score_val"]) * n_items)

# ---------------------------------------------------------------
# Compute competition-equivalent total MAE
# Predict on held-out last day, sum across regions, compare to actual totals
# ---------------------------------------------------------------
prediction_length = predictor.prediction_length

# Split: remove last prediction_length steps per item for validation
train_subset = train_data.slice_by_timestep(None, -prediction_length)
val_actuals  = train_data.slice_by_timestep(-prediction_length, None)

# Extract known covariates for the validation horizon
# (AutoGluon needs weather forecasts for the days it's predicting)
val_known_covariates = val_actuals[known_cov_names].copy()

# Predict on truncated data, providing future weather forecasts
val_predictions = predictor.predict(train_subset, known_covariates=val_known_covariates)

# Sum predictions across all regions per timestamp → total grid forecast
pred_reset = val_predictions.reset_index()
pred_total = pred_reset.groupby("timestamp")["mean"].sum()

# Sum actuals across all regions per timestamp → total grid actual
actual_reset = val_actuals.reset_index()
actual_total = actual_reset.groupby("timestamp")[target].sum()

# Align and compute MAE on the totals
aligned = pd.DataFrame({"predicted": pred_total, "actual": actual_total}).dropna()
total_MAE = (aligned["predicted"] - aligned["actual"]).abs().mean()

# ---------------------------------------------------------------
# Log summary metrics
# ---------------------------------------------------------------
best_row = leaderboard.iloc[0]
mlflow.set_tag("best_model", best_row["model"])
mlflow.log_metric("best_avg_region_MAE", abs(best_row["score_val"]))
mlflow.log_metric("best_sum_region_MAE", abs(best_row["score_val"]) * n_items)
mlflow.log_metric("competition_total_MAE", total_MAE)
mlflow.set_tag("autogluon_model_path", predictor.path)

print("Leaderboard:")
display(leaderboard)
print(f"\n{'='*55}")
print(f"Regions modeled:                    {n_items}")
print(f"Known covariates:                   {known_cov_names}")
print(f"Best model avg per-region MAE:      {abs(best_row['score_val']):,.1f} kW")
print(f"Sum of regional MAEs (upper bound): {abs(best_row['score_val']) * n_items:,.0f} kW")
print(f"Competition-equivalent total MAE:   {total_MAE:,.0f} kW")
print(f"{'='*55}")
print(f"Model saved at: {predictor.path}")

# COMMAND ----------

# DBTITLE 1,November holdout — rolling daily forecast evaluation
# ---------------------------------------------------------------
# November 2025 holdout — rolling daily forecast evaluation
# Collects 15-min predictions & actuals, plots predicted vs actual
# ---------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.timeseries import TimeSeriesDataFrame

test_days = sorted(test_input["datetime_15min"].dt.normalize().unique())
print(f"Rolling daily evaluation: {len(test_days)} days in November\n")

all_pred_rows = []   # 15-min predicted totals
all_actual_rows = [] # 15-min actual totals
daily_results = []

for i, day in enumerate(test_days):
    day_ts = pd.Timestamp(day)
    day_end = day_ts + pd.Timedelta(days=1)

    # History: all data strictly before this day (expanding window)
    hist_df = input_data[input_data["datetime_15min"] < day_ts].copy()

    # Actuals for this day
    actual_df = input_data[
        (input_data["datetime_15min"] >= day_ts) &
        (input_data["datetime_15min"] < day_end)
    ].copy()

    if len(hist_df) == 0 or len(actual_df) == 0:
        continue

    hist_ts = TimeSeriesDataFrame.from_data_frame(
        hist_df, id_column="item_id", timestamp_column="datetime_15min"
    )
    actual_ts = TimeSeriesDataFrame.from_data_frame(
        actual_df, id_column="item_id", timestamp_column="datetime_15min"
    )

    # Known covariates for the prediction window
    known_cov = actual_ts[known_cov_names].copy()

    # Predict
    preds = predictor.predict(hist_ts, known_covariates=known_cov)

    # Sum across all regions per timestamp → total grid
    pred_total = preds.reset_index().groupby("timestamp")["mean"].sum()
    actual_total = actual_ts.reset_index().groupby("timestamp")[target].sum()

    # Store 15-min rows for plotting
    for ts, val in pred_total.items():
        all_pred_rows.append({"timestamp": ts, "predicted_kW": val})
    for ts, val in actual_total.items():
        all_actual_rows.append({"timestamp": ts, "actual_kW": val})

    # Daily MAE
    aligned = pd.DataFrame({"predicted": pred_total, "actual": actual_total}).dropna()
    day_mae = (aligned["predicted"] - aligned["actual"]).abs().mean()
    daily_results.append({"date": day_ts.date(), "total_MAE_kW": day_mae, "n_steps": len(aligned)})
    print(f"  [{i+1:2d}/{len(test_days)}] {day_ts.date()}: total MAE = {day_mae:,.0f} kW")

# --- Build plot DataFrames ---
pred_df = pd.DataFrame(all_pred_rows).sort_values("timestamp").reset_index(drop=True)
actual_df_plot = pd.DataFrame(all_actual_rows).sort_values("timestamp").reset_index(drop=True)

# --- Summary ---
results_df = pd.DataFrame(daily_results)
nov_avg_mae = results_df["total_MAE_kW"].mean()
nov_median_mae = results_df["total_MAE_kW"].median()

print(f"\n{'='*55}")
print(f"November 2025 Holdout — Rolling Daily Forecast")
print(f"{'='*55}")
print(f"Days evaluated:    {len(results_df)}")
print(f"Avg daily MAE:     {nov_avg_mae:,.0f} kW")
print(f"Median daily MAE:  {nov_median_mae:,.0f} kW")
print(f"Min daily MAE:     {results_df['total_MAE_kW'].min():,.0f} kW")
print(f"Max daily MAE:     {results_df['total_MAE_kW'].max():,.0f} kW")
print(f"{'='*55}")

# --- Plot: Predicted vs Actual (15-min resolution over November) ---
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(actual_df_plot["timestamp"], actual_df_plot["actual_kW"],
        color="#2196F3", linewidth=0.6, alpha=0.85, label="Actual")
ax.plot(pred_df["timestamp"], pred_df["predicted_kW"],
        color="#FF5722", linewidth=0.6, alpha=0.85, label="Predicted")

ax.set_xlabel("November 2025", fontsize=12)
ax.set_ylabel("Total Active kW (all regions)", fontsize=12)
ax.set_title("November 2025 Holdout — Predicted vs Actual (15-min intervals)", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate(rotation=45)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# --- Log to MLflow ---
mlflow.log_metric("nov_holdout_avg_MAE", nov_avg_mae)
mlflow.log_metric("nov_holdout_median_MAE", nov_median_mae)
mlflow.log_metric("nov_holdout_min_MAE", results_df["total_MAE_kW"].min())
mlflow.log_metric("nov_holdout_max_MAE", results_df["total_MAE_kW"].max())
mlflow.end_run()
print("\nMLflow run ended.")
