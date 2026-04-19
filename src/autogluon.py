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
display(df)

# COMMAND ----------

# DBTITLE 1,Aggregate to a simple pandas DataFrame for AutoGluon prototyping
# ---------------------------------------------------------------
# Aggregate 182M raw rows → ~32K rows (one per 15-min interval)
# This is the actual prediction target: total active_kw across all clients
# ---------------------------------------------------------------
agg_df = (
    df
    .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
    .groupBy("datetime_15min")
    .agg(
        F.sum("active_kw").alias("active_kw"),
        F.countDistinct("client_id").alias("n_clients"),  # useful sanity-check column
    )
    .orderBy("datetime_15min")
)

# Safe to .toPandas() — only ~32K rows
ts_pdf = agg_df.toPandas()
ts_pdf["datetime_15min"] = pd.to_datetime(ts_pdf["datetime_15min"])
ts_pdf = ts_pdf.sort_values("datetime_15min").reset_index(drop=True)
ts_pdf['item_id'] = "Total_Grid"
ts_pdf["avg_kw_per_client"] = ts_pdf["active_kw"] / ts_pdf["n_clients"]

print(f"Shape: {ts_pdf.shape}")
print(f"Date range: {ts_pdf['datetime_15min'].min()} → {ts_pdf['datetime_15min'].max()}")
print(f"Memory: {ts_pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")
display(ts_pdf.head(10))

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

target = "avg_kw_per_client" #active_kw, avg_kw_per_client

predictor = TimeSeriesPredictor(
    path=MODEL_DIR,
    prediction_length=96, #15 minutes * 96 = 1 day
    target=target, #active_kw, avg_kw_per_client
    eval_metric="MAE", 
    freq="15min"
    # known_covariates_names=["temp_forecast", "is_holiday"]
)

#drop the non-target column
if target == "active_kw":
    input_data = ts_pdf.drop(columns=["avg_kw_per_client"])
else:
    input_data = ts_pdf.drop(columns=["active_kw"])

train_data = TimeSeriesDataFrame.from_data_frame(
    input_data,
    id_column="item_id",
    timestamp_column="datetime_15min"
)

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
    "time_limit": 120,
    "train_rows": len(train_data),
    "n_time_series": train_data.num_items,

}
mlflow.log_params(FIT_PARAMS)

if predictor.is_fit:
    print("Predictor already fitted — skipping retraining, logging existing results.")
else:
    start_time = time.time()
    predictor.fit(
        train_data,
        presets="best_quality",
        verbosity=2,
        time_limit=120,
        excluded_model_types=["TemporalFusionTransformer"],
    )
    training_duration = time.time() - start_time
    mlflow.log_metric("training_duration_sec", training_duration)
    print(f"\nTraining completed in {training_duration:.1f}s")

# COMMAND ----------

# DBTITLE 1,Log leaderboard, validation metrics, and end MLflow run
# ---------------------------------------------------------------
# Log leaderboard + model metadata (fast — no prediction needed)
# ---------------------------------------------------------------
leaderboard = predictor.leaderboard()
leaderboard_path = MODEL_DIR + "/leaderboard.csv"
leaderboard.to_csv(leaderboard_path, index=False)
mlflow.log_artifact(leaderboard_path, artifact_path="autogluon")

# Scaling factor: latest n_clients from training data
# Used to convert avg_kw_per_client metrics → total active_kw
latest_n_clients = int(train_data["n_clients"].iloc[-1])
mlflow.log_param("latest_n_clients", latest_n_clients)

# Log each model's validation score as a metric
for _, row in leaderboard.iterrows():
    model_name = row["model"].replace(" ", "_").replace("[", "_").replace("]", "")
    mlflow.log_metric(f"val_score_{model_name}", row["score_val"])
    # Also log the total-equivalent MAE for cross-target comparison
    mlflow.log_metric(f"val_total_MAE_{model_name}", abs(row["score_val"]) * latest_n_clients)

# Log best model info
mlflow.set_tag("best_model", leaderboard.iloc[0]["model"])
mlflow.log_metric("best_val_MAE", abs(leaderboard.iloc[0]["score_val"]))
mlflow.log_metric("best_val_MAE_total", abs(leaderboard.iloc[0]["score_val"]) * latest_n_clients)
mlflow.set_tag("autogluon_model_path", predictor.path)

print("Leaderboard:")
display(leaderboard)
print(f"\nScaling: avg MAE × {latest_n_clients} clients = total MAE")
print(f"Best model total MAE: {abs(leaderboard.iloc[0]['score_val']) * latest_n_clients:,.0f}")
print(f"Model saved locally at: {predictor.path}")

mlflow.end_run()
print("MLFlow run ended")
