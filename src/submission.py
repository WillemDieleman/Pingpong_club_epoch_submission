# Databricks notebook source
# MAGIC %md
# MAGIC # Energy Consumption Prediction: Submission
# MAGIC
# MAGIC ## Objective
# MAGIC Predict the **total (aggregated) energy consumption** across all clients for each
# MAGIC **15-minute interval** in **2026**.
# MAGIC
# MAGIC ## How It Works
# MAGIC 1. Add as many cells as you need **above** the Submit cell: install packages, import
# MAGIC    libraries, load additional data, do pre-computation, and define your `EnergyConsumptionModel` class.
# MAGIC 2. Run the **Submit** cell (the last cell). This triggers a scoring job that:
# MAGIC    - Re-runs **this entire notebook** with access to the full dataset (2025 + 2026)
# MAGIC    - Your `%pip install` commands, imports, and model class all run exactly as if you ran them yourself
# MAGIC    - Calls `model.predict()` to generate predictions for 2026
# MAGIC    - Computes your MAE and records it on the leaderboard
# MAGIC 3. Your **MAE score** and remaining submissions are printed once the job finishes.
# MAGIC
# MAGIC ## Model Contract
# MAGIC `predict(self, df, predict_start, predict_end)` receives a **PySpark DataFrame** with
# MAGIC **all** data (2025 + 2026):
# MAGIC - Columns: `client_id` (int), `datetime_local` (timestamp), `community_code` (string), `active_kw` (double)
# MAGIC - `predict_start` / `predict_end` define the prediction window
# MAGIC - `spark` is available as a global (you can use `spark.table()`, `spark.createDataFrame()`, etc.)
# MAGIC
# MAGIC It must return a **PySpark DataFrame** with exactly two columns:
# MAGIC - `datetime_15min` (timestamp): the 15-minute interval (floor of the timestamp)
# MAGIC - `prediction` (double): the **total** predicted `active_kw` across all clients for that interval
# MAGIC - One row per 15-minute interval in the prediction window
# MAGIC
# MAGIC ## Rules
# MAGIC - **Limited submissions** per team. Use the **exploration notebook** for local validation before submitting.
# MAGIC - Only **successful** submissions count towards the limit.
# MAGIC - **Do not modify the Submit cell** (the last cell). Everything else is yours to change.
# MAGIC
# MAGIC ## Performance Tips
# MAGIC - Use **PySpark** for all heavy data processing. Avoid `.toPandas()` on the full dataset.
# MAGIC - If using ML libraries (LightGBM, sklearn, etc.), do feature engineering in PySpark first, then
# MAGIC   `.toPandas()` only the **compact feature matrix** (e.g. aggregated 15-min intervals).
# MAGIC - Scoring has a **timeout (45 min)**. Keep the PySpark→pandas conversion for the very last step.
# MAGIC
# MAGIC ## Evaluation
# MAGIC **MAE (Mean Absolute Error)** on the aggregated 15-minute totals. Lower is better.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Packages (optional)
# MAGIC
# MAGIC Add `%pip install` cells here if your model needs additional packages.
# MAGIC These will also be installed during scoring.

# COMMAND ----------

# MAGIC %pip install "autogluon.timeseries==1.5.0" --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

# %restart_python  # NOT needed with %pip — and breaks scoring by resetting widget state

# COMMAND ----------

# >>> Set your team name <<<
TEAM_NAME = "pingpong_club_epoch"  # lowercase, no spaces, use underscores

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# COMMAND ----------

# !nvidia-smi

# COMMAND ----------

# import os
# from transformers import AutoConfig

# # Force transformers to use a specific, writable local disk in Databricks
# os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf_cache'

# # Try to pre-load the config to see if it works outside of AutoGluon
# config = AutoConfig.from_pretrained("autogluon/chronos-bolt-small")
# print("Config loaded successfully!")

# COMMAND ----------

class EnergyConsumptionModel:
    """
    Energy consumption forecasting model using AutoGluon TimeSeriesPredictor.
    Rolling 1-day-ahead predictions to avoid data leakage:
    each day is predicted using only actual data observed before that day.
    """

    @staticmethod
    def convert_train_data(df):
        """Aggregate PySpark DF to 15-min total energy → AutoGluon TimeSeriesDataFrame."""
        agg_df = (
            df
            .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
            .groupBy("datetime_15min")
            .agg(
                F.sum("active_kw").alias("active_kw"),
                F.countDistinct("client_id").alias("n_clients"),
            )
            .orderBy("datetime_15min")
        )

        ts_pdf = agg_df.toPandas()
        ts_pdf["datetime_15min"] = pd.to_datetime(ts_pdf["datetime_15min"])
        ts_pdf = ts_pdf.sort_values("datetime_15min").reset_index(drop=True)
        ts_pdf["item_id"] = "Total_Grid"
        ts_pdf["avg_kw_per_client"] = ts_pdf["active_kw"] / ts_pdf["n_clients"]

        train_data = TimeSeriesDataFrame.from_data_frame(
            ts_pdf,
            id_column="item_id",
            timestamp_column="datetime_15min",
        )
        return train_data

    def predict(self, df, predict_start, predict_end):
        # Load pre-trained predictor (prediction_length=96, i.e. 1 day of 15-min intervals)
        ag_path = "/Workspace/datathon/pingpong_club_epoch/AutogluonModels/hardy-iris-90"
        predictor = TimeSeriesPredictor.load(ag_path)
        target = "avg_kw_per_client" #active_kw, avg_kw_per_client
        # AVERAGE_PER_CLIENT = True

        # Aggregate full dataset to 15-min totals as TimeSeriesDataFrame
        all_ts = EnergyConsumptionModel.convert_train_data(df)
        if target == "active_kw":
            input_data = all_ts.drop(columns=["avg_kw_per_client"])
        else:
            input_data = all_ts.drop(columns=["active_kw"])

        predict_start_ts = pd.Timestamp(predict_start)
        predict_end_ts = pd.Timestamp(predict_end)

        # ── Rolling 1-day-ahead prediction ──────────────────────────
        # For each day D in the prediction window:
        #   1. Slice actual data up to (not including) day D  →  no leakage
        #   2. predictor.predict()  →  96 forecasts for day D
        #   3. Advance to day D+1 (actual data for D is now in the slice)
        all_preds = []
        current_day = predict_start_ts.normalize()

        while current_day < predict_end_ts:
            # Only actual observations before the day we are predicting
            mask = all_ts.index.get_level_values("timestamp") < current_day
            train_slice = TimeSeriesDataFrame(all_ts[mask])

            # 1-day-ahead forecast (96 × 15-min intervals)
            forecast = predictor.predict(train_slice)

            # Extract point predictions (AutoGluon returns 'mean' as the point forecast)
            pred_pdf = forecast.reset_index()[["timestamp", "mean"]].rename(
                columns={"timestamp": "datetime_15min", "mean": "prediction"}
            )
            if target == "avg_kw_per_client":
                pred_pdf["prediction"] = pred_pdf["prediction"] * train_slice["n_clients"].iloc[-1]
                # print(pred_pdf["prediction"].iloc[0])

            all_preds.append(pred_pdf)

            current_day += pd.Timedelta(days=1)
            print(f"Predicted up to {current_day}")

        # Combine all daily forecasts, filter to exact window, enforce types
        result_pdf = pd.concat(all_preds, ignore_index=True)
        result_pdf["datetime_15min"] = pd.to_datetime(result_pdf["datetime_15min"])
        result_pdf["prediction"] = result_pdf["prediction"].astype("float64")
        result_pdf = result_pdf[
            (result_pdf["datetime_15min"] >= predict_start_ts)
            & (result_pdf["datetime_15min"] < predict_end_ts)
        ]

        # Return as PySpark DataFrame (required by submission contract)
        return spark.createDataFrame(result_pdf[["datetime_15min", "prediction"]])

# COMMAND ----------

# test to see if model works
# df = spark.table("datathon.shared.client_consumption")
# predict_start = "2025-11-24"
# predict_end = "2025-12-01"

# COMMAND ----------

# model = EnergyConsumptionModel()
# result = model.predict(df, predict_start, predict_end)
# print(result)

# COMMAND ----------

# Get the last 7 days of actuals and predictions for plotting
# import matplotlib.pyplot as plt

# # Define the last 7 days of the train set
# train_end = pd.to_datetime(predict_end)
# plot_start = train_end - pd.Timedelta(days=7)
# plot_end = train_end

# # Aggregate actuals to 15-min intervals
# actuals_df = (
#     df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
#       .groupBy("datetime_15min")
#       .agg(F.sum("active_kw").alias("active_kw"))
#       .filter((F.col("datetime_15min") >= F.lit(plot_start)) & (F.col("datetime_15min") < F.lit(plot_end)))
#       .orderBy("datetime_15min")
# )
# actuals_pdf = actuals_df.toPandas()
# actuals_pdf["datetime_15min"] = pd.to_datetime(actuals_pdf["datetime_15min"])

# # Get model predictions for the same period
# model = EnergyConsumptionModel()
# preds_df = model.predict(df, plot_start.strftime("%Y-%m-%d"), plot_end.strftime("%Y-%m-%d"))
# preds_pdf = preds_df.toPandas()
# preds_pdf["datetime_15min"] = pd.to_datetime(preds_pdf["datetime_15min"])

# # Merge actuals and predictions
# merged = pd.merge(actuals_pdf, preds_pdf, on="datetime_15min", how="inner")

# # Plot
# plt.figure(figsize=(16, 5))
# plt.plot(merged["datetime_15min"], merged["active_kw"], label="Actual", color="black")
# plt.plot(merged["datetime_15min"], merged["prediction"], label="Prediction", color="red", alpha=0.7)
# plt.legend()
# plt.title("Final 7 Days: Actual vs. Predicted Total Energy Consumption")
# plt.xlabel("Datetime (15-min intervals)")
# plt.ylabel("Total Active kW")
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# (result.count(), len(result.columns))

# COMMAND ----------

# class EnergyConsumptionModel:
#     """
#     Energy consumption forecasting model.

#     predict(df, predict_start, predict_end):
#         Given data as a PySpark DataFrame, return predictions for every
#         15-min interval between predict_start and predict_end.
#     """

#     def predict(self, df, predict_start, predict_end):
#         """
#         Args:
#             df: PySpark DataFrame with columns client_id, datetime_local,
#                          community_code, active_kw.
#             predict_start: Start of the prediction period (inclusive), e.g. "2026-01-01".
#             predict_end: End of the prediction period (exclusive), e.g. "2027-01-01".

#         Must return a PySpark DataFrame with columns:
#           - datetime_15min (timestamp): the 15-minute interval
#           - prediction (double): total predicted active_kw for that interval
#         """
#         # Floor timestamps to 15-minute intervals
#         df = df.withColumn(
#             "datetime_15min",
#             F.window("datetime_local", "15 minutes").start
#         )

#         # Aggregate: total active_kw per 15-min interval (across all clients)
#         agg = df.groupBy("datetime_15min").agg(
#             F.sum("active_kw").alias("active_kw")
#         ).orderBy("datetime_15min")

#         # 7-day lag: use lag with window ordered by datetime_15min
#         w = Window.orderBy("datetime_15min")
#         intervals_per_week = 7 * 96  # 7 days × 96 intervals/day
#         agg = agg.withColumn("lag_7d", F.lag("active_kw", intervals_per_week).over(w))

#         # Historical mean for fallback (using data before prediction period)
#         hist_mean = agg.filter(F.col("datetime_15min") < predict_start).agg(
#             F.mean("active_kw")
#         ).collect()[0][0]

#         # Predict intervals in [predict_start, predict_end)
#         result = agg.filter(
#             (F.col("datetime_15min") >= predict_start) &
#             (F.col("datetime_15min") < predict_end)
#         ).select(
#             "datetime_15min",
#             F.coalesce(F.col("lag_7d"), F.lit(hist_mean)).alias("prediction")
#         )

#         return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submit for Scoring
# MAGIC
# MAGIC **⚠️ DO NOT CHANGE THIS CELL ⚠️**
# MAGIC
# MAGIC When you run this cell interactively, it triggers the scoring job.
# MAGIC When the scoring job re-runs this notebook, this cell generates predictions
# MAGIC and writes them for evaluation.

# COMMAND ----------

# ============================================================
# ⚠️  DO NOT CHANGE THIS CELL — submission will break  ⚠️
# ============================================================

# Provided by the organizers. Do not change.
SCORING_JOB_ID = 69902472640886  # Set automatically during setup

# --- Internal mode detection (set by the scoring job) ---
dbutils.widgets.text("mode", "interactive")
_MODE = dbutils.widgets.get("mode").strip()

if _MODE == "score":
    # ---- Score mode: generate predictions and exit ----
    from pyspark.sql import functions as _F

    _predict_start = dbutils.widgets.get("predict_start").strip()
    _predict_end = dbutils.widgets.get("predict_end").strip()

    _full_df = spark.table("datathon.shared.client_consumption")
    _model = EnergyConsumptionModel()
    _predictions = _model.predict(_full_df, _predict_start, _predict_end)

    _predictions_table = "datathon.evaluation.submissions"
    (
        _predictions
        .withColumn("team_name", _F.lit(TEAM_NAME))
        .withColumn("submitted_at", _F.current_timestamp())
        .select("team_name", "datetime_15min", "prediction", "submitted_at")
        .write.mode("overwrite").saveAsTable(_predictions_table)
    )
    print(f"Wrote {_predictions.count():,} predictions to {_predictions_table}")
    dbutils.notebook.exit("ok")

# ---- Interactive mode: trigger the scoring job ----
import json
import datetime as dt
from databricks.sdk import WorkspaceClient

assert SCORING_JOB_ID is not None, "SCORING_JOB_ID has not been set. Ask the organisers."
assert TEAM_NAME != "my_team", "Please set your TEAM_NAME in the configuration cell before submitting."

_w = WorkspaceClient()
_submitter_email = _w.current_user.me().user_name
_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)

print(f"Submitting as: {_submitter_email}")
print(f"Notebook: {_notebook_path}")

_job_run = _w.jobs.run_now(
    job_id=SCORING_JOB_ID,
    notebook_params={
        "team_name": TEAM_NAME,
        "submitter_email": _submitter_email,
        "notebook_path": _notebook_path,
    },
)
print("Job triggered. Waiting for scoring to finish (this may take a few minutes) ...")

try:
    _job_run = _job_run.result(timeout=dt.timedelta(minutes=50))
    _tasks = _w.jobs.get_run(_job_run.run_id).tasks
    _task_run_id = _tasks[0].run_id
    _output = _w.jobs.get_run_output(_task_run_id)
    _result = json.loads(_output.notebook_output.result)
except Exception as e:
    print(f"\nScoring job failed: {e}")
    _result = None

if _result and _result["status"] == "success":
    print(f"\n{'='*50}")
    print(f"  Team: {_result['team_name']}")
    print(f"  MAE:  {_result['mae']:.6f}")
    print(f"  Submissions remaining: {_result['submissions_remaining']}")
    print(f"{'='*50}")
elif _result:
    print(f"\nSubmission FAILED: {_result['message']}")
    if "submissions_remaining" in _result:
        print(f"Submissions remaining: {_result['submissions_remaining']}")
