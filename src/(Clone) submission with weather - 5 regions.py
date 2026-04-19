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

from itertools import chain

class EnergyConsumptionModel:
    """
    Energy consumption forecasting model using AutoGluon TimeSeriesPredictor
    with weather and holiday covariates, 5 climate-group time series.
    Rolling 1-day-ahead predictions to avoid data leakage.
    """

    # Community code → 5 climate groups
    CC_TO_GROUP = {
        "GA": "Atlantic_North",  "AS": "Atlantic_North",  "CB": "Atlantic_North",
        "PV": "Atlantic_North",  "NC": "Atlantic_North",  "RI": "Atlantic_North",
        "CL": "North_Central_Plateau", "MD": "North_Central_Plateau",
        "CM": "Hot_Interior",    "EX": "Hot_Interior",    "AN": "Hot_Interior",   "AR": "Hot_Interior",
        "CT": "NE_Mediterranean",
        "VC": "Levante_Islands", "MC": "Levante_Islands", "CN": "Levante_Islands",
        "CE": "Levante_Islands", "ML": "Levante_Islands",
    }

    # Admin name → community code (for weather table)
    ADMIN_TO_CC = {
        "Andalusia": "AN",  "Aragon": "AR",       "Asturias": "AS",
        "Cantabria": "CB",  "Castille-Leon": "CL", "Castille-La Mancha": "CM",
        "Canary Islands": "CN", "Catalonia": "CT", "Extremadura": "EX",
        "Galicia": "GA",    "Murcia": "MC",        "Madrid": "MD",
        "Navarre": "NC",    "Basque Country": "PV", "La Rioja": "RI",
        "Valencia": "VC",
    }
    # Admin name → climate group (for weather aggregation)
    ADMIN_TO_GROUP = {adm: CC_TO_GROUP[cc] for adm, cc in ADMIN_TO_CC.items()}

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
    HOLIDAY_COLS = ["is_holiday", "is_bridge"]
    KNOWN_COV_COLS = WEATHER_COLS + HOLIDAY_COLS

    @staticmethod
    def _load_weather_15min():
        """Load weather forecasts, aggregate per climate group, resample to 15-min."""
        weather_spark = spark.table("datathon.pingpong_club_epoch.weather_hourly_2")
        weather_pdf = weather_spark.select(
            "admin_name", "date", *EnergyConsumptionModel.WEATHER_RAW_COLS
        ).toPandas()
        weather_pdf["group"] = weather_pdf["admin_name"].map(
            EnergyConsumptionModel.ADMIN_TO_GROUP
        )
        weather_pdf = weather_pdf.dropna(subset=["group"])
        weather_pdf["date"] = pd.to_datetime(weather_pdf["date"])

        # Average across all cities within each climate group per hour
        weather_agg = (
            weather_pdf
            .groupby(["group", "date"])[EnergyConsumptionModel.WEATHER_RAW_COLS]
            .mean()
            .reset_index()
            .rename(columns=EnergyConsumptionModel.WEATHER_RENAME)
        )

        # Resample hourly → 15-min (forward-fill within each hour)
        weather_15min_parts = []
        for grp_name, grp in weather_agg.groupby("group"):
            grp = grp.set_index("date").sort_index()
            grp_15 = grp[EnergyConsumptionModel.WEATHER_COLS].resample("15min").ffill()
            grp_15["item_id"] = grp_name
            weather_15min_parts.append(
                grp_15.reset_index().rename(columns={"date": "datetime_15min"})
            )

        return pd.concat(weather_15min_parts, ignore_index=True)

    @staticmethod
    def _load_holidays():
        """Load regional holidays, aggregate per climate group (max within group)."""
        holidays_pdf = spark.table("datathon.pingpong_club_epoch.holidays_regional").toPandas()
        holidays_pdf["date"] = pd.to_datetime(holidays_pdf["date"])
        holidays_pdf["is_holiday"] = holidays_pdf["is_holiday"].astype(int)
        holidays_pdf["is_bridge"] = holidays_pdf["is_bridge"].astype(int)
        holidays_pdf["group"] = holidays_pdf["community_code"].map(
            EnergyConsumptionModel.CC_TO_GROUP
        )
        holidays_pdf = holidays_pdf.dropna(subset=["group"])
        # If ANY region in the group has a holiday, the whole group gets it
        holidays_grp = (
            holidays_pdf
            .groupby(["group", "date"])[["is_holiday", "is_bridge"]]
            .max()
            .reset_index()
        )
        return holidays_grp

    @staticmethod
    def convert_train_data(df):
        """Aggregate PySpark DF to 15-min per climate group, merge weather + holidays."""
        # PySpark mapping expression: community_code → climate group
        _map_expr = F.create_map([F.lit(x) for x in chain(*EnergyConsumptionModel.CC_TO_GROUP.items())])

        agg_df = (
            df
            .withColumn("region_group", _map_expr[F.col("community_code")])
            .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
            .groupBy("datetime_15min", "region_group")
            .agg(
                F.sum("active_kw").alias("active_kw"),
                F.countDistinct("client_id").alias("n_clients"),
            )
            .orderBy("datetime_15min", "region_group")
        )

        ts_pdf = agg_df.toPandas()
        ts_pdf["datetime_15min"] = pd.to_datetime(ts_pdf["datetime_15min"])
        ts_pdf = ts_pdf.sort_values(["region_group", "datetime_15min"]).reset_index(drop=True)
        ts_pdf = ts_pdf.rename(columns={"region_group": "item_id"})

        # Merge weather (aggregated per climate group)
        weather_15min = EnergyConsumptionModel._load_weather_15min()
        ts_pdf = ts_pdf.merge(weather_15min, on=["item_id", "datetime_15min"], how="left")

        # Merge holidays (aggregated per climate group using max)
        holidays_grp = EnergyConsumptionModel._load_holidays()
        ts_pdf["date"] = ts_pdf["datetime_15min"].dt.normalize()
        ts_pdf = ts_pdf.merge(
            holidays_grp,
            left_on=["item_id", "date"],
            right_on=["group", "date"],
            how="left",
        )
        ts_pdf.drop(columns=["group", "date"], inplace=True)
        ts_pdf["is_holiday"] = ts_pdf["is_holiday"].fillna(0).astype(int)
        ts_pdf["is_bridge"] = ts_pdf["is_bridge"].fillna(0).astype(int)

        return ts_pdf

    def predict(self, df, predict_start, predict_end):
        # Load pre-trained predictor (5 climate groups, weather + holiday covariates)
        ag_path = "/Workspace/datathon/pingpong_club_epoch/AutogluonModels/proud-iris-38"
        predictor = TimeSeriesPredictor.load(ag_path)
        target = "active_kw"

        # Aggregate full dataset per climate group, merge weather + holidays
        ts_pdf = EnergyConsumptionModel.convert_train_data(df)

        # Prepare input data (keep only model columns)
        input_data = ts_pdf.drop(columns=["n_clients"], errors="ignore")

        predict_start_ts = pd.Timestamp(predict_start)
        predict_end_ts = pd.Timestamp(predict_end)

        # ── Rolling 1-day-ahead prediction ──────────────────────────
        all_preds = []
        current_day = predict_start_ts.normalize()

        while current_day < predict_end_ts:
            day_end = current_day + pd.Timedelta(days=1)

            # History: all data strictly before this day (expanding window)
            hist_df = input_data[input_data["datetime_15min"] < current_day].copy()

            # Known covariates for this day (weather + holidays)
            day_cov_df = input_data[
                (input_data["datetime_15min"] >= current_day) &
                (input_data["datetime_15min"] < day_end)
            ].copy()

            if len(hist_df) == 0:
                current_day += pd.Timedelta(days=1)
                continue

            hist_ts = TimeSeriesDataFrame.from_data_frame(
                hist_df, id_column="item_id", timestamp_column="datetime_15min"
            )

            if len(day_cov_df) > 0:
                cov_ts = TimeSeriesDataFrame.from_data_frame(
                    day_cov_df, id_column="item_id", timestamp_column="datetime_15min"
                )
                known_cov = cov_ts[EnergyConsumptionModel.KNOWN_COV_COLS].copy()
                forecast = predictor.predict(hist_ts, known_covariates=known_cov)
            else:
                forecast = predictor.predict(hist_ts)

            # Sum across all groups per timestamp → total grid
            pred_total = forecast.reset_index().groupby("timestamp")["mean"].sum().reset_index()
            pred_total.columns = ["datetime_15min", "prediction"]

            all_preds.append(pred_total)
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

        return spark.createDataFrame(result_pdf[["datetime_15min", "prediction"]])

# COMMAND ----------

# # test to see if model works
# df = spark.table("datathon.shared.client_consumption")
# predict_start = "2025-11-24"
# predict_end = "2025-12-01"

# COMMAND ----------

# model = EnergyConsumptionModel()
# result = model.predict(df, predict_start, predict_end)
# display(result)

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
