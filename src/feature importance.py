# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Importance — `dark-ember-57`
# MAGIC
# MAGIC Two views of importance for the covariates the predictor was trained on
# MAGIC (7 weather forecasts + 2 holiday flags):
# MAGIC
# MAGIC 1. **Permutation importance on the ensemble** — model-agnostic; shuffles each feature
# MAGIC    and measures how much the forecast metric worsens.
# MAGIC 2. **Native LightGBM importance on `DirectTabular`** — gain/split counts from the
# MAGIC    LightGBM models that AutoGluon fitted under the hood. Only available if
# MAGIC    `DirectTabular` is part of the ensemble.
# MAGIC
# MAGIC The data pipeline (consumption aggregation + weather + holidays) is identical to
# MAGIC the training notebook, so the feature set AutoGluon sees here matches what it was
# MAGIC trained on.

# COMMAND ----------

# MAGIC %pip install "autogluon.timeseries==1.5.0" --quiet

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# COMMAND ----------

AG_PATH = "/Workspace/datathon/pingpong_club_epoch/AutogluonModels/dark-ember-57"

ADMIN_TO_CC = {
    "Andalusia": "AN",       "Aragon": "AR",          "Asturias": "AS",
    "Cantabria": "CB",       "Castille-Leon": "CL",   "Castille-La Mancha": "CM",
    "Canary Islands": "CN",  "Catalonia": "CT",       "Extremadura": "EX",
    "Galicia": "GA",         "Murcia": "MC",          "Madrid": "MD",
    "Navarre": "NC",         "Basque Country": "PV",  "La Rioja": "RI",
    "Valencia": "VC",
}

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
WEATHER_COLS   = list(WEATHER_RENAME.values())
HOLIDAY_COLS   = ["is_holiday", "is_bridge"]
KNOWN_COV_COLS = WEATHER_COLS + HOLIDAY_COLS
TARGET         = "active_kw"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data (same pipeline as training / submission)

# COMMAND ----------

def load_weather_15min():
    w_spark = spark.table("datathon.pingpong_club_epoch.weather_hourly_2")
    w_pdf = w_spark.select("admin_name", "date", *WEATHER_RAW_COLS).toPandas()
    w_pdf["community_code"] = w_pdf["admin_name"].map(ADMIN_TO_CC)
    w_pdf = w_pdf.dropna(subset=["community_code"])
    w_pdf["date"] = pd.to_datetime(w_pdf["date"])
    w_agg = (
        w_pdf.groupby(["community_code", "date"])[WEATHER_RAW_COLS]
              .mean().reset_index().rename(columns=WEATHER_RENAME)
    )
    parts = []
    for cc, grp in w_agg.groupby("community_code"):
        grp = grp.set_index("date").sort_index()
        grp_15 = grp[WEATHER_COLS].resample("15min").ffill()
        grp_15["item_id"] = cc
        parts.append(grp_15.reset_index().rename(columns={"date": "datetime_15min"}))
    return pd.concat(parts, ignore_index=True)


def load_holidays():
    h_pdf = spark.table("datathon.pingpong_club_epoch.holidays_regional").toPandas()
    h_pdf["date"] = pd.to_datetime(h_pdf["date"])
    h_pdf["is_holiday"] = h_pdf["is_holiday"].astype(int)
    h_pdf["is_bridge"]  = h_pdf["is_bridge"].astype(int)
    return h_pdf[["date", "community_code", "is_holiday", "is_bridge"]]


df = spark.table("datathon.shared.client_consumption")
agg = (
    df.withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
      .groupBy("datetime_15min", "community_code")
      .agg(
          F.sum("active_kw").alias("active_kw"),
          F.countDistinct("client_id").alias("n_clients"),
      )
      .orderBy("datetime_15min", "community_code")
)
ts_pdf = agg.toPandas()
ts_pdf["datetime_15min"] = pd.to_datetime(ts_pdf["datetime_15min"])
ts_pdf = ts_pdf.rename(columns={"community_code": "item_id"})
ts_pdf = ts_pdf.sort_values(["item_id", "datetime_15min"]).reset_index(drop=True)

# Weather + holidays merge
ts_pdf = ts_pdf.merge(load_weather_15min(), on=["item_id", "datetime_15min"], how="left")
h_pdf  = load_holidays()
ts_pdf["date"] = ts_pdf["datetime_15min"].dt.normalize()
ts_pdf = ts_pdf.merge(
    h_pdf, left_on=["item_id", "date"], right_on=["community_code", "date"], how="left"
)
ts_pdf.drop(columns=["community_code", "date"], inplace=True)
ts_pdf["is_holiday"] = ts_pdf["is_holiday"].fillna(0).astype(int)
ts_pdf["is_bridge"]  = ts_pdf["is_bridge"].fillna(0).astype(int)

print(f"Loaded {len(ts_pdf):,} rows × {ts_pdf['item_id'].nunique()} items")
print(f"Date range: {ts_pdf['datetime_15min'].min()} → {ts_pdf['datetime_15min'].max()}")
display(ts_pdf.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build TimeSeriesDataFrame + load predictor

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

data_ts = TimeSeriesDataFrame.from_data_frame(
    ts_pdf[["item_id", "datetime_15min", TARGET, "n_clients", *KNOWN_COV_COLS]],
    id_column="item_id",
    timestamp_column="datetime_15min",
)

predictor = TimeSeriesPredictor.load(AG_PATH)
leaderboard = predictor.leaderboard()
print("Leaderboard:")
display(leaderboard)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Permutation importance (ensemble)
# MAGIC
# MAGIC Uses the last `RECENT_DAYS` of history to keep the permutation loop fast.
# MAGIC Increase `num_iterations` for tighter confidence intervals (at proportional cost).

# COMMAND ----------

import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

RECENT_DAYS     = 14
NUM_ITERATIONS  = 3
FI_TIME_LIMIT_S = 600

# Rebuild data_ts with n_clients (required by the predictor but missing from Cell 9)
data_ts = TimeSeriesDataFrame.from_data_frame(
    ts_pdf[["item_id", "datetime_15min", TARGET, "n_clients", *KNOWN_COV_COLS]],
    id_column="item_id",
    timestamp_column="datetime_15min",
)

cutoff = data_ts.index.get_level_values("timestamp").max() - pd.Timedelta(days=RECENT_DAYS)
subset = TimeSeriesDataFrame(
    data_ts[data_ts.index.get_level_values("timestamp") >= cutoff]
)
print(f"Permuting features over last {RECENT_DAYS} days ({len(subset):,} rows)")

try:
    fi_ensemble = predictor.feature_importance(
        data=subset,
        num_iterations=NUM_ITERATIONS,
        time_limit=FI_TIME_LIMIT_S,
        random_seed=0,
    )
    print("Ensemble permutation importance:")
except RuntimeError as e:
    print(f"Ensemble failed ({e}), falling back to DirectTabular...")
    fi_ensemble = predictor.feature_importance(
        data=subset,
        model="DirectTabular",
        num_iterations=NUM_ITERATIONS,
        time_limit=FI_TIME_LIMIT_S,
        random_seed=0,
    )
    print("DirectTabular permutation importance (ensemble unavailable):")

display(fi_ensemble.sort_values("importance", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Permutation importance on `DirectTabular` (if present)
# MAGIC
# MAGIC This isolates the LightGBM-backed model from the ensemble — useful if you want to
# MAGIC know which features LightGBM itself leans on.

# COMMAND ----------

available_models = set(leaderboard["model"].tolist())
lgbm_model_name  = next(
    (m for m in ("DirectTabular", "RecursiveTabular") if m in available_models),
    None,
)

if lgbm_model_name is not None:
    print(f"Computing permutation importance for: {lgbm_model_name}")
    fi_lgbm = predictor.feature_importance(
        data=subset,
        model=lgbm_model_name,
        num_iterations=NUM_ITERATIONS,
        time_limit=FI_TIME_LIMIT_S,
        random_seed=0,
    )
    display(fi_lgbm.sort_values("importance", ascending=False))
else:
    print("No DirectTabular / RecursiveTabular model in the ensemble.")
    fi_lgbm = None

# COMMAND ----------

# DBTITLE 1,DirectTabular FI with feature names
if fi_lgbm is not None:
    fi_named = (
        fi_lgbm
        .rename_axis("feature")
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    display(fi_named)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Native LightGBM importance (gain / split) from `DirectTabular`
# MAGIC
# MAGIC Reaches into AutoGluon's internal trainer. Schema is LightGBM-specific and may
# MAGIC include engineered lag features in addition to the known_covariates.

# COMMAND ----------

native_importance = None
try:
    if lgbm_model_name is not None:
        mw_model = predictor._learner.load_trainer().load_model(lgbm_model_name)
        # MultiWindowBacktestingModel → DirectTabularModel → TabularModel → LGBModel → Booster
        dt_model = mw_model.most_recent_model
        booster = dt_model.get_tabular_model().model.model  # lightgbm.basic.Booster

        names = booster.feature_name()
        gain  = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")

        native_importance = pd.DataFrame({
            "feature": names,
            "gain": gain,
            "split": split,
        }).sort_values("gain", ascending=False).reset_index(drop=True)

        print(f"Native LightGBM importance — {lgbm_model_name} "
              f"(window {mw_model.most_recent_model_folder}, {len(names)} features):")
        display(native_importance)
    else:
        print("No DirectTabular / RecursiveTabular model in the ensemble.")
except Exception as e:
    print(f"Native importance extraction failed: {e}")
    import traceback; traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plot

# COMMAND ----------

def _plot(df, col, title):
    df2 = df.sort_values(col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(df2) + 1.5))
    ax.barh(df2.index, df2[col])
    ax.set_xlabel(col)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

_plot(fi_ensemble, "importance", "Permutation importance — ensemble (dark-ember-57)")
if fi_lgbm is not None:
    _plot(fi_lgbm, "importance", f"Permutation importance — {lgbm_model_name}")
if native_importance is not None:
    native_df = native_importance.to_frame("importance") if isinstance(native_importance, pd.Series) else native_importance
    _plot(native_df, native_df.columns[0], f"Native LightGBM importance — {lgbm_model_name}")
