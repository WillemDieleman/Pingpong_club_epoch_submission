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



# COMMAND ----------

!uv init
!uv add autogluon[tabarena]==1.5.0

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------



# COMMAND ----------

df = pd.DataFrame(['item_id', 'timestamp'])

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
train_data.head()

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

print(f"Rows:    {df.count()}")
print(f"Clients: {df.select('client_id').distinct().count()}")
df.selectExpr("min(datetime_local) as min_dt", "max(datetime_local) as max_dt").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC Feel free to add as many cells as you need here.
# MAGIC Use `df.limit(N).toPandas()` if you want to use pandas for exploration.

# COMMAND ----------

# Example: quick look at a small sample in pandas
pdf_sample = df.limit(10000).toPandas()
pdf_sample["datetime_local"] = pd.to_datetime(pdf_sample["datetime_local"])
pdf_sample.describe()

# COMMAND ----------

from pyspark.sql import functions as F

# ---------------------------------------------------------------
# 1a. SAMPLE TO A REPRESENTATIVE WINDOW
# ---------------------------------------------------------------
sample = df.filter(
    (F.col("datetime_local") >= "2025-06-01") &
    (F.col("datetime_local") <  "2025-09-01")
)

feat = (
    sample
    .withColumn("hour",       F.hour("datetime_local"))
    .withColumn("dow",        F.dayofweek("datetime_local"))
    .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
    .select("client_id", "community_code", "active_kw", "hour", "is_weekend")
)

# ---------------------------------------------------------------
# 1b. PER-CLIENT STATS + HOURLY PROFILE IN ONE PASS
# ---------------------------------------------------------------
# Aggregate to (client, is_weekend, hour) first — this collapses 182M rows
# down to ~8k clients × 2 × 24 = ~400k rows. Everything after is cheap.
agg = (
    feat.groupBy("client_id", "community_code", "is_weekend", "hour")
        .agg(
            F.mean("active_kw").alias("avg_kw"),
            F.sum("active_kw").alias("sum_kw"),
            F.sum(F.col("active_kw") * F.col("active_kw")).alias("sumsq_kw"),
            F.max("active_kw").alias("max_kw"),
            F.count(F.lit(1)).alias("n"),
        )
)

# Per-client summary stats — derived from the aggregate (no second raw scan)
client_stats = agg.groupBy("client_id").agg(
    F.first("community_code").alias("community_code"),
    (F.sum("sum_kw") / F.sum("n")).alias("mean_kw"),
    F.sqrt(
        (F.sum("sumsq_kw") / F.sum("n")) -
        F.pow(F.sum("sum_kw") / F.sum("n"), F.lit(2))
    ).alias("std_kw"),
    F.max("max_kw").alias("max_kw"),
)

# p95 needs a separate pass (percentile_approx can't be derived from aggregates),
# but it's on the already-reduced table if we approximate at the hour level.
# Better: compute p95 directly from the raw sample in one go.
p95 = feat.groupBy("client_id").agg(
    F.expr("percentile_approx(active_kw, 0.95, 1000)").alias("p95_kw")
)
client_stats = client_stats.join(p95, on="client_id", how="inner")

# ---------------------------------------------------------------
# 1c. PIVOT THE 400k-row aggregate into 48-column wide profile
# ---------------------------------------------------------------
hourly = agg.withColumn(
    "col_name",
    F.concat(
        F.lit("h"),
        F.lpad(F.col("hour").cast("string"), 2, "0"),
        F.when(F.col("is_weekend") == 1, "_we").otherwise("_wd"),
    ),
)

pivot_cols = [f"h{h:02d}_{s}" for s in ("wd", "we") for h in range(24)]
profile_wide = (
    hourly.groupBy("client_id")
          .pivot("col_name", pivot_cols)
          .agg(F.first("avg_kw"))
          .fillna(0)
)

client_features = client_stats.join(profile_wide, on="client_id", how="inner")

# Pull to pandas — ~8k rows × ~55 cols
pdf = client_features.toPandas()
print(f"Clients: {len(pdf)}, features: {pdf.shape[1]}")

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- NEW: Check for and handle NaNs ---
total_nans = pdf.isna().sum().sum()
print(f"Total missing values found: {total_nans}")

# Drop any clients that have missing values (NaNs) in their features
# This ensures X_scaled and pdf have the exact same number of rows later
pdf_clean = pdf.dropna().copy()
print(f"Clients remaining after dropping NaNs: {len(pdf_clean)}")
# --------------------------------------

# 1. Isolate the features (drop IDs and non-numeric metadata)
features = pdf_clean.drop(columns=['client_id', 'community_code'])

# 2. Scale the data (Crucial for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 3. Test different numbers of clusters (k=2 through 10)
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)  # This will now work perfectly!
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 4. Plot the Elbow Method and Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))

# Elbow Plot
color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow Method)', color=color)
ax1.plot(K_range, inertia, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Silhouette Plot
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color)  
ax2.plot(K_range, silhouette_scores, marker='s', color=color, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Determining Optimal Clusters: Elbow & Silhouette')
fig.tight_layout()  
plt.show()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. SET UP THE CHOSEN CLUSTERS ---
# Assuming pdf_clean and X_scaled are still in your environment from the last step
optimal_k = 4
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
pdf_clean['cluster'] = final_kmeans.fit_predict(X_scaled)

# Set a nice color palette for consistency across charts
cluster_colors = sns.color_palette("Set2", optimal_k)

# Create a figure with 3 subplots
fig = plt.figure(figsize=(18, 12))

# --- DIAGRAM 1: The 2D PCA Scatter Plot (How the clusters separate) ---
ax1 = plt.subplot(2, 2, 1)
# Squash the 50+ features down to 2 dimensions for visual plotting
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)

# Add the 2D coordinates back to our dataframe
pdf_clean['pca_x'] = pca_features[:, 0]
pdf_clean['pca_y'] = pca_features[:, 1]

sns.scatterplot(
    data=pdf_clean, 
    x='pca_x', 
    y='pca_y', 
    hue='cluster', 
    palette=cluster_colors, 
    alpha=0.6, 
    s=30,
    ax=ax1
)
ax1.set_title("1. 2D Map of Client Clusters (PCA)", fontsize=14)
ax1.set_xlabel("Principal Component 1 (Main Variance)")
ax1.set_ylabel("Principal Component 2")

# --- DIAGRAM 2: Community Code Breakdown ---
ax2 = plt.subplot(2, 2, 2)
# Calculate the percentage of each cluster within each community_code
community_cluster_counts = pd.crosstab(
    pdf_clean['community_code'], 
    pdf_clean['cluster'], 
    normalize='index' # Normalizes rows so they add up to 100%
) * 100

# Plot as a stacked bar chart
community_cluster_counts.plot(
    kind='bar', 
    stacked=True, 
    color=cluster_colors, 
    ax=ax2,
    edgecolor='white'
)
ax2.set_title("2. Cluster Distribution by Community Code", fontsize=14)
ax2.set_xlabel("Community Code")
ax2.set_ylabel("Percentage of Clients (%)")
ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- DIAGRAM 3: Boxplot of Average Energy Demand by Cluster ---
ax3 = plt.subplot(2, 1, 2)
sns.boxplot(
    data=pdf_clean, 
    x='cluster', 
    y='mean_kw', 
    palette=cluster_colors, 
    showfliers=False, # Hides extreme outliers so the boxes are readable
    ax=ax3
)
ax3.set_title("3. Average Energy Demand (mean_kw) per Cluster", fontsize=14)
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Average kW")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Your Model
# MAGIC
# MAGIC Iterate on the model here, then copy the final version to the **submission notebook**.
# MAGIC
# MAGIC The baseline below predicts each 15-minute interval using the value from
# MAGIC **7 days ago**, with the historical mean as a fallback.
# MAGIC
# MAGIC ### Important constraints
# MAGIC - During scoring, your `predict()` receives the **full dataset** (up to end of Feb 2026) as a **PySpark DataFrame**.
# MAGIC - `predict_start` and `predict_end` specify the period to generate predictions for.
# MAGIC - Your `predict()` must return a **PySpark DataFrame** with columns `datetime_15min` (timestamp) and `prediction` (double).
# MAGIC - `prediction` must be the **total** (sum) of `active_kw` across all clients for that interval.
# MAGIC - **Avoid `.toPandas()` on the full dataset.** Keep everything in PySpark.
# MAGIC - If using ML libraries (LightGBM, sklearn, etc.), do feature engineering in PySpark first, then
# MAGIC   `.toPandas()` only the **compact feature matrix**. Convert predictions back with `spark.createDataFrame()`.

# COMMAND ----------

class EnergyConsumptionModel:
    """
    Energy consumption forecasting model.

    predict(df, predict_start, predict_end):
        Given data as a PySpark DataFrame, return predictions for every
        15-min interval between predict_start and predict_end.
    """

    def predict(self, df, predict_start, predict_end):
        """
        Args:
            df: PySpark DataFrame with columns client_id, datetime_local,
                         community_code, active_kw.
            predict_start: Start of the prediction period (inclusive), e.g. "2025-12-01".
            predict_end: End of the prediction period (exclusive), e.g. "2026-03-01".

        Must return a PySpark DataFrame with columns:
          - datetime_15min (timestamp): the 15-minute interval
          - prediction (double): total predicted active_kw for that interval
        """
        # Floor timestamps to 15-minute intervals
        df = df.withColumn(
            "datetime_15min",
            F.window("datetime_local", "15 minutes").start
        )

        # Aggregate: total active_kw per 15-min interval (across all clients)
        agg = df.groupBy("datetime_15min").agg(
            F.sum("active_kw").alias("active_kw")
        ).orderBy("datetime_15min")

        # 7-day lag: use lag with window ordered by datetime_15min
        w = Window.orderBy("datetime_15min")
        intervals_per_week = 7 * 96  # 7 days × 96 intervals/day
        agg = agg.withColumn("lag_7d", F.lag("active_kw", intervals_per_week).over(w))

        # Historical mean for fallback (using data before prediction period)
        hist_mean = agg.filter(F.col("datetime_15min") < predict_start).agg(
            F.mean("active_kw")
        ).collect()[0][0]

        # Predict intervals in [predict_start, predict_end)
        result = agg.filter(
            (F.col("datetime_15min") >= predict_start) &
            (F.col("datetime_15min") < predict_end)
        ).select(
            "datetime_15min",
            F.coalesce(F.col("lag_7d"), F.lit(hist_mean)).alias("prediction")
        )

        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local Validation: Train/Test Split
# MAGIC
# MAGIC The training set is everything **before 2025-12-01** and the test set covers
# MAGIC **2025-12-01 to 2026-02-28**.
# MAGIC
# MAGIC **Tip:** Change `val_start` and `val_end` to test on different periods
# MAGIC (e.g. November 2025). This lets you check whether your model
# MAGIC generalises well across time before spending a submission.

# COMMAND ----------

# Train/test split: hold out Dec 2025 – Feb 2026 as a local test set.
# The model receives all data up to end of Feb 2026 (including the test period) for
# feature engineering, but must only return predictions for intervals in [val_start, val_end).
val_start = "2025-12-01"
val_end = "2026-03-01"

model = EnergyConsumptionModel()
val_preds = model.predict(df, predict_start=val_start, predict_end=val_end)

# Actuals for the hold-out period
actuals_val = (
    df.filter(
        (F.col("datetime_local") >= val_start) & (F.col("datetime_local") < val_end)
    )
    .withColumn("datetime_15min", F.window("datetime_local", "15 minutes").start)
    .groupBy("datetime_15min")
    .agg(F.sum("active_kw").alias("active_kw"))
)

merged = val_preds.join(actuals_val, on="datetime_15min", how="inner")
local_mae = merged.select(
    F.mean(F.abs(F.col("active_kw") - F.col("prediction")))
).collect()[0][0]

if local_mae is not None:
    print(f"Local validation MAE ({val_start} to {val_end}): {local_mae:.4f}")
    print(f"Intervals predicted: {merged.count():,}")
else:
    print("WARNING: No predictions matched the hold-out period. Check your model.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC When you are happy with your model's local MAE:
# MAGIC 1. **Copy** your `EnergyConsumptionModel` class (and any imports it needs) to the **submission notebook**.
# MAGIC 2. Run the **Submit** cell there.
