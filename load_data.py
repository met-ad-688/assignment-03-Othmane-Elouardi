# Loading the dataset

# Imports
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pyspark.sql import SparkSession
import re
import numpy as np
import plotly.graph_objects as go
from pyspark.sql.functions import col, split, explode, regexp_replace, transform, when
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id

# Spark session
spark = (
    SparkSession.builder
    .appName("LightcastData")
    .getOrCreate()
)

# Load the CSV
csv_path = "data/lightcast_job_postings.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", "\"")
    .csv(csv_path)
)

# Create a temp view for SQL
df.createOrReplaceTempView("job_postings")

# Quick diagnostics
print("\n--- Spark & Data Quick Check ---")
print("Spark version:", spark.version)
print("Row count (sampled below):")
df.show(5, truncate=False) 
# df.printSchema()          


####################################################################################

# Cleaning the data

# load_data.py
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ------------------------------------------------------------------------------
# Config

CSV_PATH = "data/lightcast_job_postings.csv"  # adjust if your file lives elsewhere
APP_NAME = "LightcastData"

# ------------------------------------------------------------------------------
# Helpers

def median_of(df, colname, rel_error=0.001):
    """Return median using approxQuantile; None if column is empty/null-only."""
    non_null = df.filter(F.col(colname).isNotNull())
    if non_null.head(1):
        vals = non_null.approxQuantile(colname, [0.5], rel_error)
        return float(vals[0]) if vals else None
    return None

# ------------------------------------------------------------------------------
# Main

def main():
    # Spark session
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .getOrCreate()
    )

    # Load CSV (header + schema inference + multiline, handle quotes/escapes)
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .csv(CSV_PATH)
    )

    # Make a temp view for later SQL if needed
    df.createOrReplaceTempView("job_postings")

    # ------------------------------------------------------------------------------
    # 1) Cast salary / experience columns to numeric
   
    numeric_casts = {
        "SALARY_FROM": DoubleType(),
        "SALARY_TO": DoubleType(),
        "SALARY": DoubleType(),
        "MIN_YEARS_EXPERIENCE": DoubleType(),
        "MAX_YEARS_EXPERIENCE": DoubleType(),
    }

    df = df.select(*[
        F.col(c).cast(numeric_casts[c]).alias(c) if c in numeric_casts else F.col(c)
        for c in df.columns
    ])

    # ------------------------------------------------------------------------------
    # 2) Compute medians for imputation (via approxQuantile)
    
    median_from = median_of(df, "SALARY_FROM")
    median_to = median_of(df, "SALARY_TO")
    median_salary = median_of(df, "SALARY")

    # Choose a global fallback for average salary if everything is null
    fallback = next((x for x in [median_salary, median_from, median_to] if x is not None), None)

    # ------------------------------------------------------------------------------
    # 3) Create Average_Salary with sensible fallbacks
    
    df = df.withColumn(
        "Average_Salary",
        F.when(
            F.col("SALARY_FROM").isNotNull() & F.col("SALARY_TO").isNotNull(),
            (F.col("SALARY_FROM") + F.col("SALARY_TO")) / 2.0
        ).when(
            F.col("SALARY").isNotNull(), F.col("SALARY")
        ).when(
            F.col("SALARY_FROM").isNotNull(), F.col("SALARY_FROM")
        ).when(
            F.col("SALARY_TO").isNotNull(), F.col("SALARY_TO")
        ).otherwise(F.lit(fallback))
    )

    # Optionally fill missing bounds with medians (keeps ranges consistent)
    fill_map = {}
    if median_from is not None:
        fill_map["SALARY_FROM"] = median_from
    if median_to is not None:
        fill_map["SALARY_TO"] = median_to
    if fill_map:
        df = df.fillna(fill_map)

    # ------------------------------------------------------------------------------
    # 4) Clean EDUCATION_LEVELS_NAME (remove newlines / carriage returns + trim)
    
    if "EDUCATION_LEVELS_NAME" in df.columns:
        df = df.withColumn(
            "EDUCATION_LEVELS_NAME",
            F.trim(
                F.regexp_replace(
                    F.regexp_replace(F.col("EDUCATION_LEVELS_NAME"), r"\r", " "),
                    r"\n", " "
                )
            )
        )

    # ------------------------------------------------------------------------------
    # 5) Derive REMOTE_GROUP (Remote / Hybrid / Onsite)
    
    if "REMOTE_TYPE_NAME" in df.columns:
        df = df.withColumn(
            "REMOTE_GROUP",
            F.when(F.col("REMOTE_TYPE_NAME") == "Remote", "Remote")
             .when(F.col("REMOTE_TYPE_NAME") == "Hybrid", "Hybrid")
             .otherwise("Onsite")
        )

    # Final view (clean)
    df.createOrReplaceTempView("job_postings_clean")

    # ------------------------------------------------------------------------------
    # Quick diagnostics
   
    print("\n--- Spark & Data Quick Check ---")
    print("Spark version:", spark.version)
    print("Rows retained:", df.count())
    print("Medians  ->  SALARY_FROM:", median_from,
          "| SALARY_TO:", median_to,
          "| SALARY:", median_salary)

    print("\nSample (first 5 rows):")
    cols_to_show = [c for c in [
        "Average_Salary", "SALARY", "SALARY_FROM", "SALARY_TO",
        "EDUCATION_LEVELS_NAME", "REMOTE_TYPE_NAME", "REMOTE_GROUP",
        "MIN_YEARS_EXPERIENCE", "MAX_YEARS_EXPERIENCE"
    ] if c in df.columns]
    df.select(*cols_to_show).show(5, truncate=False)

    # Stop the session when running as a script
    spark.stop()


if __name__ == "__main__":
    main()

##########################################################################################

#Salary Distribution by Industry and Employment Type

#| echo: false
#| warning: false
#| message: false

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

# Start/reuse Spark session
spark = SparkSession.builder.appName("LightcastData").getOrCreate()

# Reuse cleaned table if available; otherwise load CSV
if "job_postings_clean" in [t.name for t in spark.catalog.listTables()]:
    df_use = spark.table("job_postings_clean")
else:
    df_use = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .csv("data/lightcast_job_postings.csv")
        .withColumn("SALARY_FROM", F.col("SALARY_FROM").cast(DoubleType()))
    )

# Filter rows and select columns needed for analysis
plot_sdf = (
    df_use
    .filter((col("SALARY_FROM").isNotNull()) & (col("SALARY_FROM") > 0))
    .select("NAICS2_NAME", "SALARY_FROM", "EMPLOYMENT_TYPE_NAME")
)

# Convert to Pandas for Plotly
plot_df = plot_sdf.toPandas()
plot_df["NAICS2_NAME"] = plot_df["NAICS2_NAME"].astype(str).str.replace(r"\s+", " ", regex=True)


#########################################################################################################

# SAlary Analysis by ONET Occupation Type

#| echo: false
#| warning: false
#| message: false
#| fig-cap: "Salary Analysis by Occupation (Auto-selected) – Bubble Chart"

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
import plotly.express as px
from IPython.display import HTML, display
import re

# ---- Spark / source df
try:
    spark
except NameError:
    spark = SparkSession.builder.appName("LightcastData").getOrCreate()

if any(t.name == "job_postings_clean" for t in spark.catalog.listTables()):
    base = spark.table("job_postings_clean")
else:
    base = (
        spark.read
        .option("header", "true").option("inferSchema", "true")
        .option("multiLine", "true").option("escape", "\"")
        .csv("data/lightcast_job_postings.csv")
    )

# ---- Choose salary column
salary_col = "Average_Salary" if "Average_Salary" in base.columns else (
    "SALARY_FROM" if "SALARY_FROM" in base.columns else "SALARY"
)
df = base.withColumn(salary_col, col(salary_col).cast(DoubleType()))

# ---- Find occupation-like columns and pick the best by distinct count
pattern = re.compile(r"(ONET|OCCUP|OCC|TITLE).*", re.IGNORECASE)
occ_candidates = [c for c in df.columns if pattern.match(c)]
# also include a few common fallbacks explicitly
for c in ["JOB_TITLE_NAME", "JOB_TITLE", "OCCUPATION_NAME", "OCC_TITLE", "ONET_NAME"]:
    if c in df.columns and c not in occ_candidates:
        occ_candidates.append(c)

distinct_counts = []
for c in occ_candidates:
    # count non-empty distinct values quickly
    cnt = (df.where(F.col(c).isNotNull() & (F.length(F.col(c)) > 0))
             .agg(F.approx_count_distinct(c).alias("n")).collect()[0]["n"])
    distinct_counts.append((c, cnt))

distinct_counts.sort(key=lambda x: x[1], reverse=True)
# pick the first with at least 5 distinct values; else fall back to the top one
occ_col, occ_distinct = (distinct_counts[0] if distinct_counts else (None, 0))
for c, n in distinct_counts:
    if n >= 5:
        occ_col, occ_distinct = c, n
        break

if not occ_col or occ_distinct <= 1:
    raise ValueError(
        f"No suitable occupation column with variety was found. "
        f"Checked: {distinct_counts}. Your dataset appears to have a single ONET occupation."
    )

# ---- Clean and explode multi-valued cells (comma/semicolon/pipe)
clean = df.withColumn(occ_col, F.trim(F.regexp_replace(col(occ_col), r"\s+", " ")))
split_col = F.split(F.regexp_replace(col(occ_col), r"\s*[,;/|]\s*", "|"), r"\|")
clean = clean.withColumn("Occupation", F.explode(split_col))
clean = clean.filter(F.col("Occupation").isNotNull() & (F.length(F.col("Occupation")) > 0))

# ---- Aggregate
agg = (
    clean.filter(col(salary_col).isNotNull() & (col(salary_col) > 0))
         .groupBy("Occupation")
         .agg(
             F.percentile_approx(col(salary_col), 0.5, 10000).alias("Median_Salary"),
             F.count(F.lit(1)).alias("Job_Postings")
         )
)

TOP_N = 20
pdf = (agg.orderBy(F.col("Job_Postings").desc()).limit(TOP_N)).toPandas()
pdf = pdf.sort_values("Job_Postings", ascending=False)

# ---- Plot
fig = px.scatter(
    pdf, x="Occupation", y="Median_Salary",
    size="Job_Postings", color="Job_Postings",
    color_continuous_scale="Viridis", size_max=80,
    hover_name="Occupation",
    labels={
        "Occupation": "Occupation Name",
        "Median_Salary": "Median Salary",
        "Job_Postings": "Number of Job Postings"
    },
    title=f"Salary Analysis by Occupation (Auto: {occ_col}, {occ_distinct} distinct)"
)
fig.update_layout(width=1800, height=800, margin=dict(l=40, r=40, t=90, b=320))
fig.update_xaxes(tickangle=45)
display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


##################################################################################################################


# Salary by Education Level


#| echo: false
#| warning: false
#| message: false
#| fig-cap: "Experience vs Salary by Education Level"

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from IPython.display import HTML, display
import plotly.express as px
import numpy as np

# --- Spark / source df
try:
    spark
except NameError:
    spark = SparkSession.builder.appName("LightcastData").getOrCreate()

if any(t.name == "job_postings_clean" for t in spark.catalog.listTables()):
    df = spark.table("job_postings_clean")
else:
    df = (
        spark.read
        .option("header", "true").option("inferSchema", "true")
        .option("multiLine", "true").option("escape", "\"")
        .csv("data/lightcast_job_postings.csv")
    )

# --- Pick experience column robustly
exp_candidates = ["MAX_YEARS_EXPERIENCE", "YEARS_EXPERIENCE",
                  "YEARS_OF_EXPERIENCE", "MIN_YEARS_EXPERIENCE"]
exp_col = next((c for c in exp_candidates if c in df.columns), None)
if exp_col is None:
    raise ValueError(f"No experience column found. Tried: {exp_candidates}")

# --- Pick salary column robustly (prefer your computed Average_Salary)
if "Average_Salary" in df.columns:
    sal_col = "Average_Salary"
else:
    if "SALARY_FROM" in df.columns and "SALARY_TO" in df.columns:
        df = df.withColumn(
            "Average_Salary",
            (F.col("SALARY_FROM").cast(DoubleType()) + F.col("SALARY_TO").cast(DoubleType())) / 2.0
        )
        sal_col = "Average_Salary"
    elif "SALARY" in df.columns:
        sal_col = "SALARY"
    else:
        raise ValueError("No salary column found (looked for Average_Salary, SALARY_FROM/TO, SALARY).")

# Ensure salary is numeric
df = df.withColumn(sal_col, col(sal_col).cast(DoubleType()))

# --- Education grouping
edu_src = "EDUCATION_LEVELS_NAME" if "EDUCATION_LEVELS_NAME" in df.columns else None
if edu_src is None:
    raise ValueError("EDUCATION_LEVELS_NAME column not found.")

df = df.withColumn(
    "EDU_GROUP",
    F.when(F.lower(F.col(edu_src)).contains("phd"), "PhD")
     .when(F.lower(F.col(edu_src)).contains("master"), "Master’s")
     .when(F.lower(F.col(edu_src)).contains("bachelor"), "Bachelor")
     .otherwise("Associate or Lower")
)

# --- Optional: choose a hover occupation/title column if available
occ_candidates = ["OCCUPATION_NAME", "JOB_TITLE_NAME", "JOB_TITLE", "TITLE_RAW",
                  "ONET_NAME", "OCC_TITLE"]
occ_col = next((c for c in occ_candidates if c in df.columns), None)

# --- Build pandas dataframe for Plotly
cols = [exp_col, sal_col, "EDU_GROUP"] + ([occ_col] if occ_col else [])
pdf = (
    df.select(*cols)
      .where(col(sal_col).isNotNull() & (col(sal_col) > 0) & col(exp_col).isNotNull())
      .toPandas()
)

# Jitter experience a bit to reduce overplotting
rng = np.random.default_rng(42)
pdf["exp_jitter"] = pdf[exp_col].astype(float) + rng.uniform(-0.15, 0.15, size=len(pdf))

# Rename for pretty axes
pdf = pdf.rename(columns={sal_col: "Average Salary (USD)",
                          "exp_jitter": "Years of Experience"})

# --- Plot
hover = [occ_col] if occ_col else None
fig = px.scatter(
    pdf,
    x="Years of Experience",
    y="Average Salary (USD)",
    color="EDU_GROUP",
    hover_data=hover,
    opacity=0.7,
    title="Experience vs Salary by Education Level"
)

fig.update_layout(
    xaxis_title="Years of Experience",
    yaxis_title="Average Salary (USD)",
    legend_title="Education Group",
    width=1600, height=800,
    margin=dict(l=40, r=40, t=80, b=120)
)

from IPython.display import HTML
display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))




###################################################################################

# Salary by Remote Work Type


#| echo: false
#| warning: false
#| message: false
#| fig-cap: "Experience vs Salary by Remote Work Type"

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from IPython.display import HTML, display
import plotly.express as px
import numpy as np

# --- Spark / source df
try:
    spark
except NameError:
    spark = SparkSession.builder.appName("LightcastData").getOrCreate()

if any(t.name == "job_postings_clean" for t in spark.catalog.listTables()):
    df = spark.table("job_postings_clean")
else:
    df = (
        spark.read
        .option("header", "true").option("inferSchema", "true")
        .option("multiLine", "true").option("escape", "\"")
        .csv("data/lightcast_job_postings.csv")
    )

# --- Experience column
exp_candidates = ["MAX_YEARS_EXPERIENCE", "YEARS_EXPERIENCE",
                  "YEARS_OF_EXPERIENCE", "MIN_YEARS_EXPERIENCE"]
exp_col = next((c for c in exp_candidates if c in df.columns), None)
if exp_col is None:
    raise ValueError(f"No experience column found. Tried: {exp_candidates}")

# --- Salary column (prefer Average_Salary)
if "Average_Salary" in df.columns:
    sal_col = "Average_Salary"
else:
    if "SALARY_FROM" in df.columns and "SALARY_TO" in df.columns:
        df = df.withColumn(
            "Average_Salary",
            (F.col("SALARY_FROM").cast(DoubleType()) + F.col("SALARY_TO").cast(DoubleType())) / 2.0
        )
        sal_col = "Average_Salary"
    elif "SALARY" in df.columns:
        sal_col = "SALARY"
    else:
        raise ValueError("No salary column found (looked for Average_Salary, SALARY_FROM/TO, SALARY).")

df = df.withColumn(sal_col, col(sal_col).cast(DoubleType()))

# --- Remote grouping
# Use existing REMOTE_GROUP if present; otherwise derive from REMOTE_TYPE_NAME
if "REMOTE_GROUP" in df.columns:
    df = df.withColumn("REMOTE_GROUP",
        F.when(F.col("REMOTE_GROUP").isin("Remote", "Hybrid", "Onsite"), F.col("REMOTE_GROUP"))
         .otherwise("Onsite")  # treat blanks/others as Onsite
    )
elif "REMOTE_TYPE_NAME" in df.columns:
    df = df.withColumn("REMOTE_GROUP",
        F.when(F.col("REMOTE_TYPE_NAME") == "Remote", "Remote")
         .when(F.col("REMOTE_TYPE_NAME") == "Hybrid", "Hybrid")
         .otherwise("Onsite")
    )
else:
    # if no remote info exists, create a single group to avoid failure
    df = df.withColumn("REMOTE_GROUP", F.lit("Onsite"))

# --- Optional hover info
hover_col = None
for c in ["JOB_TITLE_NAME", "TITLE_RAW", "OCCUPATION_NAME", "JOB_TITLE"]:
    if c in df.columns:
        hover_col = c
        break

# --- Build pandas DF for plots
cols = [exp_col, sal_col, "REMOTE_GROUP"] + ([hover_col] if hover_col else [])
pdf = (
    df.select(*cols)
      .where(col(sal_col).isNotNull() & (col(sal_col) > 0) & col(exp_col).isNotNull())
      .toPandas()
)

# jitter experience to reduce overplotting
rng = np.random.default_rng(42)
pdf["exp_jitter"] = pdf[exp_col].astype(float) + rng.uniform(-0.15, 0.15, size=len(pdf))
pdf = pdf.rename(columns={sal_col: "Average Salary (USD)", "exp_jitter": "Years of Experience"})

# --- Scatter plot
hover = [hover_col] if hover_col else None
fig_scatter = px.scatter(
    pdf,
    x="Years of Experience",
    y="Average Salary (USD)",
    color="REMOTE_GROUP",
    hover_data=hover,
    opacity=0.75,
    title="Experience vs Salary by Remote Work Type"
)
fig_scatter.update_layout(
    legend_title="Remote Work Type",
    width=1600, height=800,
    margin=dict(l=40, r=40, t=80, b=120)
)

# --- Histograms of salary by remote group (faceted)
fig_hist = px.histogram(
    pdf,
    x="Average Salary (USD)",
    color="REMOTE_GROUP",
    barmode="overlay",
    opacity=0.6,
    facet_col="REMOTE_GROUP",
    facet_col_spacing=0.06,
    title="Salary Distribution by Remote Work Type"
)
fig_hist.update_layout(
    width=1600, height=500,
    margin=dict(l=40, r=40, t=70, b=60),
    legend_title="Remote Work Type"
)

# --- Render only charts on the page
display(HTML(fig_scatter.to_html(include_plotlyjs="cdn", full_html=False)))
display(HTML(fig_hist.to_html(include_plotlyjs=False, full_html=False)))

