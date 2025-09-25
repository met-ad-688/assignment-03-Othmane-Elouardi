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


