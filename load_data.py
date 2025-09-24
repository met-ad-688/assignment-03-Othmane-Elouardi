# === Imports ===
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

# === Spark session ===
spark = (
    SparkSession.builder
    .appName("LightcastData")
    .getOrCreate()
)

# === Load the CSV ===
csv_path = "data/lightcast_job_postings.csv"  # adjust if your file lives elsewhere

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", "\"")
    .csv(csv_path)
)

# Create a temp view for SQL later
df.createOrReplaceTempView("job_postings")

# === Quick diagnostics ===
print("\n--- Spark & Data Quick Check ---")
print("Spark version:", spark.version)
print("Row count (sampled below):")
df.show(5, truncate=False)   # preview only
# df.printSchema()           # uncomment if you want schema
