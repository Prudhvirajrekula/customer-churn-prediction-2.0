import sqlite3
import pandas as pd
import os

# Create SQLite connection
conn = sqlite3.connect("churn.db")
cursor = conn.cursor()

# Load CSV
df = pd.read_csv("data/customer_churn_dataset-training-master.csv")

# 👇 Normalize column names: lowercase + underscores
df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]

# 👇 Rename specific column to match expected SQL column names
df = df.rename(columns={"customerid": "customer_id"})

# Load into DB
df.to_sql("raw_customers", conn, if_exists="replace", index=False)

# Run SQL files
sql_dir = "sql"
for file in ["churn_flags.sql", "rfm_features.sql", "payment_behavior.sql", "join_features.sql"]:
    with open(os.path.join(sql_dir, file), "r") as f:
        cursor.executescript(f.read())
    print(f"✅ Executed {file}")

# Extract final features
features = pd.read_sql("SELECT * FROM model_features", conn)
features.to_csv("data/model_features.csv", index=False)
print("✅ Features exported to data/model_features.csv")

conn.close()
