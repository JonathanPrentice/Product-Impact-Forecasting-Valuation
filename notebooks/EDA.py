import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

# Download the dataset from Kaggle
os.system('kaggle datasets download -d vishakhdapat/customer-segmentation-clustering -p ./data')

# Unzip the dataset
with zipfile.ZipFile('./data/customer-segmentation-clustering.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Set the path to the file you'd like to load
file_path = "./data/customer_segmentation.csv"

# Load the data
df = pd.read_csv(file_path)

# Display the first 5 records
print("First 5 records:", df.head())

# Display the last 5 records
print("Last 5 records:", df.tail())

# Step 1: Customer Retention & Churn Analysis

# Define retention groups based on 'Recency' (days since last purchase)
df["Retention_Status"] = pd.cut(
    df["Recency"],
    bins=[-1, 30, 90, 180, 365, float("inf")],
    labels=["Active (0-30d)", "Engaged (31-90d)", "Warm (91-180d)", "Cold (181-365d)", "At Risk (>365d)"]
)

# Average spending per retention group
spending_by_retention = df.groupby("Retention_Status")[
    ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
].mean()

# Customer count per retention group
retention_counts = df["Retention_Status"].value_counts().sort_index()

# Visualization: Retention Group Distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=retention_counts.index, y=retention_counts.values, palette="Blues_r")
plt.title("Customer Retention Segments")
plt.xlabel("Retention Group")
plt.ylabel("Number of Customers")
plt.grid(axis="y")

# Visualization: Spending by Retention Group
spending_by_retention.plot(kind="bar", figsize=(12, 6), title="Average Spending by Retention Group")
plt.ylabel("Avg Spending ($)")
plt.xticks(rotation=45)
plt.grid(axis="y")