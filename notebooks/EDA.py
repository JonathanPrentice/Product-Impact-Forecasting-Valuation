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