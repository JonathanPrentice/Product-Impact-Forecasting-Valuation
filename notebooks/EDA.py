import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
file_path = "/kaggle/input/customer-segmentation-clustering/customer_segmentation.csv"  #
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataset
df.head()

# Display the last 5 rows of the dataset
df.tail()

# Display the shape of the dataset
df.shape