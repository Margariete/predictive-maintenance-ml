import pandas as pd
import matplotlib.pyplot as plt

# data from: https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset?utm_source=chatgpt.com
# Load data
df = pd.read_csv("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/data/ai4i2020.csv")

# Basic checks
print(df.head())
print(df.info())

# Check failure distribution
print(df["Machine failure"].value_counts())

# Plot distribution
plt.hist(df["Air temperature [K]"], bins=30)
plt.title("Air Temperature Distribution")
plt.xlabel("Air temperature [K]")
plt.ylabel("Count")
plt.show()

# Boxplot: torque vs failure
df.boxplot(column="Torque [Nm]", by="Machine failure")
plt.title("Torque vs Machine Failure")
plt.suptitle("")
plt.xlabel("Machine failure")
plt.ylabel("Torque [Nm]")
plt.show()