import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data from: https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset?utm_source=chatgpt.com
# Load data
df = pd.read_csv("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/data/ai4i2020.csv")
# notes on data set:
#1) It is synthetic data
#2) Column names are: UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF
#3) TWF: tool wear failur
#4) HDF: heat dissipation failre
#5) PWF: power failure
#6) OSF: overstrain Failure
#6) RNF: random failur
#7) Machine Failure - will be 1 if any of these are equal to 1

# Routine Checks ---------------------------------
# Basic checks for all ML projects
#print the first 5 rows of the dataset
print(df.head())
# give a summary of the dataset
print(df.info())

# Check failure distribution
# This is our target variable, what we want to predict.
# This printed out 0 9661 and 1 339, which tells us there's essentially no failure most of the time.
# Thus, this is what we call a class imbalance problem.
print(df["Machine failure"].value_counts())

# Plot distribution
# want to make sure the data is reasonable
# Chose a variable that is continuou, easy to visualize, and physically interpretable
plt.hist(df["Air temperature [K]"], bins=30)
plt.title("Air Temperature Distribution")
plt.xlabel("Air temperature [K]")
plt.ylabel("Count")
plt.savefig("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/figures/air_temperature_histogram.png")  # <-- saves image
plt.show()

# Boxplot: torque vs failure
# this boxplot will answer if failures are happening at higher torques.
# torque is a nice variable to plot because it is a physically meaningful variable that likely relates to failur.
# answers 'do machines under high load fail more often?'
# ties nicely to stress on the system
df.boxplot(column="Torque [Nm]", by="Machine failure")
# note that df is quick, it's built into pandas already
plt.title("Torque vs Machine Failure")
plt.suptitle("")
plt.xlabel("Machine failure")
plt.ylabel("Torque [Nm]")
plt.savefig("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/figures/Torque_machine_failure_box_plot.png")  # <-- saves image
plt.show()

sns.boxplot(x="Machine failure", y="Torque [Nm]", data=df)
#sns provides clearer syntax and is easier to extend
plt.savefig("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/figures/Torque_machine_failure_box_plot2.png")  # <-- saves image
plt.show()

sns.boxplot(x="Machine failure", y="Air temperature [K]", data=df)
#sns provides clearer syntax and is easier to extend
plt.savefig("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/figures/Air_temp_machine_failure_box_plot.png")  # <-- saves image
plt.show()
