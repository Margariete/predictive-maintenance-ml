import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Import our scikit learn - a python library that lets you train ML models and split data into train and test sets
# Also allows yo to evaluate models and preprocess data
# scikitlearn is the main ML tookkit without deep learning
#data splitting
# model_selection: how we split data, test models, tune models
# train_test_split: splits the data
# precisions
# ensemble: this contains the advanced models
# randomForestClassifier: this is the ML model we will use
# evaluation
# metrics: allows us to evaluate the model
# classification report: answers ' how well did we predict failure?'; did we catch all the failures? - evaluates the performance
# confusion matrix: helps us see false alarms and missed failures, shows prediction errors

# Load data
df = pd.read_csv("/data/ai4i2020.csv")

#create a function to train and evaluate:
def train_and_evaluate(df, label):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n=== {label} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

#model with leakage
df_leak = df.copy()

df_leak = df_leak.drop(columns=["UDI", "Product ID"])
df_leak = pd.get_dummies(df_leak, columns=["Type"], drop_first=True)

train_and_evaluate(df_leak, "WITH LEAKAGE")

#model without leakage
df_clean = df.copy()

df_clean = df_clean.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
df_clean = pd.get_dummies(df_clean, columns=["Type"], drop_first=True)

train_and_evaluate(df_clean, "WITHOUT LEAKAGE")


# Original results from this show that this del rarely raises false alarms, but misses many failures.
# Seems that we catch only 34% of failures. :

# the classification report shows:
#   precision   recall    f1-score    support
# 0  0.98        1.00         0.99     1939
# 1  0.81        0.34         0.48     61

#where class 0 is no machine failure and is slightly less critical.
#class 1 is machine failure and is most important.
# precision = TP/(TP+FP)
# when the model predicts machine failure, how often is it right?

# recall = TP/(TP+FN)
# out of all the real machine failures, how many did the model catch?

#F1 = 2* (precision x recall) / (precision + recall)
#if both recall and precision are good, this will be high

# support = number of actual samples in the non-machinefailure and the machine failure classes.



#    accuracy                           0.98      2000
#   macro avg       0.89      0.67      0.74      2000
#weighted avg       0.97      0.98      0.97      2000

# Macro avg is a simple average across the two 0 and 1 classes.
# macro average of precision is (precision_0 + precision 1)/2
#Micro average is weighted by frequency.
# Weighted avg for f-Score = (F1₀ × support₀ + F1₁ × support₁) / total
# = (0.99 × 1939 + 0.48 × 61) / 2000