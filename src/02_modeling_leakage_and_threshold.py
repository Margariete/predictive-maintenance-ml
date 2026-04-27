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
df = pd.read_csv("/Users/margarietemalenda/personal_projects/predictive-maintenance-ml/data/ai4i2020.csv")

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
    # Get predicted probabilities for class 1 = failure
    # this is the estimated probability of failure for each test row
    y_probs = model.predict_proba(X_test)[:, 1]
    # change this to:     y_probs = model.predict_proba(X_test)[:, 0] to adjust probability for class 0
    # then you would adjust the line below to:    y_pred_threshold = (y_probs < threshold).astype(int)
    # but note we almost always wanto to threshold the class related to the event of interest (class1, failure)

    # Lower threshold to catch more failures
    # as threshold lowers, the recall for class 1 will improve, but precision will drop
    threshold = 0.30
    #this turns probabilities into predictions
    y_pred_threshold = (y_probs >= threshold).astype(int)

    y_pred = model.predict(X_test)


    print("\nDefault model predictions")
    print("Threshold: 0.50")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nAdjusted threshold predictions")
    print(f"Threshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_threshold))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_threshold))
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



