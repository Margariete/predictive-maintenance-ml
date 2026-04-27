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

# Drop non-useful columns, like labels
# replace the old dataframe with an updated one
# tells the odel to only learn from meaningful physical variables
df = df.drop(columns=["UDI", "Product ID"])
# Convert categorical machine type into numeric columns
#get_dummies belongs to the pandas library. We pass in the entire data frame, and tell it to convert the type column into numeric form
df = pd.get_dummies(df, columns=["Type"], drop_first=True)


#split the dataset into x|inputs|features and y|outputs|targets
# Define features and target
#this makes a dataset with everything else except the answer
# x is standard for features
X = df.drop(columns=["Machine failure"])
# identifies what we want to predict
#y is standard for targets
y = df["Machine failure"]

# Split data (80% train, 20% test)
# setting a random seed of 42 - this controls the randomness in how the data is split into train vs test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#1. Make model
#note that randome_state here does not need to match random_state above
# random_state here controls how tress are bilt, which features are samples, and how data is bootstrapped
#step 1, create the model -> creat a model object
model = RandomForestClassifier(random_state=42)

#2. Train Model
# now fit is a method of the model object. tells the model to learn the relationship between x and y
model.fit(X_train, y_train)
# inside random forest, wehn you call, model.fit(...), it builds many decision trees
# each tree learns patterns like:
# if torque > 55, and temperature < X, failure is likely
# combines the trees into a forest.
# stores the learned rules inside of model
# after fit, the model is not trained and ready for . predict


#3. Make Predictions
#now, predict is a method of the model object.
# the model object comes with built in abilities (methods, like fit, predict, predict_proba
y_pred = model.predict(X_test)

#4. Evaluate model
# will show 'where the model gets confused'
# will show the true negatives, true positives, false positives and false negatives.
# the matrix will have 4 values in 2 columns and 2 rows. [[correct non-Failures, false alarms/false positives],[missed failures/false negatives, correctly predicted failures]]
# the top left and bottom right values will be correct

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# above this line, I think the model is cheating/has leakage - predictingfailur from the failure metrics TWF, HDF, etc.
# Below this line, let's rop these columns and see how well it performs:


