import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# import data
df = pd.read_csv('C:\\Users\\Rinki\\Downloads\\historical_data.csv')
df.head()
# check dataset
df.info()
# taking numerical data
df_numerical = df.select_dtypes(exclude="object")
df_numerical.head()
# analyis numerical data
df_numerical.info()
# check null value
df_numerical.isnull().sum()
# drop null
df_numerical.dropna(inplace=True)


df = df.drop(columns = [
    'InsurerNotes',
    'LossDate',
    'FirstPolicySubscriptionDate',
    'ReferenceId',
'FirstPartyVehicleNumber','ThirdPartyVehicleNumber'
    ])

dummies = pd.get_dummies(df[[
    'PolicyholderOccupation',
    'ClaimCause',
    'ClaimInvolvedCovers',
    'DamageImportance',
    'PolicyHolderPostCode',
    'ConnectionBetweenParties',
    'LossPostCode']])




X = df.drop(columns=['Fraud'])
y = df['Fraud']

print(df.select_dtypes(include=['object']).columns  )

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

# Define the parameter grid for each model
param_grid = [
    {
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    {
        'clf': [DecisionTreeClassifier()],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    {
        'clf': [SVC()],
        'clf__C': [0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 'auto']
    }
]

# Create a pipeline with a scaler and a placeholder for the classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the data
    ('clf', RandomForestClassifier())  # Step 2: Placeholder for the classifier
])

# Setup the grid search with all parameter grids
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Train the model with grid search
grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))




