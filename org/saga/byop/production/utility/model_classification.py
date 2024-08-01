from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load dataset
df = pd.read_csv('C:\\Users\\Rinki\\Downloads\\historical_data.csv')
# Split data into features and target
X = df.drop(columns=['Fraud'])
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, te
# Split data into train and test setsst_size=0.3, random_state=42)

# Define models and parameter grids for hyperparameter tuning
models = [
    ('SVM', SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
]

# Train and evaluate models
for name, model, param_grid in models:
    print(f"Training {name} model...")

    # Construct pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Evaluate on test set
    print(f"Evaluating {name} model...")
    y_pred = grid_search.predict(X_test)
    print(f"Classification report for {name} model:")
    print(classification_report(y_test, y_pred))

    # Print best hyperparameters
    print(f"Best hyperparameters for {name} model: {grid_search.best_params_}")
    print("")

