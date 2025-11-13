import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV

mlflow.set_tracking_uri("http://localhost:5000")

data = pd.read_csv('preprocessed_data.csv')
x = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

input_example = x_train[0:5]

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    results = {
        'Confusion Matrix': confusion_matrix(Y_test, Y_pred),
        'Accuracy': accuracy_score(Y_test, Y_pred),
        'Precision': precision_score(Y_test, Y_pred, average='weighted'),
        'Recall': recall_score(Y_test, Y_pred, average='weighted'),
        'F1-Score': f1_score(Y_test, Y_pred, average='weighted')
    }
    return results

with mlflow.start_run():
    param_space ={
        'n_estimators': (50, 200),
        'max_depth': (10, 50)
    }

    model = RandomForestClassifier(random_state=42)
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=32, cv=3, n_jobs=-1, verbose=2, random_state=42)
    rf = bayes_search.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)

    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
    )
    model.fit(x_train, y_train)

    results = evaluate_model(rf, x_test, y_test)

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        if metric in results:
            mlflow.log_metric(metric, float(results[metric]))