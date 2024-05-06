import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('web-page-phishing.csv')

df.dropna(inplace=True)

df = df.drop('n_equal', axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

models = [
    LogisticRegression(max_iter=10000),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    KNeighborsClassifier(n_neighbors=3)
]

best_accuracy = 0
best_model = None

for model in models:
    print(f"Training model - {str(model)}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__}: Accuracy = {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"\nBest Model: {best_model.__class__.__name__}, Accuracy = {best_accuracy:.2f}")

# Save the best model as a pickle file
joblib.dump(best_model, 'best_model.pkl')
