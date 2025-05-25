# src/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # The target column is 'R' (last column)
    target_col = 'R'
    print(f"Dataset shape: {df.shape}")
    print(f"Target values: {df[target_col].value_counts().to_dict()}")
    
    # Map R=0 (Rock), M=1 (Mine)
    df[target_col] = df[target_col].map({'R': 0, 'M': 1})
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def predict_sample(model, sample):
    prediction = model.predict(sample.reshape(1, -1))
    return "Rock" if prediction[0] == 0 else "Mine"