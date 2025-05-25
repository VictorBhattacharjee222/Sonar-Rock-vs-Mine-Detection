# main.py
from src.model import load_data, train_model, evaluate_model, predict_sample
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    # Load data
    X, y = load_data("data/sonar.csv")
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Predict on one sample
    sample = X_test.iloc[0].values
    prediction = predict_sample(model, sample)
    print(f"Prediction for sample: {prediction}")