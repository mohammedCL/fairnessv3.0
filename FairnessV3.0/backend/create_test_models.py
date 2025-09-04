#!/usr/bin/env python3
"""
Script to create proper test models for fairness assessment
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_test_models():
    """Create and save test models for fairness assessment using multiple datasets"""

    # Define dataset prefixes
    datasets = [
        "hiring_biased",
        "hiring_unbiased",
        "loan_biased",
        "loan_unbiased"
    ]

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Load the training data
        train_data_path = f"../fairness_test_data/{dataset}_train.csv"
        if not os.path.exists(train_data_path):
            print(f"Training data not found at {train_data_path}")
            continue

        df = pd.read_csv(train_data_path)
        print(f"Loaded training data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Assume the last column is the target
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        # Handle missing values
        X = X.fillna(X.median())

        print(f"Target column: {target_column}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Target distribution: {y.value_counts()}")

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models to create
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=10, random_state=42),
            'naive_bayes': GaussianNB(),
            'svm': SVC(probability=True, random_state=42),
        }

        # Create and save models
        for model_name, model in models.items():
            print(f"\nTraining {model_name} for dataset {dataset}...")

            # Handle SVM scaling
            if model_name == 'svm':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)

                # Save scaler
                scaler_path = f"../fairness_test_data/{dataset}_{model_name}_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"Saved scaler to {scaler_path}")
            else:
                model.fit(X_train, y_train)

            # Test the model
            if model_name == 'svm':
                score = model.score(X_test_scaled, y_test)
            else:
                score = model.score(X_test, y_test)
            print(f"Model accuracy: {score:.3f}")

            # Save model
            model_path = f"../fairness_test_data/{dataset}_{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model to {model_path}")

            # Verify the saved model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
                print(f"Verified: Model type = {type(loaded_model)}")
                print(f"Verified: Has predict = {hasattr(loaded_model, 'predict')}")
                print(f"Verified: Has predict_proba = {hasattr(loaded_model, 'predict_proba')}")
                print(f"Verified: Has score = {hasattr(loaded_model, 'score')}\n")

if __name__ == "__main__":
    create_test_models()
