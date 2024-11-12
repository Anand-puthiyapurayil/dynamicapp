# data_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import re
import numpy as np

def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    encoder_dir: str = "models/encoders",
    scaler_path: str = "models/scaler/scaler.joblib",
    processed_data_path: str = "data/processed_data.csv"
):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    scaling numerical features, and saving the encoders, scaler, and preprocessed data to disk.
    """
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Handle missing values
    df = df.dropna(subset=feature_columns + [target_column])

    # Separate features and target
    X = df[feature_columns].copy()  # Make a copy to avoid modifying the original DataFrame
    y = df[target_column].copy()

    # Debugging: Check if X and y are non-empty and contain the correct columns
    if X.empty:
        raise ValueError("The feature DataFrame X is empty after preprocessing.")
    if y.empty:
        raise ValueError("The target column y is empty after preprocessing.")
    if X.shape[1] == 0:
        raise ValueError("No feature columns were found after preprocessing.")

    # Clean numeric columns with currency symbols and commas
    def clean_numeric(value):
        if isinstance(value, str):
            cleaned_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned_value)
            except ValueError:
                return np.nan  # Use NaN for invalid numeric values
        return value

    # Apply cleaning to object columns and target
    for col in X.select_dtypes(include=['object']).columns:
        X.loc[:, col] = X[col].apply(clean_numeric)
    if y.dtype == 'object':
        y = y.apply(clean_numeric)

    # Encode categorical variables and save encoders
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        joblib.dump(le, os.path.join(encoder_dir, f"{col}_encoder.joblib"))

    # Scale numerical features and save the scaler
    numeric_columns = X.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        X.loc[:, numeric_columns] = scaler.fit_transform(X[numeric_columns])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = None  # No numeric columns to scale

    # Combine features and target for display purposes
    preprocessed_data = X.copy()
    preprocessed_data[target_column] = y.values

    # Replace non-JSON-compliant values in the DataFrame before saving
    preprocessed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    preprocessed_data.fillna(0, inplace=True)  # Replace NaN with 0 or another placeholder

    # Save the preprocessed data as a CSV file
    preprocessed_data.to_csv(processed_data_path, index=False)

    return X, y, encoders, scaler, preprocessed_data
