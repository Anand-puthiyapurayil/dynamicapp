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
    processed_data_path: str = "data/processed_data.csv",
    dtype_path: str = "models/feature_dtypes.joblib",
    feature_columns_path: str = "models/feature_columns.joblib",
    numeric_columns_path: str = "models/numeric_columns.joblib"
):
    # Create necessary directories
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Handle missing values in feature and target columns
    df = df.dropna(subset=feature_columns + [target_column])

    # Separate features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Save data types of feature columns
    feature_dtypes = X.dtypes.to_dict()
    joblib.dump(feature_dtypes, dtype_path)

    # Identify categorical and numeric columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    joblib.dump(list(categorical_columns), "models/categorical_columns.joblib")

    numeric_columns = X.select_dtypes(include=['number']).columns
    joblib.dump(list(numeric_columns), numeric_columns_path)

    print(f"Categorical columns to encode: {list(categorical_columns)}")
    print(f"Numeric columns to scale: {list(numeric_columns)}")

    # Clean numeric columns by removing non-numeric characters
    def clean_numeric(value):
        if isinstance(value, str):
            cleaned_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned_value)
            except ValueError:
                return np.nan
        return value

    for col in numeric_columns:
        X[col] = X[col].apply(clean_numeric)
        X[col] = pd.to_numeric(X[col], errors='coerce')

    y = y.apply(clean_numeric)
    y = pd.to_numeric(y, errors='coerce')

    if y.isna().any():
        X = X[~y.isna()]
        y = y.dropna()

    # Encode categorical variables and save encoders
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        joblib.dump(le, os.path.join(encoder_dir, f"{col}_encoder.joblib"))
        print(f"Encoded column: {col}")

    # Initialize scaler as None
    scaler = None
    # Conditionally apply scaler if there are numeric columns
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        joblib.dump(scaler, scaler_path)
        print(f"Scaled columns: {list(numeric_columns)}")
    else:
        print("No numeric columns to scale; skipping scaling.")

    final_feature_columns = X.columns.tolist()
    joblib.dump(final_feature_columns, feature_columns_path)
    print(f"Final feature columns after preprocessing: {final_feature_columns}")

    # Save preprocessed data
    preprocessed_data = X.copy()
    preprocessed_data[target_column] = y.values
    preprocessed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    preprocessed_data.fillna(0, inplace=True)
    preprocessed_data.to_csv(processed_data_path, index=False)

    print("Training data after preprocessing:\n", preprocessed_data.head())
    print("Training target variable distribution:\n", y.describe())

    return X, y, encoders, scaler, preprocessed_data

def preprocess_data_for_inference(input_df, encoder_dir, scaler_path, dtype_path="models/feature_dtypes.joblib", feature_columns_path="models/feature_columns.joblib", numeric_columns_path="models/numeric_columns.joblib", categorical_columns_path="models/categorical_columns.joblib"):
    # Load feature columns
    if os.path.exists(feature_columns_path):
        final_feature_columns = joblib.load(feature_columns_path)
        input_df = input_df.reindex(columns=final_feature_columns, fill_value=0)
    else:
        raise FileNotFoundError("Feature columns file not found.")

    # Load and enforce feature data types
    if os.path.exists(dtype_path):
        feature_dtypes = joblib.load(dtype_path)
        for col, dtype in feature_dtypes.items():
            input_df[col] = input_df[col].astype(dtype, errors='ignore')
    else:
        raise FileNotFoundError("Feature data types file not found.")

    # Load categorical columns
    if os.path.exists(categorical_columns_path):
        categorical_columns = joblib.load(categorical_columns_path)
        input_df[categorical_columns] = input_df[categorical_columns].astype(str)
    else:
        raise FileNotFoundError("Categorical columns file not found.")

    # Load numeric columns
    if os.path.exists(numeric_columns_path):
        numeric_columns = joblib.load(numeric_columns_path)
    else:
        raise FileNotFoundError("Numeric columns file not found.")

    # Clean numeric columns for inference
    for column in numeric_columns:
        if column in input_df.columns and input_df[column].dtype == 'object':
            input_df[column] = input_df[column].replace(r'[₹$,]', '', regex=True).astype(float, errors='ignore')
            input_df[column] = pd.to_numeric(input_df[column], errors='coerce')

    # Apply encoding for categorical columns
    for column in categorical_columns:
        encoder_path = os.path.join(encoder_dir, f"{column}_encoder.joblib")
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            input_df[column] = input_df[column].fillna("unknown")  # Replace NaNs with "unknown"
            input_df[column] = input_df[column].apply(lambda x: x if x in encoder.classes_ else "unknown")
            input_df[column] = encoder.transform(input_df[column])
        else:
            print(f"Warning: Encoder for column '{column}' not found.")

    # Apply scaling to numeric columns
    if len(numeric_columns) > 0 and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    print("Final preprocessed inference data:\n", input_df.head())
    return input_df
