import joblib
import pandas as pd
import os

def load_model(model_dir, model_name):
    """
    Load a specified trained model from disk.
    
    Parameters:
    - model_dir (str): Directory where models are stored.
    - model_name (str): Name of the model to load (without file extension).
    
    Returns:
    - model: Loaded model object.
    """
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = joblib.load(model_path)
    return model

def make_prediction(model, input_data: pd.DataFrame):
    """
    Make predictions using the loaded model.
    
    Parameters:
    - model: Trained model object.
    - input_data (pd.DataFrame): Data for prediction.
    
    Returns:
    - list: Model predictions.
    """
    predictions = model.predict(input_data)
    return predictions
