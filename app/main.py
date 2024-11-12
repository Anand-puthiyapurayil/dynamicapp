from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scripts.data_processing import preprocess_data
from scripts.model_training import train_model
from scripts.model_inference import load_model, make_prediction
import os

app = FastAPI()

MODEL_DIR = "models"  # Directory to store all models
ENCODER_DIR = "models/encoders"
SCALER_PATH = "models/scaler/scaler.joblib"
PROCESSED_DATA_PATH = "data/processed_data.csv"

# Define data classes for request bodies
class PreprocessRequest(BaseModel):
    data: list  # List of dictionaries representing rows of the dataframe
    feature_columns: list
    target_column: str

class TrainRequest(BaseModel):
    data: list  # List of dictionaries representing rows of the dataframe
    feature_columns: list
    target_column: str

class InferenceRequest(BaseModel):
    input_data: list  # List of dictionaries representing rows of the dataframe for prediction
    model_name: str  # Name of the model to use for prediction

    class Config:
        protected_namespaces = ()  # Disable protected namespaces warning

@app.post("/preprocess/")
async def preprocess(request: PreprocessRequest):
    """
    Preprocess data by handling missing values, encoding categorical variables, and scaling.
    """
    try:
        # Convert list of dictionaries to a DataFrame
        df = pd.DataFrame(request.data)
        
        # Run preprocessing
        X, y, encoders, scaler, preprocessed_data = preprocess_data(
            df,
            request.feature_columns,
            request.target_column,
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH,
            processed_data_path=PROCESSED_DATA_PATH
        )
        
        # Return preprocessed data as JSON
        return {
            "message": "Data preprocessed successfully",
            "preprocessed_data": preprocessed_data.to_dict(orient="records")  # Convert DataFrame to JSON serializable format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/")
async def train(request: TrainRequest):
    """
    Train multiple regression models and return metrics for each model.
    """
    try:
        # Convert list of dictionaries to a DataFrame
        df = pd.DataFrame(request.data)
        
        # Preprocess the data for training
        X, y, encoders, scaler, _ = preprocess_data(
            df,
            request.feature_columns,
            request.target_column,
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH,
            processed_data_path=PROCESSED_DATA_PATH
        )
        
        # Train multiple models and save each one
        results = train_model(X, y, model_dir=MODEL_DIR)
        
        # Return training metrics for each model
        return {
            "message": "Model(s) trained successfully",
            "model_metrics": results["model_metrics"],
            "best_model_name": results["best_model_name"],
            "best_model_mse": results["best_model_mse"],
            "best_model_r2": results["best_model_r2"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def predict(request: InferenceRequest):
    """
    Make predictions using a selected trained model.
    """
    try:
        # Load the specified model from the directory
        model = load_model(MODEL_DIR, request.model_name)  # Ensure both model_dir and model_name are provided
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.input_data)
        
        # Make predictions
        predictions = make_prediction(model, input_df)
        
        # Return predictions as JSON
        return {
            "predictions": predictions.tolist()  # Convert numpy array to list for JSON serialization
        }
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=str(fnf_error))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

