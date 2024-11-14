from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scripts.data_processing import preprocess_data, preprocess_data_for_inference
from scripts.model_training import train_model
from scripts.model_inference import load_model, make_prediction
import os
import time 

app = FastAPI()

MODEL_DIR = "models"
ENCODER_DIR = "models/encoders"
SCALER_PATH = "models/scaler/scaler.joblib"
PROCESSED_DATA_PATH = "data/processed_data.csv"

# Define data classes for request bodies
class PreprocessRequest(BaseModel):
    data: list
    feature_columns: list
    target_column: str


class TrainRequest(BaseModel):
    data: list
    feature_columns: list
    target_column: str


class InferenceRequest(BaseModel):
    input_data: list
    model_name: str

    class Config:
        protected_namespaces = ()  # Disable protected namespaces warning

@app.post("/preprocess/")
async def preprocess(request: PreprocessRequest):
    start_time = time.time()
    try:
        df = pd.DataFrame(request.data)
        
        X, y, encoders, scaler, preprocessed_data = preprocess_data(
            df,
            request.feature_columns,
            request.target_column,
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH,
            processed_data_path=PROCESSED_DATA_PATH
        )
        
        preprocessed_data.to_csv(PROCESSED_DATA_PATH, index=False)
        elapsed_time = time.time() - start_time
        print(f"Preprocessing time: {elapsed_time:.2f} seconds")  # Log preprocessing time to terminal
        
        return {
            "message": "Data preprocessed successfully",
            "preprocessed_data": preprocessed_data.to_dict(orient="records") 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
async def train(request: TrainRequest):
    start_time = time.time()
    try:
        if os.path.exists(PROCESSED_DATA_PATH):
            df = pd.read_csv(PROCESSED_DATA_PATH)
            X = df[request.feature_columns]
            y = df[request.target_column]
        else:
            df = pd.DataFrame(request.data)
            X, y, _, _, _ = preprocess_data(
                df,
                request.feature_columns,
                request.target_column,
                encoder_dir=ENCODER_DIR,
                scaler_path=SCALER_PATH,
                processed_data_path=PROCESSED_DATA_PATH
            )
        
        results = train_model(X, y, model_dir=MODEL_DIR)
        elapsed_time = time.time() - start_time
        print(f"Training time: {elapsed_time:.2f} seconds")
        
        return {
            "message": "Model(s) trained successfully",
            "model_metrics": results["model_metrics"],
            "best_model_name": results["best_model_name"],
            "best_model_metrics": results["best_model_metrics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(request: InferenceRequest):
    start_time = time.time()
    try:
        model = load_model(MODEL_DIR, request.model_name)
        
        input_df = pd.DataFrame(request.input_data)
        
        X_inference = preprocess_data_for_inference(
            input_df, 
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH
        )
        
        predictions = make_prediction(model, X_inference)
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.2f} seconds") 
        
        return {
            "predictions": predictions.tolist()
        }
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=str(fnf_error))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
