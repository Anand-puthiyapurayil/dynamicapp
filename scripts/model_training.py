import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y, model_dir="models"):
    """
    Train multiple regression models and save each one to disk. 
    
    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable.
    - model_dir (str): Directory to save all models.
    
    Returns:
    - dict: Metrics for each model and paths to the saved models.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models to compare
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    best_model = None
    best_mse = float("inf")
    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the model to disk
        model_file_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_file_path)

        # Store metrics and path for each model
        results[model_name] = {
            "mse": mse,
            "r2": r2,
            "model_path": model_file_path
        }

        # Update best model based on MSE
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = model_name

    # Return metrics and paths for all models
    return {
        "model_metrics": results,
        "best_model_name": best_model_name,
        "best_model_mse": best_mse,
        "best_model_r2": r2
    }
