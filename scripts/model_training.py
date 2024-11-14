import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_model_name = None
    best_mse = float("inf")
    best_model_metrics = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Evaluate on training and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Save model
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        # Store metrics
        results[model_name] = {
            "train_mse": train_mse,
            "train_r2": train_r2,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "sample_predictions": y_test_pred[:5].tolist(),
            "model_path": model_path
        }

        # Update the best model metrics
        if test_mse < best_mse:
            best_mse = test_mse
            best_model_name = model_name
            best_model_metrics = {
                "train_mse": train_mse,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "model_path": model_path
            }

    return {
        "model_metrics": results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics
    }
