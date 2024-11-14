import streamlit as st
import pandas as pd
import requests
import numpy as np

API_URL = "http://localhost:8000"

st.title("Machine Learning Application")

# Step 1: Upload and Preview Data
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # Initialize session state for features and target columns if they don't exist
    if "feature_columns" not in st.session_state:
        st.session_state["feature_columns"] = []
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None

    # Step 2: Select Feature Columns and Target Column
    feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=st.session_state["feature_columns"])
    target_column = st.selectbox("Select target column", [col for col in df.columns if col not in feature_columns])

    # Update session state to retain selections
    st.session_state["feature_columns"] = feature_columns
    st.session_state["target_column"] = target_column

    # Step 3: Data Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Step 4: Data Preprocessing
    if st.button("Preprocess Data"):
        data = df.to_dict(orient="records")
        payload = {"data": data, "feature_columns": feature_columns, "target_column": target_column}

        try:
            response = requests.post(f"{API_URL}/preprocess/", json=payload)
            
            if response.status_code == 200:
                st.success("Data preprocessed successfully")
                preprocessed_data = pd.DataFrame(response.json()["preprocessed_data"])

                # Store preprocessed data in session state
                st.session_state["preprocessed_data"] = preprocessed_data
                
                # Display preprocessed data
                st.write("Preprocessed Data:", preprocessed_data)
                
                # Provide a download link for the preprocessed data
                st.download_button(
                    label="Download Preprocessed Data as CSV",
                    data=preprocessed_data.to_csv(index=False),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

    # Display preprocessed data if it exists in session state
    if "preprocessed_data" in st.session_state:
        st.write("Preprocessed Data (from session):", st.session_state["preprocessed_data"])

    # Step 5: Train Model and Display Metrics for Multiple Models
    if st.button("Train Model"):
        if "preprocessed_data" in st.session_state:
            # Use preprocessed data from session state
            preprocessed_data = st.session_state["preprocessed_data"]
            data = preprocessed_data.to_dict(orient="records")
            payload = {"data": data, "feature_columns": feature_columns, "target_column": target_column}

            try:
                response = requests.post(f"{API_URL}/train/", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Model(s) trained successfully")

                    # Display metrics for each model using .get() to handle missing keys gracefully
                    st.write("### Model Metrics")
                    for model_name, metrics in result["model_metrics"].items():
                        st.write(f"**{model_name}**")
                        st.write(f"Train Mean Squared Error (MSE): {metrics.get('train_mse', 'N/A')}")
                        st.write(f"Train R² Score: {metrics.get('train_r2', 'N/A')}")
                        st.write(f"Test Mean Squared Error (MSE): {metrics.get('test_mse', 'N/A')}")
                        st.write(f"Test R² Score: {metrics.get('test_r2', 'N/A')}")
                        st.write(f"Sample Predictions: {metrics.get('sample_predictions', [])}")

                    # Display best model information
                    st.write("### Best Model Summary")
                    st.write(f"**Best Model:** {result['best_model_name']}")
                    st.write(f"Best Model Test MSE: {result['best_model_metrics'].get('test_mse', 'N/A')}")
                    st.write(f"Best Model Test R² Score: {result['best_model_metrics'].get('test_r2', 'N/A')}")

                    # Store available model names for prediction selection
                    st.session_state["model_names"] = list(result["model_metrics"].keys())
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("No preprocessed data found. Please preprocess the data first.")

    # Step 6: Model Inference (Make Prediction)
    st.header("Make Prediction")
    
    # Display model selection dropdown if models are available
    if "model_names" in st.session_state:
        selected_model = st.selectbox("Select model for prediction", st.session_state["model_names"])
        input_data = {feature: st.text_input(f"Enter value for {feature}") for feature in feature_columns}
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            payload = {
                "input_data": input_df.to_dict(orient="records"),
                "model_name": selected_model
            }

            try:
                response = requests.post(f"{API_URL}/predict/", json=payload)
                
                if response.status_code == 200:
                    prediction = response.json()["predictions"]
                    st.write("Predicted value:", prediction[0])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please train models first to enable model selection for predictions.")
