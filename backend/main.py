from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_models_pkl", "gradient_boosting_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "preprocessing_objects", "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR, "preprocessing_objects", "label_encoder.pkl"
)

# ...e
# Initialize FastAPI app
app = FastAPI(
    title="Migraine Symptom Classifier API",
    description="API for predicting migraine type based on symptoms.",
)

# Global variables to store the loaded model and preprocessing objects
model = None
scaler = None
label_encoder = None


# Input data model using Pydantic for validation.
# THIS MUST EXACTLY MATCH THE 23 FEATURES YOUR MODEL WAS TRAINED ON, IN ORDER,
# and their expected data types (float for numerical, str for binary inputs from Streamlit).
class MigraineSymptoms(BaseModel):
    Age: float
    Duration: float
    Frequency: float
    Location: str  # Will be "1-sided" or "2-sided"
    Character: str  # Will be "Pulsating" or "Pressing"
    Intensity: float
    Nausea: str  # Will be "No" or "Yes"
    Vomit: str  # Will be "No" or "Yes"
    Phonophobia: str  # Will be "No" or "Yes"
    Photophobia: str  # Will be "No" or "Yes"
    Visual: str  # Will be "No" or "Yes"
    Sensory: str  # Will be "No" or "Yes"
    Dysphasia: str  # Will be "No" or "Yes"
    Dysarthria: str  # Will be "No" or "Yes"
    Vertigo: str  # Will be "No" or "Yes"
    Tinnitus: str  # Will be "No" or "Yes"
    Hypoacusis: str  # Will be "No" or "Yes"
    Diplopia: str  # Will be "No" or "Yes"
    Defect: str  # Will be "No" or "Yes"
    Ataxia: str  # Will be "No" or "Yes"
    Conscience: str  # Will be "No" or "Yes"
    Paresthesia: str  # Will be "No" or "Yes"
    DPF: str  # Will be "No" or "Yes"


# Function to load model and preprocessing objects when the app starts
@app.on_event("startup")
async def startup_event():
    global model, scaler, label_encoder
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from {SCALER_PATH}")

        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"LabelEncoder loaded successfully from {LABEL_ENCODER_PATH}")

    except FileNotFoundError as e:
        print(
            f"ERROR: Failed to load resources. Ensure {MODEL_PATH}, {SCALER_PATH}, and {LABEL_ENCODER_PATH} exist in the correct locations."
        )
        raise RuntimeError(
            f"Failed to load resources: {e}. Did you run 'migraine.py' to save them?"
        )
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during resource loading: {e}")
        raise RuntimeError(f"An error occurred during resource loading: {e}")


# Basic root endpoint to confirm API is running
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Migraine Symptom Classifier API! Use /predict to get predictions."
    }


# Prediction endpoint
@app.post("/predict")
async def predict_migraine_type(symptoms: MigraineSymptoms):
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing objects not loaded. Server might be starting up or encountered an error.",
        )

    try:
        # Helper function to convert binary string inputs to numerical (0.0 or 1.0)
        def map_binary_input_to_numeric(
            value_from_streamlit: str, positive_string_option: str
        ) -> float:
            """Maps a string input (e.g., 'Yes', '2-sided') to 1.0, otherwise 0.0."""
            return 1.0 if value_from_streamlit == positive_string_option else 0.0

        # Convert incoming string inputs to numerical values (0.0 or 1.0).
        # The order here MUST EXACTLY MATCH 'selected_feature_columns' in migraine.py
        processed_input = [
            symptoms.Age,
            symptoms.Duration,
            symptoms.Frequency,
            map_binary_input_to_numeric(
                symptoms.Location, "2-sided"
            ),  # '2-sided' from Streamlit maps to 1.0
            map_binary_input_to_numeric(
                symptoms.Character, "Pressing"
            ),  # 'Pressing' from Streamlit maps to 1.0
            symptoms.Intensity,
            map_binary_input_to_numeric(symptoms.Nausea, "Yes"),
            map_binary_input_to_numeric(symptoms.Vomit, "Yes"),
            map_binary_input_to_numeric(symptoms.Phonophobia, "Yes"),
            map_binary_input_to_numeric(symptoms.Photophobia, "Yes"),
            map_binary_input_to_numeric(symptoms.Visual, "Yes"),
            map_binary_input_to_numeric(symptoms.Sensory, "Yes"),
            map_binary_input_to_numeric(symptoms.Dysphasia, "Yes"),
            map_binary_input_to_numeric(symptoms.Dysarthria, "Yes"),
            map_binary_input_to_numeric(symptoms.Vertigo, "Yes"),
            map_binary_input_to_numeric(symptoms.Tinnitus, "Yes"),
            map_binary_input_to_numeric(symptoms.Hypoacusis, "Yes"),
            map_binary_input_to_numeric(symptoms.Diplopia, "Yes"),
            map_binary_input_to_numeric(symptoms.Defect, "Yes"),
            map_binary_input_to_numeric(symptoms.Ataxia, "Yes"),
            map_binary_input_to_numeric(symptoms.Conscience, "Yes"),
            map_binary_input_to_numeric(symptoms.Paresthesia, "Yes"),
            map_binary_input_to_numeric(symptoms.DPF, "Yes"),
        ]

        input_data_array = np.array(processed_input).reshape(
            1, -1
        )  # Reshape for single sample prediction

        # Apply the same scaling used during training
        scaled_input_data = scaler.transform(input_data_array)

        # Make prediction (numeric label)
        prediction_numeric = model.predict(scaled_input_data)

        # Inverse transform the numeric prediction to get the original string label
        prediction_label = label_encoder.inverse_transform(prediction_numeric)

        return {"predicted_migraine_type": prediction_label[0]}

    except Exception as e:
        print(
            f"Prediction failed due to: {e}"
        )  # Log the detailed error on the server side
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. Check server logs for more details.",
        )


# app.run()
