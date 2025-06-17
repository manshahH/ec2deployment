import streamlit as st
import requests
import json
import os  # <-- Add this import statement

# Define the FastAPI backend URL using an environment variable
# In Docker Compose, we'll set the 'FASTAPI_BACKEND_URL' env var.
# If not set (e.g., during local development without Docker Compose),
# it will default to 'http://127.0.0.1:8000/predict'.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")


# ... rest of your app.py code ...iguration for a nicer look
st.set_page_config(
    page_title="Migraine Symptom Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🧠 Migraine Symptom Classifier")
st.markdown(
    "Please provide details about the migraine symptoms to get a predicted type."
)


# Helper function to map 'Yes'/'No' (or other binary strings) to 1.0/0.0
# This is mainly for consistent display in the input_data dictionary,
# but the backend will do the actual numerical conversion.
def map_radio_to_string(value: str) -> str:
    """Returns the string value directly from st.radio."""
    return value


# Input fields for symptoms using Streamlit forms
with st.form("migraine_prediction_form"):
    st.header("Patient & Symptom Details")
    col1, col2, col3 = st.columns(3)  # Use columns for a more organized layout

    with col1:
        st.subheader("General & Pain Characteristics")
        age = st.number_input(
            "Age (Years)",
            min_value=0.0,
            max_value=120.0,
            value=30.0,
            help="Enter the patient's age (e.g., 30)",
        )
        duration = st.number_input(
            "Duration (Hours)",
            min_value=0.0,
            value=4.0,
            help="How long does the migraine attack typically last? (e.g., 4.5)",
        )
        frequency = st.number_input(
            "Frequency (Per Month)",
            min_value=0.0,
            value=2.0,
            help="How many attacks per month? (e.g., 2)",
        )
        intensity = st.number_input(
            "Intensity (1-10)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            help="Pain intensity on a scale of 1-10 (e.g., 7 for severe)",
        )

        st.markdown("---")
        location = st.radio(
            "Pain Location",
            ["1-sided", "2-sided"],
            index=0,
            help="Is the pain typically on one side or both sides of the head?",
        )
        character = st.radio(
            "Pain Character",
            ["Pulsating", "Pressing"],
            index=0,
            help="Is the pain pulsating/throbbing or more of a constant, pressing sensation?",
        )

    with col2:
        st.subheader("Associated Symptoms")
        nausea = st.radio(
            "Nausea",
            ["No", "Yes"],
            index=0,
            help="Do you experience nausea during an attack?",
        )
        vomit = st.radio(
            "Vomiting",
            ["No", "Yes"],
            index=0,
            help="Do you experience vomiting during an attack?",
        )
        phonophobia = st.radio(
            "Phonophobia (Sound Sensitivity)",
            ["No", "Yes"],
            index=0,
            help="Are you sensitive to sound during an attack?",
        )
        photophobia = st.radio(
            "Photophobia (Light Sensitivity)",
            ["No", "Yes"],
            index=0,
            help="Are you sensitive to light during an attack?",
        )

        st.markdown("---")
        st.subheader("Aura Symptoms")
        visual = st.radio(
            "Visual Disturbances (Aura)",
            ["No", "Yes"],
            index=0,
            help="Do you have visual disturbances (e.g., zig-zags, flashing lights) before or during a migraine?",
        )
        sensory = st.radio(
            "Sensory Disturbances (Aura)",
            ["No", "Yes"],
            index=0,
            help="Do you have sensory disturbances (e.g., numbness, tingling) before or during a migraine?",
        )
        dysphasia = st.radio(
            "Dysphasia (Speech Difficulty)",
            ["No", "Yes"],
            index=0,
            help="Do you experience difficulty speaking or finding words?",
        )
        dysarthria = st.radio(
            "Dysarthria (Articulation Difficulty)",
            ["No", "Yes"],
            index=0,
            help="Do you experience difficulty articulating words clearly?",
        )

    with col3:
        st.subheader("Other Neurological Symptoms & Triggers")
        vertigo = st.radio(
            "Vertigo (Dizziness)",
            ["No", "Yes"],
            index=0,
            help="Do you experience dizziness or a spinning sensation?",
        )
        tinnitus = st.radio(
            "Tinnitus (Ringing Ears)",
            ["No", "Yes"],
            index=0,
            help="Do you hear ringing or buzzing in your ears?",
        )
        hypoacusis = st.radio(
            "Hypoacusis (Hearing Loss)",
            ["No", "Yes"],
            index=0,
            help="Do you experience decreased hearing?",
        )
        diplopia = st.radio(
            "Diplopia (Double Vision)",
            ["No", "Yes"],
            index=0,
            help="Do you experience double vision?",
        )
        defect = st.radio(
            "Visual Field Defect",
            ["No", "Yes"],
            index=0,
            help="Do you have blind spots or a partial loss of vision?",
        )
        ataxia = st.radio(
            "Ataxia (Coordination Loss)",
            ["No", "Yes"],
            index=0,
            help="Do you experience loss of muscle coordination?",
        )
        conscience = st.radio(
            "Conscience Impairment",
            ["No", "Yes"],
            index=0,
            help="Do you experience impaired consciousness (e.g., confusion, fainting)?",
        )
        paresthesia = st.radio(
            "Paresthesia (Numbness/Tingling)",
            ["No", "Yes"],
            index=0,
            help="Do you experience unusual sensations like numbness or tingling?",
        )
        dpf = st.radio(
            "DPF (Definite Precipitating Factor)",
            ["No", "Yes"],
            index=0,
            help="Is there a specific, definite factor that triggers your migraines?",
        )

    submitted = st.form_submit_button("Predict Migraine Type")

    if submitted:
        # Collect all input data into a dictionary, sending the raw string values
        # The backend will handle mapping these strings to 0.0/1.0
        input_data = {
            "Age": age,
            "Duration": duration,
            "Frequency": frequency,
            "Location": map_radio_to_string(location),
            "Character": map_radio_to_string(character),
            "Intensity": intensity,
            "Nausea": map_radio_to_string(nausea),
            "Vomit": map_radio_to_string(vomit),
            "Phonophobia": map_radio_to_string(phonophobia),
            "Photophobia": map_radio_to_string(photophobia),
            "Visual": map_radio_to_string(visual),
            "Sensory": map_radio_to_string(sensory),
            "Dysphasia": map_radio_to_string(dysphasia),
            "Dysarthria": map_radio_to_string(dysarthria),
            "Vertigo": map_radio_to_string(vertigo),
            "Tinnitus": map_radio_to_string(tinnitus),
            "Hypoacusis": map_radio_to_string(hypoacusis),
            "Diplopia": map_radio_to_string(diplopia),
            "Defect": map_radio_to_string(defect),
            "Ataxia": map_radio_to_string(ataxia),
            "Conscience": map_radio_to_string(conscience),
            "Paresthesia": map_radio_to_string(paresthesia),
            "DPF": map_radio_to_string(dpf),
        }

        try:
            # Send the data to the FastAPI backend
            response = requests.post(FASTAPI_URL, json=input_data)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            prediction_result = response.json()
            predicted_type = prediction_result.get("predicted_migraine_type")

            if predicted_type:
                st.success(f"**Predicted Migraine Type:** {predicted_type}")
            else:
                st.error(
                    "Prediction failed: Could not retrieve a predicted type from the backend response."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. Please ensure the backend server is running at the correct URL."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the API request: {e}.")
            if response is not None:
                st.error(
                    f"Backend response: {response.text}"
                )  # Show backend's error message if available
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Machine Learning model (specifically, the Gradient Boosting Classifier "
    "from your 'trained_models_pkl' folder) to predict migraine types based on symptom inputs. "
    "The backend is powered by FastAPI and the frontend by Streamlit."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Input Guide:**")
st.sidebar.markdown(
    "- Numerical inputs (e.g., Age, Duration, Frequency, Intensity) use number fields."
)
st.sidebar.markdown(
    "- Binary inputs (Yes/No, 1-sided/2-sided, Pulsating/Pressing) use intuitive radio buttons."
)
st.sidebar.markdown(
    "  - 'Yes' and '2-sided' and 'Pressing' map to a numerical `1` for the model."
)
st.sidebar.markdown(
    "  - 'No' and '1-sided' and 'Pulsating' map to a numerical `0` for the model."
)
