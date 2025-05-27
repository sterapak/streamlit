import streamlit as st
import requests
import json
import os
from openai import OpenAI

# --- Load environment variables ---
AZURE_ENDPOINT_URI = os.getenv("AZURE_ENDPOINT_URI")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration Constants ---
AZURE_ML_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
    "color", "qual_bool"
]
OPENAI_MODEL_NAME = "gpt-4" # Or "gpt-4o" or other preferred model

# Validate required environment variables
if not all([AZURE_ENDPOINT_URI, AZURE_API_KEY, OPENAI_API_KEY]):
    st.error("⚠️ Missing required environment variables. Please set AZURE_ENDPOINT_URI, AZURE_API_KEY, and OPENAI_API_KEY.")
    st.stop()

# --- API Client Initialization ---
@st.cache_resource
def get_openai_client():
    """Initializes and returns an OpenAI client instance."""
    return OpenAI(api_key=OPENAI_API_KEY)

openai_client = get_openai_client()

st.title("🍷 AI Wine Quality Assistant")

# --- User Interface Elements ---
fixed_acidity = st.slider("Fixed Acidity", 4.0, 15.0, 7.4)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 1.9)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.076)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 11.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 34.0)
density = st.slider("Density", 0.9900, 1.0050, 0.9978)
ph = st.slider("pH", 2.8, 4.0, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)
color = st.selectbox("Wine Color", ["red", "white"])
qual_bool = st.selectbox("Is it quality wine (qual_bool)?", [True, False])

# --- Helper function for data payload ---
def get_input_data_payload(
    fixed_acidity_val, volatile_acidity_val, citric_acid_val, residual_sugar_val, chlorides_val,
    free_sulfur_dioxide_val, total_sulfur_dioxide_val, density_val, ph_val, sulphates_val,
    alcohol_val, color_val, qual_bool_val
):
    """Constructs the input data payload for the Azure ML endpoint."""
    return {
        "columns": AZURE_ML_COLUMNS,
        "data": [[
            fixed_acidity_val, volatile_acidity_val, citric_acid_val, residual_sugar_val,
            chlorides_val, free_sulfur_dioxide_val, total_sulfur_dioxide_val, density_val,
            ph_val, sulphates_val, alcohol_val, color_val, qual_bool_val
        ]]
    }

# --- Helper function for OpenAI prompt ---
def construct_openai_prompt(
    quality_score: float,
    color: str,
    alcohol: float,
    ph: float,
    residual_sugar: float,
    fixed_acidity: float,
    volatile_acidity: float,
    sulphates: float
) -> str:
    """Constructs the prompt for the OpenAI API based on wine characteristics."""
    return f"""
    A {color} wine has a predicted quality score of {quality_score:.1f}.
    Characteristics:
    - Alcohol: {alcohol}%
    - pH: {ph}
    - Residual sugar: {residual_sugar}g/L
    - Fixed acidity: {fixed_acidity}
    - Volatile acidity: {volatile_acidity}
    - Sulphates: {sulphates}

    Write a 2-sentence tasting description and suggest a food pairing.
    """

# --- Cached API Call Functions ---
@st.cache_data
def fetch_prediction_from_azure(input_data_json_str: str) -> tuple[list | None, str | None]:
    """
    Calls the Azure ML endpoint to get a wine quality prediction.
    Returns (prediction_result, error_message).
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_API_KEY}"
        }
        response = requests.post(AZURE_ENDPOINT_URI, headers=headers, data=input_data_json_str)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json(), None
    except requests.exceptions.HTTPError as http_err:
        return None, f"Azure ML call failed ({http_err.response.status_code}): {http_err.response.text}"
    except Exception as e:
        return None, f"Error calling Azure ML: {e}"

@st.cache_data
def generate_description_with_openai(prompt: str) -> tuple[str | None, str | None]:
    """
    Generates a tasting note using the OpenAI API.
    Returns (tasting_note, error_message).
    """
    if not openai_client: # Access global client
        return None, "OpenAI client not available. Check OPENAI_API_KEY."
    try:
        gpt_response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200 # Increased slightly for potentially longer food pairings
        )
        return gpt_response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, f"Error calling OpenAI: {e}"

# --- Main UI Logic Function ---
def display_results(
    azure_payload: dict,
    prompt_parameters: dict
):
    """Handles the prediction, GPT note generation, and Streamlit UI updates."""
    st.write("🧪 Raw input data for Azure ML:", azure_payload) # For transparency

    input_data_json = json.dumps({"input_data": azure_payload})
    prediction_result, error_msg_azure = fetch_prediction_from_azure(input_data_json)

    if error_msg_azure:
        st.error(f"❌ {error_msg_azure}")
        return

    if prediction_result is None: # Should ideally be caught by error_msg_azure
        st.error("❌ Azure ML call did not return a valid result (prediction_result is None).")
        return

    st.write("✅ Raw model response from Azure ML:", prediction_result)
    
    quality = None
    if isinstance(prediction_result, list) and len(prediction_result) > 0:
        if isinstance(prediction_result[0], (int, float)):
            quality = float(prediction_result[0])
        else:
            st.warning(f"⚠️ Prediction value from Azure ML is not a number: {prediction_result[0]}")
    elif isinstance(prediction_result, list) and len(prediction_result) == 0:
        st.warning("⚠️ Azure ML returned an empty list for prediction.")
    else:
        st.warning(f"⚠️ Unexpected prediction result format from Azure ML: {prediction_result}")

    if quality is not None:
        st.success(f"Predicted Wine Quality Score: {quality:.2f}")
        
        # Construct prompt using the dedicated function and prompt_parameters
        prompt_text = construct_openai_prompt(
            quality_score=quality,
            color=prompt_parameters["color"],
            alcohol=prompt_parameters["alcohol"],
            ph=prompt_parameters["ph"],
            residual_sugar=prompt_parameters["residual_sugar"],
            fixed_acidity=prompt_parameters["fixed_acidity"],
            volatile_acidity=prompt_parameters["volatile_acidity"],
            sulphates=prompt_parameters["sulphates"]
        )
        
        tasting_note, error_msg_openai = generate_description_with_openai(prompt_text)

        if error_msg_openai:
            st.error(f"⚠️ OpenAI Error: {error_msg_openai}")
        elif tasting_note:
            st.markdown("### 🍷 GPT-Generated Tasting Note")
            st.write(tasting_note)
        else:
            st.warning("⚠️ GPT could not generate a tasting note (no specific error provided).")
    else:
        st.warning("⚠️ Could not determine wine quality from Azure ML response. Cannot generate tasting notes.")

# --- Main Execution Block ---
if st.button("Predict & Generate Tasting Note"):
    with st.spinner("Analyzing wine and crafting notes..."):
        # Construct the payload using current UI values
        current_payload = get_input_data_payload(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates,
            alcohol, color, qual_bool
        )

        # Prepare parameters for the OpenAI prompt
        prompt_params = {
            "color": color,
            "alcohol": alcohol,
            "ph": ph,
            "residual_sugar": residual_sugar,
            "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity,
            "sulphates": sulphates
        }

        display_results(
            azure_payload=current_payload,
            prompt_parameters=prompt_params
        )
