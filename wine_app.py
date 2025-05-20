import streamlit as st
import requests
import json
import os

# --- Load environment variables ---
AZURE_ENDPOINT_URI = os.getenv("AZURE_ENDPOINT_URI")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not all([AZURE_ENDPOINT_URI, AZURE_API_KEY, OPENAI_API_KEY]):
    st.error("⚠️ Missing required environment variables. Please set AZURE_ENDPOINT_URI, AZURE_API_KEY, and OPENAI_API_KEY.")
    st.stop()

st.title("🍷 AI Wine Quality Assistant")

# --- User inputs ---
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

# --- Prepare input for Azure ML endpoint ---
input_data = {
    "columns": [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
        "color", "qual_bool"
    ],
    "data": [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol,
        color,
        qual_bool
    ]]
}

def predict_and_generate_note(input_data, color, alcohol, ph, residual_sugar, fixed_acidity, volatile_acidity, sulphates):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        body = json.dumps({"input_data": input_data})
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_API_KEY}"
        }
        response = requests.post(AZURE_ENDPOINT_URI, headers=headers, data=body)
        if response.status_code == 200:
            result = response.json()
            st.write("✅ Raw model response:", result)
            quality = result[0] if isinstance(result, list) and len(result) > 0 else None
            if quality is not None:
                st.success(f"Predicted Wine Quality Score: {quality:.2f}")
                prompt = f"""
                A {color} wine has a predicted quality score of {quality:.1f}.
                Characteristics:
                - Alcohol: {alcohol}%
                - pH: {ph}
                - Residual sugar: {residual_sugar}g/L
                - Fixed acidity: {fixed_acidity}
                - Volatile acidity: {volatile_acidity}
                - Sulphates: {sulphates}

                Write a 2-sentence tasting description and suggest a food pairing.
                """
                gpt_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                st.markdown("### 🍷 GPT-Generated Tasting Note")
                st.write(gpt_response.choices[0].message.content.strip())
            else:
                st.warning("Prediction returned no value.")
        else:
            st.error(f"❌ Azure ML call failed ({response.status_code})")
            st.text(response.text)
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

# --- Prediction and GPT output ---
if st.button("Predict & Generate Tasting Note"):
    with st.spinner("Predicting wine quality..."):
        predict_and_generate_note(
            input_data,
            color,
            alcohol,
            ph,
            residual_sugar,
            fixed_acidity,
            volatile_acidity,
            sulphates
        )
