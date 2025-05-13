import streamlit as st
import requests
import json
from openai import OpenAI
import streamlit.components.v1 as components

# Inject Google Tag Manager using Measurement ID from secrets
GA_MEASUREMENT_ID = st.secrets["GA_MEASUREMENT_ID"]

components.html(
    f"""
    <!-- Google Tag Manager -->
    <script>
      (function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{'gtm.start':
      new Date().getTime(),event:'gtm.js'}});var f=d.getElementsByTagName(s)[0],
      j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
      'https://www.googletagmanager.com/gtm.js?id=' + i + dl;f.parentNode.insertBefore(j,f);
      }})(window,document,'script','dataLayer','{GA_MEASUREMENT_ID}');
    </script>
    <!-- End Google Tag Manager -->
    """,
    height=0,
    width=0
)

# Azure ML endpoint setup
AZURE_ENDPOINT_URI = st.secrets["AZURE_ENDPOINT_URI"]
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]

# OpenAI setup
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

st.title("🍷 AI Wine Quality Assistant")

# Collect user input
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

# Prepare input data
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

# Predict and generate tasting note
if st.button("Predict & Generate Tasting Note"):
    with st.spinner("Predicting wine quality..."):
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

                    prompt = (
                        f"A {color} wine has a predicted quality score of {quality:.1f}.\n"
                        f"Characteristics:\n"
                        f"- Alcohol: {alcohol}%\n"
                        f"- pH: {ph}\n"
                        f"- Residual sugar: {residual_sugar}g/L\n"
                        f"- Fixed acidity: {fixed_acidity}\n"
                        f"- Volatile acidity: {volatile_acidity}\n"
                        f"- Sulphates: {sulphates}\n\n"
                        "Write a 2-sentence tasting description and suggest a food pairing."
                    )

                    gpt_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=150
                    )

                    st.markdown("### 🍷 GPT-Generated Tasting Note")
                    st.write(gpt_response.choices[0].message.content.strip())
                else:
                    st.error("Prediction returned no value.")
            else:
                st.error(f"❌ Azure ML call failed ({response.status_code})")
                st.text(response.text)

        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
