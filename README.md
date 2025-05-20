# wine-quality-app
🍷 AI Wine Quality Assistant
This Streamlit app uses a machine learning model deployed via Azure Machine Learning to predict wine quality from chemical properties. It then uses OpenAI's GPT-4o to generate a tasting note and food pairing suggestion based on the prediction.

🚀 Features
Predicts wine quality score based on chemical properties

Generates tasting notes using OpenAI GPT

Built with Streamlit

Backed by Azure ML for model inference

Configured via environment variables:
- AZURE_ENDPOINT_URI: Your Azure ML endpoint URI
- AZURE_API_KEY: Your Azure ML API key
- OPENAI_API_KEY: Your OpenAI API key

### 🚀 Live Demo
Check out the app live: [wine-quality2.streamlit.app](https://wine-quality2.streamlit.app)


## 📊 Data Source

This app uses the **Wine Quality dataset** from the UCI Machine Learning Repository:

- UCI Link: [https://archive.ics.uci.edu/ml/datasets/wine+quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Citation:
  > P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
  > Modeling wine preferences by data mining from physicochemical properties.  
  > In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

The dataset includes physicochemical properties and quality scores for red and white Portuguese "Vinho Verde" wines.

