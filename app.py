import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sleep Disorder Screening App", layout="wide")

st.title("Sleep Disorder Screening App")
st.markdown(
    """
This application uses a trained Random Forest model to classify sleep stages
from wearable-inspired physiological feature data.

Important:
- Upload a CSV file that has the same feature columns used to train your model.
- This tool is designed for early screening and demonstration purposes.
- It is not a clinical diagnostic system.
"""
)

@st.cache_resource
def load_model():
    return joblib.load("rf_model_final.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load rf_model_final.pkl: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        predictions = model.predict(data)

        results_df = data.copy()
        results_df["Predicted Stage"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(results_df)

        st.subheader("Prediction Distribution")
        pred_counts = results_df["Predicted Stage"].value_counts()

        fig, ax = plt.subplots()
        pred_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Sleep Stage")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Sleep Stage Distribution")
        st.pyplot(fig)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(data)
            prob_df = pd.DataFrame(probas, columns=model.classes_)
            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df.head())

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
else:
    st.info("Upload a CSV file to generate predictions.")
