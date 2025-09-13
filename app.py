import streamlit as st
import pandas as pd

from datasets_search import load_openml_dataset, load_hf_dataset
from preprocessing import preprocess_data
from models import train_models
from utils import plot_corr_matrix, plot_roc_curve

st.set_page_config(page_title="AI/ML Pipeline", layout="wide")

st.title("🔍 AI/ML Dataset Explorer & Model Trainer")

# ------------------ STEP 1: DATASET SELECTION ------------------
st.header("Step 1: Choose a dataset")

source = st.radio(
    "Select dataset source:",
    ["Upload CSV", "OpenML", "Hugging Face"],
    horizontal=True
)

df = None

if source == "Upload CSV":
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)

elif source == "OpenML":
    openml_id = st.text_input("Enter OpenML dataset ID (e.g., 61 for Iris)")
    if st.button("Load from OpenML") and openml_id:
        try:
            df = load_openml_dataset(openml_id)
        except Exception as e:
            st.error(f"Failed to load OpenML dataset: {e}")

elif source == "Hugging Face":
    hf_name = st.text_input("Enter Hugging Face dataset name (e.g., 'imdb')")
    if st.button("Load from HuggingFace") and hf_name:
        try:
            df = load_hf_dataset(hf_name)
        except Exception as e:
            st.error(f"Failed to load Hugging Face dataset: {e}")

# Show dataset preview
if df is not None:
    st.success(f"✅ Loaded dataset with shape {df.shape}")
    st.dataframe(df.head())

    # ------------------ STEP 2: PREPROCESSING ------------------
    st.header("Step 2: Preprocess Data")
    target_col = st.selectbox("Select target column", df.columns)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
    st.write("✅ Data preprocessed!")

    # ------------------ STEP 3: MODEL TRAINING ------------------
    st.header("Step 3: Train Models")
    results, trained_models = train_models(X_train, X_test, y_train, y_test)
    st.dataframe(results)

    # ------------------ STEP 4: VISUALIZATIONS ------------------
    st.header("Step 4: Visualizations")
    st.subheader("Correlation Matrix")
    st.pyplot(plot_corr_matrix(df))

    st.subheader("ROC Curves")
    st.pyplot(plot_roc_curve(trained_models, X_test, y_test))
