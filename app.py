import streamlit as st
import pandas as pd
import numpy as np
from datasets_search import load_openml_dataset, load_huggingface_dataset
from preprocessing import preprocess_data
from models import train_and_evaluate
from utils import (
    plot_correlation_matrix, plot_feature_distributions, plot_boxplots_numeric_vs_target,
    plot_roc_curve, plot_confusion_matrix, plot_feature_importance, plot_residuals,
    plot_prediction_probabilities, plot_actual_vs_predicted
)

st.set_page_config(page_title="ML Playground", layout="wide")

st.title("🚀 Machine Learning Playground")
st.markdown("**Advanced ML with Comprehensive EDA & Model Evaluation**")

# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

# Add info about dataset sources
with st.sidebar.expander("ℹ️ Dataset Info"):
    st.write("""
    **📁 Upload CSV**: Upload your own dataset
    
    **🔬 OpenML**: 1000+ datasets for ML research
    - Classification: Iris, Wine, Breast Cancer, etc.
    - Regression: Boston Housing, Auto MPG, etc.
    
    **🤗 Hugging Face**: NLP and text datasets
    - Sentiment: IMDB, Amazon, Yelp reviews
    - Classification: News, DBpedia
    - GLUE benchmark tasks
    """)

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    st.sidebar.subheader("📋 Popular Datasets")
    
    # Popular OpenML datasets
    popular_datasets = {
        "Iris (Classification)": "61",
        "Wine (Classification)": "187", 
        "Breast Cancer (Classification)": "13",
        "Diabetes (Classification)": "37",
        "Heart Disease (Classification)": "45",
        "Boston Housing (Regression)": "529",
        "Auto MPG (Regression)": "9",
        "Abalone (Regression)": "183",
        "CPU Performance (Regression)": "562",
        "Servo (Regression)": "871",
        "Glass Identification (Classification)": "40",
        "Sonar (Classification)": "151",
        "Vehicle (Classification)": "54",
        "Segment (Classification)": "36",
        "Waveform (Classification)": "60"
    }
    
    # Dataset selection method
    selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom ID"])
    
    if selection_method == "Popular Datasets":
        selected_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_datasets.keys()),
            index=0
        )
        openml_id = popular_datasets[selected_dataset]
        st.sidebar.caption(f"Dataset ID: {openml_id}")
    else:
        openml_id = st.sidebar.text_input("Enter OpenML dataset ID", "61")
        st.sidebar.caption("👉 Example: 61 = Iris dataset")
    
    if st.sidebar.button("Load from OpenML"):
        with st.spinner(f"🔄 Loading dataset {openml_id}..."):
            df = load_openml_dataset(openml_id)
            if df is None:
                st.error("❌ Failed to load dataset from OpenML")
            else:
                st.success(f"✅ Successfully loaded dataset {openml_id}")

elif dataset_source == "Hugging Face":
    st.sidebar.subheader("📋 Popular NLP Datasets")
    
    # Popular Hugging Face datasets
    popular_hf_datasets = {
        "IMDB Reviews (Sentiment)": "imdb",
        "Amazon Reviews (Sentiment)": "amazon_polarity",
        "Yelp Reviews (Sentiment)": "yelp_review_full",
        "AG News (Classification)": "ag_news",
        "DBpedia (Classification)": "dbpedia_14",
        "20 Newsgroups (Classification)": "newsgroup",
        "SQuAD (QA)": "squad",
        "CoLA (Grammar)": "glue",
        "SST-2 (Sentiment)": "glue",
        "MRPC (Paraphrase)": "glue",
        "QQP (Paraphrase)": "glue",
        "MNLI (NLI)": "glue",
        "QNLI (NLI)": "glue",
        "RTE (NLI)": "glue",
        "WNLI (NLI)": "glue"
    }
    
    # Dataset selection method
    hf_selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom Name"])
    
    if hf_selection_method == "Popular Datasets":
        selected_hf_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_hf_datasets.keys()),
            index=0
        )
        hf_name = popular_hf_datasets[selected_hf_dataset]
        st.sidebar.caption(f"Dataset: {hf_name}")
    else:
        hf_name = st.sidebar.text_input("Enter Hugging Face dataset name", "imdb")
        st.sidebar.caption("👉 Example: imdb = sentiment analysis dataset")
    
    if st.sidebar.button("Load from Hugging Face"):
        with st.spinner(f"🔄 Loading dataset {hf_name}..."):
            df = load_huggingface_dataset(hf_name)
            if df is None:
                st.error("❌ Failed to load dataset from Hugging Face")
            else:
                st.success(f"✅ Successfully loaded dataset {hf_name}")

if df is not None and not df.empty:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(df.columns))
    with col2:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col3:
        st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'category']).columns))

    target_col = st.selectbox("🎯 Select target column", df.columns)
    
    if target_col:
        # Determine problem type
        unique_targets = df[target_col].nunique()
        is_classification = unique_targets < 20
        
        st.info(f"**Problem Type:** {'Classification' if is_classification else 'Regression'} ({unique_targets} unique values)")
        
        # Preprocessing
        try:
            with st.spinner("🔄 Preprocessing data..."):
                X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
            
            # Check if preprocessing was successful
            if X_train is None or len(X_train) == 0:
                st.error("❌ Preprocessing failed: No valid data after preprocessing")
                st.stop()
            
            st.success(f"✅ Data preprocessed successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.info("💡 Try selecting a different target column or check your data for issues")
            st.stop()
        
        # Model Training
        try:
            with st.spinner("🤖 Training models..."):
                results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
            
            if not results or not trained_models:
                st.error("❌ Model training failed")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.info("💡 This might be due to data type issues or insufficient data")
            st.stop()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Results", "📈 Predictions", "ℹ️ Dataset Info"])
        
        with tab1:
            st.header("📊 Exploratory Data Analysis (EDA)")
            
            # Correlation Matrix
            st.subheader("📊 Correlation Heatmap")
            corr_plot = plot_correlation_matrix(df)
            if corr_plot:
                st.pyplot(corr_plot)
            else:
                st.warning("Not enough numeric features for correlation analysis")
            
            # Feature Distributions
            st.subheader("📈 Feature Distributions")
            dist_plot = plot_feature_distributions(df)
            if dist_plot:
                st.pyplot(dist_plot)
            else:
                st.warning("No features available for distribution analysis")
            
            # Boxplots for numeric vs target
            if is_classification:
                st.subheader("📉 Numeric Features vs Target")
                box_plot = plot_boxplots_numeric_vs_target(df, target_col)
                if box_plot:
                    st.pyplot(box_plot)
                else:
                    st.warning("Target column is not numeric or no numeric features available")
        
        with tab2:
            st.header("🤖 Model Results & Evaluation")
            
            # Model Performance Metrics
            st.subheader("📊 Performance Metrics")
            for model_name, metrics in results.items():
                with st.expander(f"**{model_name}**"):
                    if is_classification:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{metrics['MSE']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.3f}")
            
            # Model Evaluation Plots
            st.subheader("✅ Model Evaluation Visualizations")
            
            for model_name, metrics in results.items():
                st.write(f"#### {model_name}")
                
                if is_classification:
                    # ROC Curve
                    if 'probabilities' in metrics:
                        roc_plot = plot_roc_curve(y_test, metrics['probabilities'][:, 1], model_name)
                        if roc_plot:
                            st.pyplot(roc_plot)
                    
                    # Confusion Matrix
                    conf_plot = plot_confusion_matrix(y_test, metrics['predictions'], model_name)
                    if conf_plot:
                        st.pyplot(conf_plot)
                    
                    # Feature Importance (for tree-based models)
                    if hasattr(metrics['model'], 'feature_importances_'):
                        feat_imp_plot = plot_feature_importance(
                            metrics['model'], X_train.columns, model_name
                        )
                        if feat_imp_plot:
                            st.pyplot(feat_imp_plot)
                
                else:
                    # Residual Plot for regression
                    resid_plot = plot_residuals(y_test, metrics['predictions'], model_name)
                    if resid_plot:
                        st.pyplot(resid_plot)
                    
                    # Feature Importance (for tree-based models)
                    if hasattr(metrics['model'], 'feature_importances_'):
                        feat_imp_plot = plot_feature_importance(
                            metrics['model'], X_train.columns, model_name
                        )
                        if feat_imp_plot:
                            st.pyplot(feat_imp_plot)
        
        with tab3:
            st.header("📈 Predictions")
            
            # Model Selection for Predictions
            model_names = list(trained_models.keys())
            selected_model_name = st.selectbox("Select model for predictions:", model_names)
            selected_model = trained_models[selected_model_name]
            
            # Prediction Type Selection
            pred_type = st.radio("Choose prediction type:", ["Manual Input", "Batch Upload"])
            
            if pred_type == "Manual Input":
                st.subheader("🔮 Manual Predictions")
                st.write("Enter feature values for prediction:")
                
                # Create input form
                input_data = {}
                feature_cols = X_train.columns
                
                cols = st.columns(min(3, len(feature_cols)))
                for i, col in enumerate(feature_cols):
                    with cols[i % len(cols)]:
                        if X_train[col].dtype in ['int64', 'float64']:
                            # Numeric input
                            min_val = float(X_train[col].min())
                            max_val = float(X_train[col].max())
                            input_data[col] = st.number_input(
                                f"{col}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=float(X_train[col].median())
                            )
                        else:
                            # Categorical input
                            unique_vals = X_train[col].unique()
                            input_data[col] = st.selectbox(f"{col}", unique_vals)
                
                if st.button("🔮 Make Prediction"):
                    # Convert to DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    if is_classification:
                        pred = selected_model.predict(input_df)[0]
                        pred_proba = selected_model.predict_proba(input_df)[0]
                        
                        st.success(f"**Prediction:** {pred}")
                        st.write("**Class Probabilities:**")
                        classes = selected_model.classes_
                        for i, prob in enumerate(pred_proba):
                            st.write(f"  {classes[i]}: {prob:.3f}")
                    else:
                        pred = selected_model.predict(input_df)[0]
                        st.success(f"**Prediction:** {pred:.3f}")
            
            else:  # Batch Upload
                st.subheader("📁 Batch Predictions")
                uploaded_pred_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
                
                if uploaded_pred_file is not None:
                    pred_df = pd.read_csv(uploaded_pred_file)
                    st.write("**Uploaded Data Preview:**")
                    st.dataframe(pred_df.head())
                    
                    if st.button("🔮 Make Batch Predictions"):
                        # Ensure same features as training data
                        missing_cols = set(X_train.columns) - set(pred_df.columns)
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            # Make predictions
                            pred_df_subset = pred_df[X_train.columns]
                            predictions = selected_model.predict(pred_df_subset)
                            
                            # Add predictions to dataframe
                            result_df = pred_df.copy()
                            result_df['Prediction'] = predictions
                            
                            if is_classification:
                                probabilities = selected_model.predict_proba(pred_df_subset)
                                for i, class_name in enumerate(selected_model.classes_):
                                    result_df[f'Probability_{class_name}'] = probabilities[:, i]
                            
                            st.write("**Prediction Results:**")
                            st.dataframe(result_df)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
            
            # Prediction Visualizations
            st.subheader("📊 Prediction Visualizations")
            
            if is_classification and 'probabilities' in results[selected_model_name]:
                # Probability Distribution
                prob_plot = plot_prediction_probabilities(
                    results[selected_model_name]['probabilities'], selected_model_name
                )
                if prob_plot:
                    st.pyplot(prob_plot)
                
                # ROC Curve for predictions
                roc_plot = plot_roc_curve(
                    y_test, results[selected_model_name]['probabilities'][:, 1], selected_model_name
                )
                if roc_plot:
                    st.pyplot(roc_plot)
            
            else:
                # Actual vs Predicted for regression
                actual_pred_plot = plot_actual_vs_predicted(
                    y_test, results[selected_model_name]['predictions'], selected_model_name
                )
                if actual_pred_plot:
                    st.pyplot(actual_pred_plot)
        
        with tab4:
            st.header("ℹ️ Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("📈 Statistical Summary")
                st.dataframe(df.describe())
            
            st.subheader("🔍 Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.bar_chart(missing_data)
            else:
                st.success("✅ No missing values found!")
            
            st.subheader("🎯 Target Variable Analysis")
            st.write(f"**Target Column:** {target_col}")
            st.write(f"**Unique Values:** {df[target_col].nunique()}")
            st.write(f"**Data Type:** {df[target_col].dtype}")
            
            if is_classification:
                st.write("**Class Distribution:**")
                class_counts = df[target_col].value_counts()
                st.bar_chart(class_counts)
            else:
                st.write("**Target Statistics:**")
                st.write(df[target_col].describe())
