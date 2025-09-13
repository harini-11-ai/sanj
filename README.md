# 🚀 Machine Learning Playground

An advanced Streamlit application for comprehensive machine learning analysis with extensive EDA, model evaluation, and prediction capabilities.

## ✨ Features

### 📊 Exploratory Data Analysis (EDA)
- **📊 Correlation Heatmap** - Visualize relationships between numeric features
- **📈 Feature Distributions** - Histograms for numeric features, bar plots for categorical
- **📉 Boxplots** - Numeric features vs target variable analysis
- **📊 Dataset Overview** - Comprehensive data type and missing value analysis

### 🤖 Model Training & Evaluation
- **Automatic Problem Detection** - Classification vs Regression based on target variable
- **Multiple Algorithms** - Logistic Regression, Random Forest (Classification & Regression)
- **Performance Metrics** - Accuracy, MSE, RMSE with detailed reporting

### ✅ Model Evaluation Visualizations
- **ROC Curve** - For binary classification with AUC scores
- **Confusion Matrix** - Classification performance visualization
- **Feature Importance** - Tree-based model feature rankings
- **Residual Plots** - Regression model diagnostics with Q-Q plots

### 📈 Advanced Predictions
- **Manual Predictions** - Interactive form for single predictions
- **Batch Predictions** - CSV upload for bulk predictions with download
- **Prediction Visualizations**:
  - Classification: Probability distributions & ROC curves
  - Regression: Actual vs Predicted scatter plots with R² scores

### 📂 Data Sources
- **CSV Upload** - Local file upload
- **OpenML** - Access to 1000+ datasets
- **Hugging Face** - NLP and other datasets

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**
   - Navigate to `http://localhost:8501`
   - Choose your dataset source
   - Select target column
   - Explore EDA, train models, and make predictions!

## 📋 Requirements

- streamlit
- pandas
- numpy==1.26.4
- scikit-learn==1.3.2
- openml
- datasets
- matplotlib
- seaborn
- plotly
- scipy

## 🎯 Usage Examples

### Example 1: Iris Dataset (Classification)
1. Select "OpenML" as data source
2. Enter dataset ID: `61`
3. Select target column: `class`
4. Explore EDA tab for feature distributions
5. Check Model Results for ROC curves and confusion matrices
6. Use Predictions tab for manual or batch predictions

### Example 2: Boston Housing (Regression)
1. Select "OpenML" as data source
2. Enter dataset ID: `529`
3. Select target column: `MEDV`
4. View correlation heatmap and boxplots
5. Analyze residual plots in Model Results
6. Make predictions with actual vs predicted visualization

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit with tabbed interface
- **Backend**: scikit-learn for ML algorithms
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Sources**: OpenML API, Hugging Face datasets

### Key Components
- `app.py` - Main Streamlit application
- `utils.py` - Visualization functions
- `models.py` - Model training and evaluation
- `preprocessing.py` - Data preprocessing
- `datasets_search.py` - External dataset loading

### Supported Algorithms
- **Classification**: Logistic Regression, Random Forest Classifier
- **Regression**: Linear Regression, Random Forest Regressor

## 📊 Visualization Features

### EDA Plots
- Correlation matrices with customizable color schemes
- Multi-panel distribution plots for all feature types
- Boxplots showing feature-target relationships
- Missing value analysis and data type summaries

### Model Evaluation
- ROC curves with AUC calculations
- Confusion matrices with class labels
- Feature importance rankings for tree models
- Residual analysis with Q-Q plots for regression

### Prediction Analysis
- Probability distributions for classification
- Actual vs predicted scatter plots for regression
- Interactive prediction forms
- Batch prediction results with downloadable CSV

## 🎨 UI Features

- **Responsive Design** - Works on desktop and mobile
- **Tabbed Interface** - Organized workflow (EDA → Models → Predictions → Info)
- **Interactive Elements** - Dropdowns, sliders, file uploads
- **Progress Indicators** - Loading spinners for long operations
- **Error Handling** - Graceful error messages and validation
- **Download Options** - CSV export for prediction results

## 🔮 Future Enhancements

- Additional ML algorithms (SVM, XGBoost, Neural Networks)
- Hyperparameter tuning interface
- Cross-validation analysis
- Model comparison tools
- Advanced feature engineering
- Time series analysis support

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy Machine Learning! 🎉**
