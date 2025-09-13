import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_col):
    """Enhanced preprocessing with better error handling"""
    try:
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove rows where target is NaN
        df_processed = df_processed.dropna(subset=[target_col])
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle missing values in features
        # For numeric columns, use median imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:  # Check if column still exists after imputation
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Remove any remaining NaN values
        X = X.fillna(X.median())
        
        # Check if we have valid data
        if X.empty or len(X) == 0:
            raise ValueError("No valid data after preprocessing")
        
        # Ensure we have at least 2 samples for train/test split
        if len(X) < 2:
            raise ValueError("Not enough data for train/test split")
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        # Fallback to simple preprocessing
        print(f"Advanced preprocessing failed, using fallback: {e}")
        
        # Simple fallback
        df_processed = df.dropna()
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Simple encoding
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        # Ensure numeric types
        X = X.astype(float)
        y = y.astype(float)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
