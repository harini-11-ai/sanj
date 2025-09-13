import openml
import pandas as pd
from datasets import load_dataset

def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Handle different data types and ensure proper DataFrame creation
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        
        if isinstance(y, pd.Series):
            df[dataset.default_target_attribute] = y
        else:
            df[dataset.default_target_attribute] = pd.Series(y)
        
        # Clean column names (remove special characters)
        df.columns = [str(col).replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load OpenML dataset {dataset_id}: {str(e)}")
        return None

def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        df = dataset['train'].to_pandas()
        return df.head(2000)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
