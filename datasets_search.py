import pandas as pd
import openml
from datasets import load_dataset

def load_openml_dataset(dataset_id: str):
    dataset = openml.datasets.get_dataset(int(dataset_id))
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    df = pd.concat([X, y], axis=1)
    return df

def load_hf_dataset(name: str):
    dataset = load_dataset(name, split="train[:1000]")  # limit rows for demo
    return pd.DataFrame(dataset)
