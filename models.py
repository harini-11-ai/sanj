from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}
    is_classification = len(set(y_train)) < 20

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            
            results[name] = {
                "accuracy": acc,
                "predictions": preds,
                "probabilities": pred_proba,
                "model": model
            }
            trained_models[name] = model
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            
            results[name] = {
                "MSE": mse, 
                "RMSE": np.sqrt(mse),
                "predictions": preds,
                "model": model
            }
            trained_models[name] = model
    
    return results, trained_models
