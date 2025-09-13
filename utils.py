import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# EDA Visualizations
def plot_correlation_matrix(df):
    """📊 Correlation Heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                square=True, cbar_kws={"shrink": .8})
    plt.title("📊 Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """📈 Feature Distribution plots (histograms for numeric, bar plots for categorical)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Calculate subplot dimensions
    total_cols = len(numeric_cols) + len(categorical_cols)
    if total_cols == 0:
        return None
    
    n_cols = min(3, total_cols)
    n_rows = (total_cols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric features as histograms
    for col in numeric_cols:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title(f"📊 {col} (Numeric)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
    
    # Plot categorical features as bar plots
    for col in categorical_cols:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f"📊 {col} (Categorical)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("📈 Feature Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_boxplots_numeric_vs_target(df, target_col):
    """📉 Boxplots for numeric features vs target (if target is numeric)"""
    if target_col not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0 or df[target_col].dtype not in [np.number]:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
            axes[i].set_title(f"📉 {col} vs {target_col}", fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"📉 Numeric Features vs Target ({target_col})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Model Evaluation Visualizations
def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """✅ ROC Curve for binary classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'✅ ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """✅ Confusion Matrix for classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'✅ Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Model", top_n=10):
    """✅ Feature Importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    # Adjust top_n if there are fewer features
    actual_top_n = min(top_n, n_features)
    indices = np.argsort(importances)[::-1][:actual_top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(actual_top_n), importances[indices])
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'✅ Feature Importance - {model_name}', fontsize=16, fontweight='bold')
    ax.set_xticks(range(actual_top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, model_name="Model"):
    """✅ Residual Plot for regression"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'✅ Residuals vs Predicted - {model_name}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot of Residuals - {model_name}', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Prediction Visualizations
def plot_prediction_probabilities(y_pred_proba, model_name="Model"):
    """📊 Probability Distribution for Classification Predictions"""
    if y_pred_proba.ndim == 1:
        # Binary classification
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        # Multi-class classification
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(y_pred_proba.shape[1]):
            ax.hist(y_pred_proba[:, i], bins=30, alpha=0.6, label=f'Class {i}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """📊 Scatter plot of actual vs predicted for regression"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'📊 Actual vs Predicted - {model_name}', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig
