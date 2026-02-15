"""
Streamlit app for ML Assignment 2: Classification models on Adult (UCI) dataset.
Features: CSV upload, model selection, evaluation metrics, confusion matrix & classification report.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "adult.csv")
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]
FEATURES = COLUMNS[:-1]
TARGET = "income"

MODEL_DISPLAY_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
    "XGBoost (Ensemble)",
]


def load_artifacts():
    """Load preprocessor, metrics, and model names """
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    if "metrics" not in st.session_state:
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            st.session_state.metrics = json.load(f)
    return st.session_state.preprocessor, st.session_state.metrics


def get_model_filename(display_name):
    """Map display name to saved joblib filename."""
    name = display_name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"model_{name}.joblib"


def load_model(display_name):
    """Load a single model by display name."""
    path = os.path.join(MODEL_DIR, get_model_filename(display_name))
    return joblib.load(path)


def preprocess_df(df):
    """Preprocess uploaded DataFrame"""
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace("?", np.nan)
    df = df.dropna()
    if TARGET in df.columns:
        y = (df[TARGET].astype(str).str.strip() == ">50K").astype(int).values
    else:
        y = None
    X = df[FEATURES]
    return X, y


def run_inference(preprocessor, model, X):
    """Transform features and predict"""
    X_enc = preprocessor.transform(X)
    X_enc = np.asarray(X_enc)
    y_pred = model.predict(X_enc)
    y_proba = model.predict_proba(X_enc)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_proba


def main():
    st.set_page_config(page_title="ML Assignment 2 – Classification", layout="wide")
    st.title("Classification Models – Adult (UCI) Income")
    st.markdown("Upload a test CSV (same schema as Adult dataset) or use default data. Select a model to view metrics and predictions.")

    preprocessor, metrics = load_artifacts()

    # --- Dataset upload (CSV) [1 mark] ---
    st.header("1. Dataset")
    use_default = st.checkbox("Use default test data (sample from data/adult.csv)", value=True)
    df = None
    if use_default:
        if os.path.isfile(DATA_PATH):
            df_full = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, skipinitialspace=True)
            df = df_full.sample(n=min(1000, len(df_full)), random_state=42)
        else:
            st.warning("Default data file not found. Please upload a CSV.")
    else:
        uploaded_file = st.file_uploader("Upload test CSV (Adult schema: 14 features + income column)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, header=None, names=COLUMNS, skipinitialspace=True)

    if df is not None:
        try:
            X, y_true = preprocess_df(df.copy())
        except Exception as e:
            st.error(f"Preprocessing error. Ensure CSV has columns matching Adult dataset. {e}")
            return
        if len(X) == 0:
            st.error("No rows left after dropping missing values. Check for '?' or missing data.")
            return
        st.success(f"Loaded **{len(X)}** instances.")
    else:
        X = None
        y_true = None

    # --- Model selection ---
    st.header("2. Model selection")
    selected_model = st.selectbox("Choose a model", MODEL_DISPLAY_NAMES, key="model_select")
    model = load_model(selected_model)

    # --- Display evaluation metrics  ---
    st.header("3. Evaluation metrics (on full test set)")
    m = metrics.get(selected_model, {})
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{m.get('Accuracy', 0):.4f}")
    col2.metric("AUC", f"{m.get('AUC', 0):.4f}")
    col3.metric("Precision", f"{m.get('Precision', 0):.4f}")
    col4.metric("Recall", f"{m.get('Recall', 0):.4f}")
    col5.metric("F1", f"{m.get('F1', 0):.4f}")
    col6.metric("MCC", f"{m.get('MCC', 0):.4f}")

    # Full comparison table
    st.subheader("Comparison table (all 6 models)")
    rows = []
    for name in MODEL_DISPLAY_NAMES:
        row = {"ML Model": name, **metrics.get(name, {})}
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)

    # --- Confusion matrix / Classification report  ---
    st.header("4. Predictions on your data")
    if X is not None and len(X) > 0:
        y_pred, y_proba = run_inference(preprocessor, model, X)
        if y_true is not None and len(y_true) == len(y_pred):
            # Classification report
            st.subheader("Classification report")
            report = classification_report(
                y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True
            )
            st.dataframe(pd.DataFrame(report).round(4).T, use_container_width=True, hide_index=True)
            # Confusion matrix
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
            # Metrics on this upload
            st.caption("Metrics on uploaded/sample data:")
            acc = accuracy_score(y_true, y_pred)
            if y_proba is not None and len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_proba)
                st.write(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
            else:
                st.write(f"Accuracy: {acc:.4f}")
        else:
            st.info("No ground-truth 'income' column in upload – showing prediction counts only.")
            st.write(pd.Series(y_pred).value_counts().rename("Count"))
    else:
        st.info("Upload a CSV or use default data to see confusion matrix and classification report.")

    st.markdown("---")
    st.caption("ML Assignment 2 – BITS PILANI | Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost.")


if __name__ == "__main__":
    main()
