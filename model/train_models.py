"""
Run project: python model/train_models.py
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import xgboost as xgb

# Column names from UCI adult.names (features + target)
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]
FEATURES = COLUMNS[:-1]
TARGET = "income"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "adult.csv")
MODEL_DIR = SCRIPT_DIR


def load_and_preprocess(data_path=DATA_PATH):
        
    df = pd.read_csv(data_path, header=None, names=COLUMNS, skipinitialspace=True)
    
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].str.strip()
    df = df.replace("?", np.nan)
    df = df.dropna()
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df


def get_preprocessor(df):
    """Build preprocessing pipeline: numeric scale + categorical one-hot."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_cols = [c for c in FEATURES if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor


def get_metrics(y_true, y_pred, y_proba=None):
    """Compute all required metrics."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["AUC"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["AUC"] = 0.0
    return metrics


def main():
    print("Loading data...")
    df = load_and_preprocess()
    print(f"Shape after dropna: {df.shape}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(df)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    X_train_enc = np.asarray(X_train_enc)
    X_test_enc = np.asarray(X_test_enc)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost (Ensemble)": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    all_metrics = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train_enc, y_train)
        y_pred = clf.predict(X_test_enc)
        y_proba = clf.predict_proba(X_test_enc)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics = get_metrics(y_test, y_pred, y_proba)
        all_metrics[name] = metrics
        joblib.dump(clf, os.path.join(MODEL_DIR, f"model_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"))
        print(f"  {name}: Accuracy={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}")

    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Saving feature names after encoding (optional)
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURES))
    print("Models and metrics saved in model/")


if __name__ == "__main__":
    main()
