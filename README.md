# ML Assignment 2 – Classification Models & Streamlit App

## a. Problem Statement

Predict whether a person’s income exceeds $50K/year based on census attributes (binary classification: >50K vs <=50K). The task is to implement multiple classifiers, compare them with standard metrics, and deploy an interactive Streamlit app for model selection and evaluation.

## b. Dataset Description [1 mark]

- Source: UCI Machine Learning Repository – Adult (Census Income) dataset.
- Link: [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult)
- Task: Binary classification (income >50K or ≤50K).
- Instances: 48,842 total; after removing missing values, 30,162 instances used (meets minimum 500).
- Features: 14 attributes (meets minimum 12):
  - Numeric: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Target: `income` (binary: `>50K` / `<=50K`).
- Preprocessing: Rows with missing values (`?`) were dropped; numeric features scaled; categorical features one-hot encoded.

## c. Models Used [6 marks]

Comparison table of evaluation metrics for all 6 models (on the same preprocessed test set):

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression      | 0.8470   | 0.9025| 0.7380    | 0.5972 | 0.6602| 0.5681|
| Decision Tree             | 0.8150   | 0.7504| 0.6301    | 0.6217 | 0.6259| 0.5031|
| kNN                       | 0.8380   | 0.8859| 0.7040    | 0.6020 | 0.6491| 0.5474|
| Naive Bayes               | 0.5648   | 0.7784| 0.3586    | 0.9489 | 0.5205| 0.3523|
| Random Forest (Ensemble)   | 0.8517   | 0.8992| 0.7370    | 0.6287 | 0.6786| 0.5861|
| XGBoost (Ensemble)        | 0.8659   | 0.9213| 0.7700    | 0.6580 | 0.7096| 0.6264|

### Observations on Model Performance [3 marks]

| ML Model Name            | Observation about model performance |
|--------------------------|--------------------------------------|
| Logistic Regression      | Strong AUC (0.90) and accuracy; good baseline. Lower recall for the minority class (>50K). |
| Decision Tree             | Fast and interpretable; lower AUC and metrics than linear/ensemble models; prone to overfitting. |
| kNN                       | Good AUC and accuracy; sensitive to scale (handled by preprocessing); no probabilistic calibration. |
| Naive Bayes               | High recall but low precision and accuracy; strong class imbalance sensitivity; best for detecting >50K when missing positives is costly. |
| Random Forest (Ensemble)  | Balanced accuracy, AUC, and F1; robust to overfitting; good trade-off between performance and stability. |
| XGBoost (Ensemble)        | Best overall accuracy, AUC, F1, and MCC; best choice for this dataset among the six; higher tuning potential. |

---

## Repository Structure

```
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── adult.csv
├── model/
│   ├── train_models.py
│   ├── preprocessor.joblib
│   ├── metrics.json
│   └── model_*.joblib (saved models)
└── adult/ (dataset documentation)
```

## How to Run

1. Train models (optional):  
   `python model/train_models.py`  
   (Requires: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`.)

2. Run Streamlit app:  
   `streamlit run app.py`

3. Deploy: Push to GitHub and deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) with `app.py` as main file.
