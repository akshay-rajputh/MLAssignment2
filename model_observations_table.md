# Observations on the performance of each model on the chosen dataset

| **ML Model Name**        | **Observation about model performance** |
|--------------------------|----------------------------------------|
| Logistic Regression      | Strong overall: high accuracy (84.7%) and very high AUC (0.90), indicating good probability calibration. Recall (0.60) is the lowest among the top models, so it misses more >50K cases than the ensembles. |
| Decision Tree            | Moderate performance: accuracy 81.5% and lowest AUC (0.75) among non–Naive Bayes models. Single tree is more prone to overfitting; precision and recall are balanced but weaker than ensemble methods. |
| kNN                      | Mid-tier performance: accuracy 83.8% and AUC 0.89. Similar precision–recall trade-off to Logistic Regression. High-dimensional one-hot features may limit gains from distance-based voting. |
| Naive Bayes              | Poor fit for this dataset: accuracy only 56.5% with very low precision (0.36) and very high recall (0.95), leading to many false positives. Gaussian and independence assumptions are violated after one-hot encoding. |
| Random Forest (Ensemble) | Second-best overall: accuracy 85.2%, high AUC (0.90), and good balance across precision, recall, F1, and MCC. Ensemble of trees handles non-linearity and mixed feature types well. |
| XGBoost (Ensemble)       | Best performer: highest accuracy (86.6%), precision (0.77), recall (0.66), F1 (0.71), MCC (0.63), and AUC (0.92). Gradient boosting suits the Adult dataset’s mix of numeric and categorical features and class imbalance. |
