# 2025aa05657-income-predictor-app
Income Predictor App using ML 

**Problem Statement**

The goal of this project is to build and deploy multiple Machine Learning classification models that predict whether a person earns more than $50K per year based on demographic and employment attributes.
This project demonstrates a complete end-to-end Machine Learning deployment workflow including:
- Data preprocessing
- Training multiple ML models
- Performance evaluation using multiple metrics
- Building an interactive Streamlit web app
- Deploying the application on Streamlit Community Cloud

**Dataset Description**
Dataset: **Adult Census Income Dataset (UCI / Kaggle)**
This dataset contains demographic and employment information collected from the US Census Bureau.

Dataset Characteristics
- Instances: 48,842
- Features: 14 input features + 1 target
- Problem Type: Binary Classification
- Target Variable
  income
  0 → Income ≤ 50K
  1 → Income > 50K
- Example Features
  Age
  Education level
  Occupation
  Marital status
  Hours per week
  Capital gain / loss
  Native country

This dataset is ideal for comparing multiple classification algorithms because it contains both numerical and categorical features.

**Machine Learning Models Implemented**
Six classification models were trained on the same dataset:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor (KNN)
- Naive Bayes (GaussianNB)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

** Model Comparison Results**

| Model                | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|----------------------|----------|------|-----------|--------|------|------|
| Logistic Regression  | 0.818    | 0.845| 0.718     | 0.440  | 0.546| 0.461|
| Decision Tree        | 0.798    | 0.736| 0.589     | 0.613  | 0.601| 0.466|
| KNN                  | 0.822    | 0.841| 0.663     | 0.578  | 0.618| 0.505|
| Naive Bayes          | 0.793    | 0.841| 0.675     | 0.323  | 0.437| 0.362|
| Random Forest        | 0.851    | 0.899| 0.739     | 0.622  | 0.675| 0.583|
| XGBoost              | 0.860    | 0.921| 0.754     | 0.650  | 0.698| 0.611|


**Model Performance Observations**
| Model | Observation about Model Performance |
|-------|--------------------------------------|
| Logistic Regression | Provides a strong baseline with good AUC and precision, but low recall indicates difficulty capturing all high-income cases. Works well for linear decision boundaries. |
| Decision Tree       | Balanced recall and F1 score but lower AUC shows weaker probability estimation. Tends to overfit and lacks generalization compared to ensemble methods. |
| KNN                 | Good accuracy and balanced precision–recall. However, performance is slightly lower than ensemble models and can be computationally expensive for large datasets. |
| Naive Bayes         | Fast and simple but weakest overall performance, especially very low recall. Assumption of feature independence limits effectiveness on this dataset. |
| Random Forest (Ensemble) | Significant performance improvement with strong accuracy, AUC, F1, and MCC. Handles non-linearity and feature interactions well. Good balance between bias and variance. |
| XGBoost (Ensemble)       | Best performing model overall with highest Accuracy, AUC, F1, and MCC. Excellent at capturing complex patterns and handling imbalanced data. Recommended final model. |
