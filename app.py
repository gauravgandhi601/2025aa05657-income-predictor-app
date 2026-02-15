import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Income Predictor", layout="wide")

st.title("💰 Adult Income Classification Dashboard")
st.write("Predict whether income exceeds $50K/year")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Logistic Regression","Decision Tree","KNN","Naive Bayes","Random Forest","XGBoost"]
)

uploaded_file = st.file_uploader("Upload TEST CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = pickle.load(open(f"models/{model_choice}.pkl","rb"))
    preds = model.predict(X)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, preds))

    with col2:
        st.subheader("Classification Report")
        st.text(classification_report(y, preds))

    st.success("Prediction completed!")