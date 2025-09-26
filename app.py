import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Keras 3-compatible load (works with TF 2.17+)
from tensorflow import keras

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_model_and_assets():
    # Model: prefer new Keras loader; disable compile for portability
    model = keras.models.load_model("model.h5", compile=False)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Optional: exact feature order saved during training (if you created it)
    feature_order = None
    if os.path.exists("feature_order.pkl"):
        with open("feature_order.pkl", "rb") as f:
            feature_order = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler, feature_order


model, label_encoder_gender, onehot_encoder_geo, scaler, feature_order = load_model_and_assets()

# -----------------------------
# UI
# -----------------------------
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 40)
balance = st.number_input("Balance", value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", value=650, step=1)
estimated_salary = st.number_input("Estimated Salary", value=0.0, step=100.0)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=0)
is_active_member = st.selectbox("Is Active Member", [0, 1], index=0)

# -----------------------------
# Build model input
# -----------------------------
# base numeric / label-encoded features
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# one-hot Geography (works for sparse or dense)
geo = onehot_encoder_geo.transform([[geography]])
if hasattr(geo, "toarray"):  # sparse matrix case
    geo = geo.toarray()

geo_df = pd.DataFrame(
    geo,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# combine
X = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

# enforce exact training column order if available
if feature_order is not None:
    # any missing columns become 0; any extra columns are dropped
    X = X.reindex(columns=feature_order, fill_value=0)

# scale (assumes scaler fitted on the final X during training)
X_scaled = scaler.transform(X)

# -----------------------------
# Predict
# -----------------------------
proba = float(model.predict(X_scaled, verbose=0)[0][0])
st.write(f"Churn Probability: {proba:.2f}")

st.write("Prediction:", "Likely to churn" if proba > 0.5 else "Not likely to churn")
