import streamlit as st
import pandas as pd
import torch
import os, sys
import joblib

# make sure we can import from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.finance_model import finance_model

# Fixed feature set (must match training!)
FEATURES = [
    'customer_tenure', 'industry', 'some_other_features',
    'company_size', 'region', 'operating_margin',
    'debt_ratio', 'log_revenue', 'gross_profit'
]

@st.cache_resource
def load_artifacts(model_path="models/finance_model.pth", input_dim=9):
    model = finance_model(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, scaler, encoders

st.title("📊 Business Forecasting App")
st.write("Predict **Revenue**, **Risk**, and **Churn** from business financial data using PyTorch.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with business features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data (raw)")
    st.dataframe(df.head())

    # --- Column check ---
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        # Drop extra columns automatically
        df = df[FEATURES]

        # --- Apply encoders ---
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
        for col in df.select_dtypes(include=['object']).columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
            else:
                st.error(f"No encoder found for column {col}")
                st.stop()

        # --- Scale numeric features ---
        X = scaler.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # --- Load model ---
        model, _, _ = load_artifacts(input_dim=len(FEATURES))

        # --- Run predictions ---
        with torch.no_grad():
            revenue, risk, churn = model(X_tensor)

        results = pd.DataFrame({
            "Revenue_Pred": revenue.numpy(),
            "Risk_Score": risk.numpy(),
            "Churn_Prob": torch.sigmoid(churn).numpy()
        })

        st.write("### Predictions")
        st.dataframe(results.head())

        # Visualization
        st.write("### 📈 Forecast Visualization")
        st.line_chart(results[["Revenue_Pred"]])
        st.bar_chart(results[["Churn_Prob"]])
