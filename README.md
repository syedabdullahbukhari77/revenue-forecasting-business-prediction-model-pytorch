# 📊 Business KPI Prediction from Financial Data (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/syedabdullahbukhari77/Revenue_Forecasting_Business_Prediction_model_Pytorch/blob/main/notebooks/demo.ipynb](https://colab.research.google.com/drive/1PZDRWDH1c7dxteG9uOvPkiQ5mfuMPHw7))

---

## 1. Introduction
This repository implements a **multi-task deep learning model** for predicting key business performance indicators (KPIs) from tabular financial/accounting data.  
The model jointly learns three predictive tasks:

- **Revenue Growth** — regression  
- **Risk Score** — regression  
- **Customer Churn** — binary classification  

By training on all three tasks simultaneously, the network captures richer patterns and relationships in financial data.

---

## 2. Motivation
Conventional financial forecasting methods (linear models, rule-based systems, spreadsheets) struggle with **nonlinear dependencies** and **noisy real-world data**.  

This project demonstrates how **multi-task learning (MTL)** can serve as a scalable, modern approach to forecasting KPIs, providing:

- **One model → multiple outputs**  
- **Feature integration** across categorical, numerical, and temporal inputs  
- **Extensible baseline** for real-world datasets  

---

## 3. Dataset
- **Source:** `synthetic_financial_data_bukharii.csv` (synthetic but realistic)  
- **Records:** ~34,000 rows (500 companies × 68 months)  
- **Features:**  
  - *Numerical*: revenue, gross_profit, operating_margin, debt_ratio, log_revenue  
  - *Categorical*: industry, region, company_size  
  - *Temporal*: customer_tenure, date  
- **Targets:** revenue growth, risk score, churn indicator  

⚠️ Note: Dataset is synthetic and designed for experimentation.

---

## 4. Model Architecture
- **Base network:** Fully connected layers with BatchNorm + Dropout  
- **Output heads:**  
  - Revenue growth → Linear regression head (MSELoss)  
  - Risk score → Linear regression head (MSELoss)  
  - Churn → Binary classification head (BCEWithLogitsLoss)  
- **Framework:** PyTorch  

---

## 5. Preprocessing Pipeline
All preprocessing steps are modularized in `preprocessing.py`:
- Missing value handling  
- Feature scaling (with `scaler.pkl`)  
- Label/categorical encoding (with `encoders.pkl`)  
- Automatic column validation during inference  

This ensures **consistency between training and deployment**.

---

## 6. Training Setup
- **Split:** 2020–2024 → training | 2024–2025 → validation/testing  
- **Optimizer:** Adam (`lr = 5e-5`)  
- **Loss:** MSE(revenue) + MSE(risk) + BCE(churn)
- **Batch size:** 128  
- **Epochs:** 100  

### Sample Training Output
| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1     | 1.1861     | 1.2510          |
| 5     | 0.7073     | 0.7825          |
| 10    | 0.6764     | 0.6891          |
| 15    | 0.5981     | 0.6107          |
| 20    | 0.6054     | 0.6232          |

---

## 7. Results
- Model shows **stable convergence** on all three tasks.  
- Training and validation losses are closely aligned → low overfitting on synthetic data.  
- Demonstrates feasibility of **joint KPI forecasting** using deep learning.  

*(Future improvement: report R² for regression tasks and AUC/F1 for churn classification.)*

---

## 8. Deployment (Streamlit App)
This repository includes a **Streamlit web app** for interactive predictions.

### Run Locally
Make sure you have trained the model and saved artifacts (`finance_model.pth`, `scaler.pkl`, `encoders.pkl`) in the `models/` folder.

```bash
# Step 1: Train the model
python train.py

# Step 2: Validate model performance
python validate.py

# Step 3: Launch Streamlit app
streamlit run server/app.py

