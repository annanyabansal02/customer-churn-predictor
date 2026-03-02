# 📊 Customer Churn Predictor

A machine learning web application that predicts customer churn risk in real time.

## 🎯 Problem Statement
Customer churn is one of the biggest challenges in retail and telecom. 
This app helps businesses identify at-risk customers before they leave.

## 🚀 Live Demo
[Click here to try the app](#) ← we'll update this link after deployment

## 💡 Features
- Predicts churn probability in real time
- Smart feature selection — reduced from 19 to 7 features with only 1.7% accuracy loss
- Human-readable inputs with business context
- Actionable recommendations for retention
- Feature importance visualisation

## 🛠️ Tech Stack
- Python
- Scikit-learn (Random Forest Classifier)
- Streamlit
- Pandas
- Joblib

## 📊 Model Performance
- Algorithm: Random Forest Classifier
- Accuracy: 77%
- Dataset: Telco Customer Churn (7,043 customers)

## 🔍 Key Insights
- Top 3 features drive 51.8% of prediction power
- Month-to-Month contracts have highest churn risk
- New customers with high bills and no support are most likely to churn

## ⚙️ How to Run Locally
```bash
pip install -r requirements.txt
python model.py
streamlit run app.py
```

## 📁 Project Structure
```
churn_predictor/
├── app.py              # Streamlit UI
├── model.py            # ML model training
├── churn_model.pkl     # Saved model
├── model_columns.pkl   # Saved feature names
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```