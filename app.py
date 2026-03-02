import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to predict churn risk.")
st.divider()

# Input fields — clean and simple
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        help="How long has the customer been with us?"
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=150.0,
        value=50.0,
        help="Customer's current monthly bill"
    )

    TotalCharges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure * MonthlyCharges),
        help="Total amount charged to date"
    )

    Contract = st.selectbox(
        "Contract Type",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Month-to-Month",
            1: "One Year",
            2: "Two Year"
        }[x],
        help="Type of customer contract"
    )

with col2:
    PaymentMethod = st.selectbox(
        "Payment Method",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Bank Transfer",
            1: "Credit Card",
            2: "Electronic Check",
            3: "Mailed Check"
        }[x]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "No",
            1: "Yes",
            2: "No Internet Service"
        }[x]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "No",
            1: "Yes",
            2: "No Internet Service"
        }[x]
    )

st.divider()

# Predict button
if st.button("🔍 Predict Churn Risk", use_container_width=True):
    input_data = pd.DataFrame([[
        TotalCharges, MonthlyCharges, tenure,
        Contract, PaymentMethod, OnlineSecurity, TechSupport
    ]], columns=model_columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    # Result
    if prediction == 1:
        st.error("⚠️ High Churn Risk!")
        st.metric(
            label="Churn Probability",
            value=f"{probability[1]*100:.1f}%",
            delta="Needs Attention",
            delta_color="inverse"
        )
        st.markdown("**💡 Recommendation:** Consider offering a discounted long-term contract or loyalty reward.")
    else:
        st.success("✅ Low Churn Risk!")
        st.metric(
            label="Retention Probability",
            value=f"{probability[0]*100:.1f}%",
            delta="Healthy",
            delta_color="normal"
        )
        st.markdown("**💡 Recommendation:** Customer is stable. Good candidate for upselling.")

    # Feature importance chart
    st.divider()
    st.subheader("📈 Top Factors Driving This Prediction")
    importance_df = pd.DataFrame({
        'Feature': model_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))

    # Summary table
    st.divider()
    st.subheader("📋 Customer Summary")
    summary = pd.DataFrame({
        'Feature': ['Tenure', 'Monthly Charges', 'Total Charges',
                   'Contract', 'Payment Method', 'Online Security', 'Tech Support'],
        'Value': [
            f"{tenure} months",
            f"${MonthlyCharges:.2f}",
            f"${TotalCharges:.2f}",
            {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}[Contract],
            {0: "Bank Transfer", 1: "Credit Card",
             2: "Electronic Check", 3: "Mailed Check"}[PaymentMethod],
            {0: "No", 1: "Yes", 2: "No Internet"}[OnlineSecurity],
            {0: "No", 1: "Yes", 2: "No Internet"}[TechSupport]
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)