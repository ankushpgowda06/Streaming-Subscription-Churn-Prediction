import streamlit as st
import boto3
import numpy as np
import pandas as pd
import joblib
import io

st.set_page_config(layout="wide")

def load_s3_object(bucket_name, object_key):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_stream = io.BytesIO(response["Body"].read()) 
    return joblib.load(file_stream)

# Load model and label encoder from S3
bucket_name = "streaming-churn-bucket"
model = load_s3_object(bucket_name, "churn_prediction_model.pkl")
label_encoders = load_s3_object(bucket_name, "encoders.pkl")

def get_retention_strategy(probability):
    if probability > 0.8:
        return "High risk! Offer a loyalty discount or exclusive content."
    elif probability > 0.5 and probability < 0.8:
        return "Moderate risk. Engage with personalized recommendations."
    else:
        return "Low risk. Maintain engagement through regular updates."
    
st.title("Streaming Subscription Churn Prediction and Retention Strategies")
st.write("Enter user details to predict churn risk and get retention strategies.")
with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Age**")
        age = st.number_input("Enter the user's age in years (18+).", min_value=18, max_value=100, step=1)

        st.markdown("**Number of Subscription Pauses (0-5)**")
        num_subscription_pauses = st.number_input("Total times the user has paused their subscription.", min_value=0, max_value=5, step=1)

        st.markdown("**Weekly Hours**")
        weekly_hours = st.number_input("Average hours the user spends on the platform per week (0-50).", min_value=0.0, step=0.1)

    with col2:
        st.markdown("**Subscription Type**")
        subscription_type = st.selectbox("Type of subscription the user has.", ['Free', 'Premium', 'Family', 'Student'])

        st.markdown("**Customer Service Inquiries**")
        customer_service_inquiries = st.selectbox("User's level of engagement with customer service.", ['Low', 'Medium', 'High'])

        st.markdown("**Song Skip Rate**")
        song_skip_rate = st.number_input("Percentage of songs the user does not finish. (0-100).", min_value=0, max_value=100, step=1)/100

    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    encoded_inputs = {
        'age': age,
        'subscription_type': label_encoders['subscription_type'].transform([subscription_type])[0],
        'num_subscription_pauses': num_subscription_pauses,
        'customer_service_inquiries': label_encoders['customer_service_inquiries'].transform([customer_service_inquiries])[0],
        'weekly_hours': weekly_hours,
        'song_skip_rate': song_skip_rate
    }

    data = pd.DataFrame([encoded_inputs])

    prediction_prob = model.predict_proba(data)[0][1]
    prediction = model.predict(data)[0]
    churn_risk = "May Churn" if prediction == 1 else "Loyal User"
    strategy = get_retention_strategy(prediction_prob)
    
    if prediction_prob > 0.8:
        color = "red"
    elif 0.5 < prediction_prob <= 0.8:
        color = "orange"
    else:
        color = "green"

    st.markdown(f"""
        <span style='color:black; font-weight:bold; font-size:30px;'>Prediction: </span> 
        <span style='color:{color}; font-size:30px;'> {churn_risk}</span>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <span style='color:black; font-weight:bold; font-size:30px;'>Recommended Strategy: </span> 
        <span style='color:{color}; font-size:30px;'> {strategy}</span>
        """, unsafe_allow_html=True)
