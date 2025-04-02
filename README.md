# 🎵 Music Streaming Subscription Churn Prediction

## 🎯 Project Overview
This project aims to predict customer churn for a music streaming service using machine learning. By leveraging user behavior data, we built a predictive model to identify users who are likely to cancel their subscriptions. The final model is deployed as a **Streamlit web app** and hosted on **AWS EC2** for interactive use.

🔴 **Live Project**: [http://16.171.143.216:8501/](http://16.171.143.216:8501/)  
📹 **Video Demo**: [https://drive.google.com/file/d/1kWxPQIMKvLJwuGuRenQ3h8kKGxKZ5JHA/view?usp=sharing](https://drive.google.com/file/d/1kWxPQIMKvLJwuGuRenQ3h8kKGxKZ5JHA/view?usp=sharing/)

---
## 📊 Dataset Overview
The dataset, sourced from Kaggle, contains user information and engagement metrics. Key features include:
- **User demographics**: `customer_id`, `age`, `location`
- **Subscription details**: `subscription_type`, `payment_plan`, `num_subscription_pauses`, `payment_method`
- **User activity metrics**: `weekly_hours`, `average_session_length`, `song_skip_rate`, `weekly_songs_played`, `weekly_unique_songs`
- **Social engagement**: `num_favorite_artists`, `num_platform_friends`, `num_playlists_created`, `num_shared_playlists`, `notifications_clicked`
- **Customer interactions**: `customer_service_inquiries`
- **Target variable**: `churned` (0 = Active, 1 = Churned)

---
## 📈 Exploratory Data Analysis (EDA)
Before building the model, extensive **EDA** was conducted:
- **Numerical Feature Analysis**: Histograms and distribution plots to understand feature distributions.
- **Categorical Feature Analysis**: Bar charts to check class imbalances.
- **Boxplots**: Identified outliers for numerical features.
- **Correlation Heatmap**: Explored relationships between features.

---
## 🏗️ Data Preprocessing
- **Label Encoding**: Encoded categorical variables.
- **Train-Test Split**: Prepared data for model training and evaluation.

---
## 🧠 Model Building & Optimization
Three machine learning models were trained and evaluated:
1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **XGBoost Classifier** ✅ (Best performing model)

### **Hyperparameter Tuning**
- **RandomizedSearchCV** was used to fine-tune the **XGBoost** model for optimal performance.
- **SHAP Values** were analyzed to remove redundant features and improve model interpretability.

---
## 🚀 Deployment
### **AWS Setup**
- **Model & Encoder Storage**: Uploaded trained model and label encoder to **AWS S3**.
- **Interactive Web App**: Built a **Streamlit app** for user-friendly churn prediction.
- **Cloud Hosting**: Deployed the Streamlit app on **AWS EC2** for seamless access.

---
🚀 **Built with passion for AI & Data Science!** 🎵📊

