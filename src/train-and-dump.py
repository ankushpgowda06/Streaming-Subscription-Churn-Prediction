# importing required package
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import boto3

# loading dataset, training and dumping models
df = pd.read_csv("dataset/train.csv")
df = df.drop(columns = ["customer_id", 'num_platform_friends', 'average_session_length', 'num_playlists_created',
                    'weekly_songs_played', 'signup_date', 'num_shared_playlists', 'num_favorite_artists',
                    'location', 'payment_method', 'payment_plan', 'notifications_clicked', 'weekly_unique_songs'])

encoders = {}
categorical_features = ["subscription_type", "customer_service_inquiries"]

for column in categorical_features:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

X = df.iloc[:, :-1]
y = df.iloc[:,-1]

param_grid_xgb = {
    'n_estimators': randint(100, 500),      
    'learning_rate': uniform(0.01, 0.19),   
    'max_depth': randint(3, 10),          
    'subsample': uniform(0.6, 0.4),        
    'colsample_bytree': uniform(0.6, 0.4),    
    'gamma': randint(0, 5),              
    'lambda': randint(0, 10),            
    'alpha': randint(0, 10)               
}

random_search = RandomizedSearchCV(XGBClassifier(random_state=42), param_grid_xgb, n_iter=20, cv=8, scoring='accuracy', n_jobs=-1)
random_search.fit(X, y)
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

final_model = XGBClassifier(**random_search.best_params_, random_state=42)
final_model.fit(X, y)

joblib.dump(final_model, "models/churn_prediction_model.pkl")
joblib.dump(encoders, 'models/encoders.pkl')

print("Final model trained and saved successfully!")

s3 = boto3.client("s3")
bucket_name = "streaming-churn-bucket"
s3.upload_file("models/churn_prediction_model.pkl", bucket_name, "churn_prediction_model.pkl")
s3.upload_file("models/encoders.pkl", bucket_name, "encoders.pkl")

print("Stored all the models in AWS s3 bucket")



