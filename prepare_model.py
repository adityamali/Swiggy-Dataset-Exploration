import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
df_swiggy = pd.read_csv('./dataset/swiggy.csv')

df = df_swiggy.copy()
df['rating'] = df['rating'].str.replace('--', '0')
df['cost'] = df['cost'].str.replace('₹', '')
df['rating_count'] = df['rating_count'].str.replace('+','').str.replace('K','000').str.replace('Too Few Ratings', '0').str.replace('ratings', '')

df['rating'] = df['rating'].astype(float)
df['rating_count'] = df['rating_count'].fillna(0).astype(int)
df['cost'] = df['cost'].fillna(0).astype(int)


df = df.dropna()
df = df[df['rating_count'] > 0]

# Data cleaning
df = df.dropna()
df = df[df['rating_count'] > 0]

# Feature engineering
df['cost_per_person'] = df['cost'] / 2
df['popularity'] = df['rating'] * np.log1p(df['rating_count'])

# Encode categorical variables
le_city = LabelEncoder()
le_cuisine = LabelEncoder()
df['city_encoded'] = le_city.fit_transform(df['city'])
df['cuisine_encoded'] = le_cuisine.fit_transform(df['cuisine'])

# Select features for the model
features = ['city_encoded', 'cuisine_encoded', 'rating', 'rating_count', 'popularity']
target = 'cost_per_person'

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save all necessary files
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(le_city, 'le_city.pkl')
joblib.dump(le_cuisine, 'le_cuisine.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("All files saved successfully.")