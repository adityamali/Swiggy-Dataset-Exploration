import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import os

st.title('Restaurant Dish Price Predictor')

# Check if all required files exist
required_files = ['random_forest_model.joblib', 'le_city.pkl', 'le_cuisine.pkl', 'scaler.pkl']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"Error: The following files are missing: {', '.join(missing_files)}")
    st.info("Please run the 'prepare_model.py' script to generate these files before using this app.")
else:
    # Load the model and preprocessing objects
    model = joblib.load('random_forest_model.joblib')
    le_city = joblib.load('le_city.pkl')
    le_cuisine = joblib.load('le_cuisine.pkl')
    scaler = joblib.load('scaler.pkl')

    # Input fields
    city = st.selectbox('Select City', le_city.classes_)
    cuisine = st.selectbox('Select Cuisine', le_cuisine.classes_)
    rating = st.slider('Rating', 0.0, 5.0, 3.0, 0.1)
    rating_count = st.number_input('Number of Ratings', min_value=1, value=100)

    # Calculate popularity
    popularity = rating * np.log1p(rating_count)

    # Encode and scale input
    input_data = np.array([[
        le_city.transform([city])[0],
        le_cuisine.transform([cuisine])[0],
        rating,
        rating_count,
        popularity
    ]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    if st.button('Predict Price'):
        prediction = model.predict(input_scaled)[0]
        st.success(f'The predicted best price for a dish is: ₹{prediction:.2f}')

    st.write('Note: This prediction is based on the cost per person.')