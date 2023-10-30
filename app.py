import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model and data
model = pickle.load(open('LinearRegression.pkl', 'rb'))
car = pd.read_csv('Cleaned Car.csv')

st.title('Car Price Predictor')

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
year = sorted(car['year'].unique(), reverse=True)
fuel_type = car['fuel_type'].unique()

companies.insert(0, 'Select Company')

# Create widgets for user input
selected_company = st.selectbox('Select the company:', companies)
selected_car_model = st.selectbox('Select the model:', car_models)
selected_year = st.selectbox('Select Year of Purchase:', year)
selected_fuel_type = st.selectbox('Select the Fuel Type:', fuel_type)
kilometers_driven = st.text_input('Enter the Number of Kilometres that the car has traveled:')

# Define a function to make predictions
def predict_price():
    input_data = np.array([selected_car_model, selected_company, selected_year, kilometers_driven, selected_fuel_type]).reshape(1, 5)
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=input_data))
    return round(prediction[0], 2)

# Display the prediction result
if st.button('Predict Price'):
    prediction_result = predict_price()
    st.write(f'Prediction: ₹{prediction_result}')

st.write('This app predicts the price of a car you want to sell. Fill in the details above.')

# Note: Streamlit will automatically run the app when you run this script.
