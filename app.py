import streamlit as st
import pandas as pd
from sklearn import joblib  # For scikit-learn version < 0.23
# If using scikit-learn version >= 0.23, use:
# from sklearn import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Function for data preprocessing
def preprocess_data(df):
    # One-hot encode categorical variables
    df = pd.get_dummies(df)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Scale numerical features
    columns_to_scale = ["registered_year", "engine_capacity", "kms_driven", "max_power", "mileage"]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df

# Streamlit app
st.title('Car Price Prediction App')

# Input form
registered_year = st.number_input('Registered Year', min_value=1900, max_value=2024, value=2020)
engine_capacity = st.number_input('Engine Capacity (cc)', min_value=300, max_value=10000, value=1300)
insurance = st.selectbox('Insurance Type', ['Comprehensive', 'Not Available', 'Third Party', 'Zero Dep'])
transmission_type = st.selectbox('Transmission Type', ['Automatic', 'Manual'])
kms_driven = st.number_input('Kilometers Driven', min_value=0, value=100000)
owner_type = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner','Fifth Owner'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG' , 'Electric', 'LPG'])
max_power = st.number_input('Max Power (bhp)', min_value=0, value=0)
seats = st.number_input('Number of Seats', min_value=1, value=1)
mileage = st.number_input('Mileage (kmpl)', min_value=0, value=0)
body_type = st.selectbox('Body Type', ['Convertibles', 'Hatchback', 'Sedan', 'SUV', 'Minivans', 'Coupe', 'Pickup', 'Wagon'])
city = st.selectbox('City', ['Agra', 'Gurgaon', 'Lucknow', 'Delhi', 'Chandigarh', 'Bangalore', 'Jaipur', 'Kolkata', 'Ahmedabad', 'Chennai', 'Pune', 'Mumbai', 'Hyderabad'])
brand = st.selectbox('Brand', ['Audi', 'BMW', 'Chevrolet', 'Citroen', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'] )

# Prediction button
if st.button('Predict'):
    input_data = preprocess_data(pd.DataFrame({
        'registered_year': [registered_year],
        'engine_capacity': [engine_capacity],
        'insurance': [insurance],
        'transmission_type': [transmission_type],
        'kms_driven': [kms_driven],
        'owner_type': [owner_type],
        'fuel_type': [fuel_type],
        'max_power': [max_power],
        'seats': [seats],
        'mileage': [mileage],
        'body_type': [body_type],
        'city': [city],
        'brand': [brand]
    }))

    # Make predictions using the trained model
    prediction = model.predict(input_data)[0]

    st.success(f'The predicted car resale price is {prediction} INR')
