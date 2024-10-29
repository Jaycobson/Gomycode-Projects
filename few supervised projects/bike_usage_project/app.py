import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px


data = pd.read_csv(r'C:\Users\User\Desktop\MY_PROJECT\train.csv\train.csv')
# Load pre-trained models and encoders
def load_model_and_encoder():
    with open(r'C:\Users\User\Desktop\MY_PROJECT\stored_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(r'C:\Users\User\Desktop\MY_PROJECT\stored_encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    with open(r'C:\Users\User\Desktop\MY_PROJECT\stored_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, encoder, scaler

def main():
    # Load models and encoders
    model, encoder, scaler = load_model_and_encoder()

    # Define Streamlit application
    st.title('PREDICTION FOR OLA LOGISTICS APP')
    st.image(r"C:\Users\User\Desktop\olabike.png")
    st.title('App Developed by: OLADIMEJI.S.ADEWUYI')
    st.write('Fill in the details below to get a prediction.')

    #fig = px.bar(data, x='stored+AF8-flag', y='total+AF8-amount', title='Total-Amount vs Stored-flag')
    #st.plotly_chart(fig)

    # Input fields for the user
    vendorid = st.sidebar.selectbox('Vendor ID', [1, 2])  # Assuming Vendor ID is 1 or 2
    drivertip = st.sidebar.number_input('Driver Tip ($)', min_value=0.0, max_value=100.0, step=1.0)
    mtatax = st.sidebar.number_input('MTA Tax ($)', min_value=0.0, max_value=100.0, step=1.0)
    distance = st.sidebar.number_input('Distance (miles)', min_value=0.0, max_value=1000.0, step=0.1)
    numpassengers = st.sidebar.number_input('Number of Passengers', min_value=1, max_value=10, step=1)
    tollamount = st.sidebar.number_input('Toll Amount ($)', min_value=0.0, max_value=100.0, step=1.0)
    paymentmethod = st.sidebar.number_input('Payment Method', min_value=0.0, max_value=4.0, step=1.0)
    ratecode = st.sidebar.selectbox('Rate Code', [1, 2, 3, 4, 5])  # Assuming these are the possible values
    storedflag = st.sidebar.selectbox('Stored Flag', ['N', 'Y'])  # Assuming 'N' and 'Y' are the possible values
    extracharges = st.sidebar.number_input('Extra Charges ($)', min_value=0.0, max_value=100.0, step=1.0)
    #improvementcharge = st.sidebar.number_input('Improvement Charge ($)', min_value=0.0, max_value=1.0, step=1.0)

    # Dummy date inputs for pickup and drop time
    pickup_time = st.date_input('Pickup Time')
    drop_time = st.date_input('Drop Time')

    # Convert date inputs to datetime
    pickup_time = pd.to_datetime(pickup_time)
    drop_time = pd.to_datetime(drop_time)

    # Calculate delivery time spent
    deliv_time_spent = (drop_time - pickup_time).total_seconds() / 60

    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        'vendorid': [vendorid],
        'drivertip': [drivertip],
        'mtatax': [mtatax],
        'distance': [distance],
        'numpassengers': [numpassengers],
        'tollamount': [tollamount],
        'paymentmethod': [paymentmethod],
        'ratecode': [ratecode],
        'storedflag': [storedflag],
        'extracharges': [extracharges],
        #'improvementcharge': [improvementcharge],
        'deliv_time_spent': [deliv_time_spent]
    })

    # Encode categorical features
    input_data['storedflag'] = encoder.transform(input_data['storedflag'])[0]

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    if st.button('Predict result'):
        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        st.write(f'The predicted total amount is: ${prediction[0]:.2f}')

# Run the app
if __name__ == "__main__":
    main()
