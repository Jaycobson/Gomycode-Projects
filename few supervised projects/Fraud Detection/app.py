import pickle
import numpy as np 
import streamlit as st
import pandas as pd 

# Load data, model, and encoders
df = pd.read_csv(r'C:\Users\HP\Downloads\Project_folder\fraud test.csv')

with open(r'C:\Users\HP\Downloads\Project_folder\model.pkl', 'rb') as file:
    LGBMClassifier_model = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\merchant_encoder.pkl', 'rb') as file:
    merchant_encoder = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\category_encoder.pkl', 'rb') as file:
    category_encoder = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\city_encoder.pkl', 'rb') as file:
    city_encoder = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\state_encoder.pkl', 'rb') as file:
    state_encoder = pickle.load(file)

with open(r'C:\Users\HP\Downloads\Project_folder\job_encoder.pkl', 'rb') as file:
    job_encoder = pickle.load(file)

print('Successful code')

def main():
    st.title('Fraud Detection Application')
    st.write('Developed By: Austin')
    st.write('Prediction features')
    
    st.sidebar.header('Enter your features here')
    
    # User input
    st.sidebar.subheader('Enter your details here')

    # Categorical features
    gender = st.sidebar.selectbox('Gender', gender_encoder.classes_)
    job = st.sidebar.selectbox('Job', job_encoder.classes_)
    category = st.sidebar.selectbox('Category', category_encoder.classes_)
    city = st.sidebar.selectbox('City', city_encoder.classes_)
    state = st.sidebar.selectbox('State', state_encoder.classes_)
    merchant = st.sidebar.selectbox('Merchant', merchant_encoder.classes_)

    # Numerical features
    age = st.sidebar.number_input('Age', min_value=18, max_value=99)
    Amount_withdraw = st.sidebar.number_input('Amount', min_value=9, max_value=3000)
    latitude = st.sidebar.slider('Latitude', min_value=20.0271, max_value=65.6899)
    longitude = st.sidebar.slider('Longitude', min_value=-165.6723, max_value=-67.9503)
    merchant_lat = st.sidebar.number_input('Merchant Latitude', min_value=19.163455, max_value=65.951727)
    merchant_long = st.sidebar.slider('Merchant Longitude', min_value=-166.464422, max_value=-66.960745)
    
    # Creating a DataFrame with user inputs
    data = {
         'merchant': [merchant],
          'category': [category],
           'amt': [Amount_withdraw],  # Ensure this matches the name used during training
             'gender': [gender],
             'city': [city],
              'state': [state],  
        'lat': [latitude],  # Ensure this matches the name used during training
        'long': [longitude],  # Ensure this matches the name used during training
         'job': [job],
          'merch_lat': [merchant_lat],
        'merch_long': [merchant_long],  # Ensure this matches the name used during training
        'age': [age],
    }

    input_df = pd.DataFrame(data, index = [0])
    
    # Ensure feature names match those used during training
    # Example renaming if needed
    input_df.rename(columns={
        'amount_withdrawn': 'amt',  # Rename to match training feature name
        'latitude': 'lat',
        'longitude': 'lng',
        'merchant_long': 'merchant_lng'
        # Add more renames as necessary
    }, inplace=True)

    # Encoding categorical features
    input_df['gender'] = gender_encoder.transform(input_df['gender'])
    input_df['job'] = job_encoder.transform(input_df['job'])
    input_df['category'] = category_encoder.transform(input_df['category'])
    input_df['city'] = city_encoder.transform(input_df['city'])
    input_df['state'] = state_encoder.transform(input_df['state'])
    input_df['merchant'] = merchant_encoder.transform(input_df['merchant'])

    # Scaling numerical features
    scaled_features = scaler.transform(input_df)
    
    if st.button('Predict'):
        # Making prediction
        prediction = LGBMClassifier_model.predict(scaled_features)
        prediction_prob = LGBMClassifier_model.predict_proba(scaled_features)
        print(prediction)
        st.subheader('Prediction')
        st.write('Fraud' if prediction[0] == 1 else 'Not Fraud')
        st.write(f'Probability of Fraud: {prediction_prob[0][1]:.2f}')
    
    
if __name__ == "__main__":
    main()
