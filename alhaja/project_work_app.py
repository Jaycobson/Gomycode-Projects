import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
print("Current Working Directory:", os.getcwd())
# data = pd.read_csv(r"c:/Users/USER/Downloads/project_work_app.py",on_bad_lines='skip')

# Load the model, encoders, and scaler

with open('Mothers_Qualification_encoder.pkl', 'rb') as file:
    Mothers_Qualification_encoder = pickle.load(file)

with open("SVR_model.pkl", "rb") as file:
    model = pickle.load(file)

with open('School_encoder.pkl', 'rb') as file:
    School_encoder = pickle.load(file)

with open('Gender_encoder.pkl', 'rb') as file:
    Gender_encoder = pickle.load(file)

with open('Age_encoder.pkl', 'rb') as file:
    Age_encoder = pickle.load(file)

with open('Address_encoder.pkl', 'rb') as file:
    Address_encoder = pickle.load(file)

with open('Family_Size_encoder.pkl', 'rb') as file:
    Family_Size_encoder = pickle.load(file)

with open('Parent_Status_encoder.pkl', 'rb') as file:
    Parent_Status_encoder = pickle.load(file)

with open('Fathers_Qualification_encoder.pkl', 'rb') as file:
    Fathers_Qualification_encoder = pickle.load(file)

with open('Mothers_Occupation_encoder.pkl', 'rb') as file:
    Mothers_Occupation_encoder = pickle.load(file)

with open('Fathers_Occupation_encoder.pkl', 'rb') as file:
    Fathers_Occupation_encoder = pickle.load(file)

with open('Extra_educational_support_encoder.pkl', 'rb') as file:
    Extra_educational_support_encoder = pickle.load(file)

with open('Family_Support_encoder.pkl', 'rb') as file:
    Family_Support_encoder = pickle.load(file)

with open('Extra_Class_paid_encoder.pkl', 'rb') as file:
    Extra_Class_paid_encoder = pickle.load(file)

with open('Extracurricula_encoder.pkl', 'rb') as file:
    Extracurricula_encoder = pickle.load(file)

with open('Internet_encoder.pkl', 'rb') as file:
    Internet_encoder = pickle.load(file)

with open('Freetime_encoder.pkl', 'rb') as file:
    Freetime_encoder = pickle.load(file)

# with open('Health_encoder.pkl', 'rb') as file:
#     Health_encoder = pickle.load(file)

with open('Absences_encoder.pkl', 'rb') as file:
    Absences_encoder = pickle.load(file)

with open('Final_Grade_encoder.pkl', 'rb') as file:
    Final_Grade_encoder = pickle.load(file)

def main():
    st.title('Student Performance Dataset Exploration')
    
    # Summary statistics
    st.subheader('Summary Statistics')
    # st.write(data.describe())
    
    # Visualizations
    st.subheader('Visualizations')

    School = st.sidebar.selectbox('School', ['GP', 'MP'])
    Gender = st.sidebar.selectbox('Gender', ('F', 'M'))
    Age = st.sidebar.number_input('Age', min_value=15, max_value=21, step=1)
    Address = st.sidebar.radio('Address', ('U', 'R'))
    Family_Size = st.sidebar.selectbox('Family_Size', ('GT3', 'LE3'))
    Parent_Status = st.sidebar.radio('Parent_Status', ('A', 'T'))
    Mothers_Qualification = st.sidebar.selectbox('Mothers_Qualification', ('High School', 'College', 'Graduate', 'Others'))
    Fathers_Qualification = st.sidebar.multiselect('Fathers_Qualification', ('High School', 'College', 'Graduate', 'Others'))
    Mothers_Occupation = st.sidebar.selectbox('Mothers_Occupation', ('other', 'at_home', 'teacher', 'health'))
    Fathers_Occupation = st.sidebar.selectbox('Fathers_Occupation', ('other', 'at_home', 'teacher', 'health'))
    Studytime = st.sidebar.number_input('Studytime', min_value=1, max_value=4, step=1)
    Extra_educational_support = st.sidebar.selectbox('Extra_educational_support', ('yes', 'no'))
    Family_Support = st.sidebar.selectbox('Family_Support', ('yes', 'no'))
    Extracurricula = st.sidebar.selectbox('Extracurricula', ('yes', 'no'))
    Internet = st.sidebar.selectbox('Internet', ('yes', 'no'))
    Freetime = st.sidebar.selectbox('Freetime', ('1', '2', '3', '4'))
    Absences = st.sidebar.number_input('Absences', min_value=0, max_value=75, step=1)
    # Final_Grade = st.sidebar.number_input('Final_Grade', min_value=0, max_value=20, step=1)

    School_encoded = School_encoder.transform([School])[0]
    Gender_encoded = Gender_encoder.transform([Gender])[0]
    Address_encoded = Address_encoder.transform([Address])[0]
    Family_Size_encoded = Family_Size_encoder.transform([Family_Size])[0]
    Parent_Status_encoded = Parent_Status_encoder.transform([Parent_Status])[0]
    Mothers_Qualification_encoded = Mothers_Qualification_encoder.transform([Mothers_Qualification])[0]
    Fathers_Qualification_encoded = [Fathers_Qualification_encoder.transform([f])[0] for f in Fathers_Qualification]
    Mothers_Occupation_encoded = Mothers_Occupation_encoder.transform([Mothers_Occupation])[0]
    Fathers_Occupation_encoded = Fathers_Occupation_encoder.transform([Fathers_Occupation])[0]
    Family_Support_encoded = Family_Support_encoder.transform([Family_Support])[0]
    Extra_educational_support_encoded = Extra_educational_support_encoder.transform([Extra_educational_support])[0]
    Internet_encoded = Internet_encoder.transform([Internet])[0]

    # 'REGION': REGION_encoded,
    dataset = {
        'School': School_encoded,
        'Gender': Gender_encoded,
        'Age': Age,
        'Address': Address_encoded,
        'Family_Size': Family_Size_encoded,
        'Parent_Status': Parent_Status_encoded,
        'Mothers_Qualification': Mothers_Qualification_encoded,
        'Fathers_Qualification': Fathers_Qualification_encoded,
        'Mothers_Occupation': Mothers_Occupation_encoded,
        'Fathers_Occupation': Fathers_Occupation_encoded,
        'Studytime': Studytime,
        'Extra_educational_support': Extra_educational_support_encoded,
        'Family_Support': Family_Support_encoded,
        'Extracurricula': Extracurricula,
        'Internet': Internet_encoded,
        'Freetime': Freetime,
        'Absences': Absences,
        # 'Final_Grade': Final_Grade,
    }
    input_dataset = pd.DataFrame(dataset)
    

    # Assume the scaler has been fit previously
    # input_scaled = Scaler.transform(input_dataset)
    input_scaled = input_dataset #remove


    st.subheader('User Input parameters')
    st.write(input_dataset)

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(input_scaled)
        st.write(f'Prediction: {prediction[0]}')   

    # # Scatter plot of Studytime vs Final_Grade colored by Gender
    # fig_scatter = px.scatter(data, x='Studytime', y='Final_Grade', color='Gender', 
    #                          title='Study Time vs Final Grade', 
    #                          labels={'Studytime': 'Study Time', 'Final_Grade': 'Final Grade'})
    # st.plotly_chart(fig_scatter)
    
    # # Bar plot of Mothers_Qualification
    # fig_bar = px.bar(data, x='Mothers_Qualification', y='Final_Grade', 
    #                  title='Average Final Grade by Mothers Qualification',
    #                  labels={'Mothers_Qualification': 'Mothers Qualification', 'Final_Grade': 'Final Grade'})
    # st.plotly_chart(fig_bar)
    
    # # Bar plot for Health
    # fig_bar = px.bar(data, x='Health', y='Final_Grade', 
    #                  title='Average Final Grade by Health',
    #                  labels={'Health': 'Health Check', 'Final_Grade': 'Final Grade'})
    # st.plotly_chart(fig_bar)
    
    # # Histogram of Ages
    # fig_hist = px.histogram(data, x='Age', nbins=10, title='Age Distribution')
    # st.plotly_chart(fig_hist)

if __name__ == "__main__":
    main()