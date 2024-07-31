import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
print("Current Working Directory:", os.getcwd())
# data = pd.read_csv(r"c:/Users/USER/Downloads/project_work_app.py",on_bad_lines='skip')

# Load the model, encoders, and scaler
with open("SVR_model.pkl", "rb") as file:
    model = pickle.load(file)
