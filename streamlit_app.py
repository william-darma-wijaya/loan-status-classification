import streamlit as st
import joblib
import pandas as pd
st.title('Loan Status Prediction')

# Load model
model = joblib.load("savedPickle/model.pkl")

# Load encoders
labelEncoders = joblib.load("savedPickle/label_encoders.pkl")
ordinalEncoders = joblib.load("savedPickle/ordinal_encoders.pkl")

