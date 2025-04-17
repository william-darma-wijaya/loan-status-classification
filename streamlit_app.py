import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("savedPickle/model.pkl")

# Load encoders
labelEncoders = joblib.load("savedPickle/label_encoders.pkl")
ordinalEncoders = joblib.load("savedPickle/ordinal_encoders.pkl")

def main():
  st.title('Loan Status Prediction')
  st.write("Insert your data below, then click **PREDICT**")

if __name__ == "__main__":
  main()
