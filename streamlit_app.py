import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("savedPickle/model.pkl")

# Load encoders
labelEncoders = joblib.load("savedPickle/label_encoders.pkl")
ordinalEncoders = joblib.load("savedPickle/ordinal_encoders.pkl")

def user_input_form():
    with st.form("loan_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
        income = st.number_input("Income", min_value=0.0, value=50000.0)
        exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
        home = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        loan_amnt = st.number_input("Loan Amount", min_value=5.0, value=300.0)
        intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5)
        percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.3)
        cred_hist_len = st.number_input("Credit History Length", min_value=0.0, value=10.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        prev_default = st.selectbox("Previous Loan Default?", ['No', 'Yes'])

        submitted = st.form_submit_button("Predict")
    
    if submitted:
        return {
            "person_age": age,
            "person_gender": gender,
            "person_education": education,
            "person_income": income,
            "person_emp_exp": exp,
            "person_home_ownership": home,
            "loan_amnt": loan_amnt,
            "loan_intent": intent,
            "loan_int_rate": interest_rate,
            "loan_percent_income": percent_income,
            "cb_person_cred_hist_length": cred_hist_len,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": prev_default
        }
    else:
        return None

def predict_loan(input_dict):
    df = pd.DataFrame([input_dict])

    df[["person_education", "previous_loan_defaults_on_file"]] = ordinalEncoders.transform(
        df[["person_education", "previous_loan_defaults_on_file"]]
    )

    for col, le in labelEncoders.items():
        df[col] = le.transform(df[col])

    pred = model.predict(df)[0]
    return pred

def main():
    st.title("📊 Loan Status Prediction App")
    st.write("Input your data below to predict your loan status.")

    user_data = user_input_form()

    if user_data:
        result = predict_loan(user_data)
        if result == 1:
            st.success("YOUR LOAN IS APPROVED")
        else: 
            st.error("YOUR LOAN IS REJECTED")

if __name__ == "__main__":
    main()
