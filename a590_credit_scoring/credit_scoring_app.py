import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing, encoder_Credit_Mix, encoder_Payment_Behaviour, encoder_Payment_of_Min_Amount
from prediction import prediction

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=130)
with col2:
    st.header('Credit Scoring App (Prototype) :sparkles:')

data = pd.DataFrame()

col1, col2, col3 = st.columns(3)

with col1:
    Credit_Mix = st.selectbox(label='Credit_Mix', options=encoder_Credit_Mix.classes_, index=1)
    data["Credit_Mix"] = [Credit_Mix]

with col2:
    Payment_of_Min_Amount = st.selectbox(label='Payment_of_Min_Amount', options=encoder_Payment_of_Min_Amount.classes_, index=1)
    data["Payment_of_Min_Amount"] = [Payment_of_Min_Amount]

with col3:
    Payment_Behaviour = st.selectbox(label='Payment_Behaviour', options=encoder_Payment_Behaviour.classes_, index=5)
    data["Payment_Behaviour"] = Payment_Behaviour

col1, col2, col3, col4 = st.columns(4)

with col1:
    # st.header("Kolom 1")
    Age = int(st.number_input(label='Age', value=23))
    data["Age"] = Age

with col2:
    Num_Bank_Accounts = int(st.number_input(label='Num_Bank_Accounts', value=3))
    data["Num_Bank_Accounts"] = Num_Bank_Accounts

with col3:
    Num_Credit_Card = int(st.number_input(label='Num_Credit_Card', value=4))
    data["Num_Credit_Card"] = Num_Credit_Card

with col4:
    Interest_Rate = float(st.number_input(label='Interest_Rate', value=3))
    data["Interest_Rate"] = Interest_Rate


col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_of_Loan = int(st.number_input(label='Num_of_Loan', value=4))
    data["Num_of_Loan"] = Num_of_Loan

with col2:
    # st.header("Kolom 1")
    Delay_from_due_date = int(st.number_input(label='Delay_from_due_date', value=3))
    data["Delay_from_due_date"] = Delay_from_due_date

with col3:
    Num_of_Delayed_Payment = int(st.number_input(label='Num_of_Delayed_Payment', value=7))
    data["Num_of_Delayed_Payment"] = Num_of_Delayed_Payment

with col4:
    Changed_Credit_Limit = float(st.number_input(label='Changed_Credit_Limit', value=11.27))
    data["Changed_Credit_Limit"] = Changed_Credit_Limit

col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_Credit_Inquiries = float(st.number_input(label='Num_Credit_Inquiries', value=5))
    data["Num_Credit_Inquiries"] = Num_Credit_Inquiries

with col2:
    Outstanding_Debt = float(st.number_input(label='Outstanding_Debt', value=809.98))
    data["Outstanding_Debt"] = Outstanding_Debt

with col3:
    Monthly_Inhand_Salary = float(st.number_input(label='Monthly_Inhand_Salary', value=1824.8))
    data["Monthly_Inhand_Salary"] = Monthly_Inhand_Salary

with col4:
    Monthly_Balance = float(st.number_input(label='Monthly_Balance', value=186.26))
    data["Monthly_Balance"] = Monthly_Balance

col1, col2, col3 = st.columns(3)

with col1:
    Amount_invested_monthly = float(st.number_input(label='Amount_invested_monthly', value=236.64))
    data["Amount_invested_monthly"] = Amount_invested_monthly

with col2:
    Total_EMI_per_month = float(st.number_input(label='Total_EMI_per_month', value=49.5))
    data["Total_EMI_per_month"] = Total_EMI_per_month

with col3:
    Credit_History_Age = float(st.number_input(label='Credit_History_Age', value=216))
    data["Credit_History_Age"] = Credit_History_Age

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Credit Scoring: {}".format(prediction(new_data)))