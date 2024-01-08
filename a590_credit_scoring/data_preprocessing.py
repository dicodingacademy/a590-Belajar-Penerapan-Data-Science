import joblib
import numpy as np
import pandas as pd

# Load all the data preprocessing objects
encoder_Credit_Mix = joblib.load("model/encoder_Credit_Mix.joblib")
encoder_Payment_Behaviour = joblib.load("model/encoder_Payment_Behaviour.joblib")
encoder_Payment_of_Min_Amount = joblib.load("model/encoder_Payment_of_Min_Amount.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_Age = joblib.load("model/scaler_Age.joblib")
scaler_Amount_invested_monthly = joblib.load("model/scaler_Amount_invested_monthly.joblib")
scaler_Changed_Credit_Limit = joblib.load("model/scaler_Changed_Credit_Limit.joblib")
scaler_Credit_History_Age = joblib.load("model/scaler_Credit_History_Age.joblib")
scaler_Delay_from_due_date = joblib.load("model/scaler_Delay_from_due_date.joblib")
scaler_Interest_Rate = joblib.load("model/scaler_Interest_Rate.joblib")
scaler_Monthly_Balance =joblib.load("model/scaler_Monthly_Balance.joblib")
scaler_Monthly_Inhand_Salary = joblib.load("model/scaler_Monthly_Inhand_Salary.joblib")
scaler_Num_Bank_Accounts = joblib.load("model/scaler_Num_Bank_Accounts.joblib")
scaler_Num_Credit_Card = joblib.load("model/scaler_Num_Credit_Card.joblib")
scaler_Num_Credit_Inquiries = joblib.load("model/scaler_Num_Credit_Inquiries.joblib")
scaler_Num_of_Delayed_Payment = joblib.load("model/scaler_Num_of_Delayed_Payment.joblib")
scaler_Num_of_Loan = joblib.load("model/scaler_Num_of_Loan.joblib")
scaler_Outstanding_Debt = joblib.load("model/scaler_Outstanding_Debt.joblib")
scaler_Total_EMI_per_month = joblib.load("model/scaler_Total_EMI_per_month.joblib")

pca_numerical_columns_1 = [
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_History_Age'
]

pca_numerical_columns_2 = [
    "Monthly_Inhand_Salary",
    "Monthly_Balance", 
    "Amount_invested_monthly", 
    "Total_EMI_per_month"
]

def data_preprocessing(data):
    """PPreprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    df["Age"] = scaler_Age.transform(np.asarray(data["Age"]).reshape(-1,1))[0]
    
    df["Credit_Mix"] = encoder_Credit_Mix.transform(data["Credit_Mix"])[0]
    df["Payment_of_Min_Amount"] = encoder_Payment_of_Min_Amount.transform(data["Payment_of_Min_Amount"])
    df["Payment_Behaviour"] = encoder_Payment_Behaviour.transform(data["Payment_Behaviour"])
    
    # PCA 1
    data["Num_Bank_Accounts"] = scaler_Num_Bank_Accounts.transform(np.asarray(data["Num_Bank_Accounts"]).reshape(-1,1))[0]
    data["Num_Credit_Card"] = scaler_Num_Credit_Card.transform(np.asarray(data["Num_Credit_Card"]).reshape(-1,1))[0]
    data["Interest_Rate"] = scaler_Interest_Rate.transform(np.asarray(data["Interest_Rate"]).reshape(-1,1))[0]
    data["Num_of_Loan"] = scaler_Num_of_Loan.transform(np.asarray(data["Num_of_Loan"]).reshape(-1,1))[0]
    data["Delay_from_due_date"] = scaler_Delay_from_due_date.transform(np.asarray(data["Delay_from_due_date"]).reshape(-1,1))[0]
    data["Num_of_Delayed_Payment"] = scaler_Num_of_Delayed_Payment.transform(np.asarray(data["Num_of_Delayed_Payment"]).reshape(-1,1))[0]
    data["Changed_Credit_Limit"] = scaler_Changed_Credit_Limit.transform(np.asarray(data["Changed_Credit_Limit"]).reshape(-1,1))[0]
    data["Num_Credit_Inquiries"] = scaler_Num_Credit_Inquiries.transform(np.asarray(data["Num_Credit_Inquiries"]).reshape(-1,1))[0]
    data["Outstanding_Debt"] = scaler_Outstanding_Debt.transform(np.asarray(data["Outstanding_Debt"]).reshape(-1,1))[0]
    data["Credit_History_Age"] = scaler_Credit_History_Age.transform(np.asarray(data["Credit_History_Age"]).reshape(-1,1))[0]
    
    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5"]] = pca_1.transform(data[pca_numerical_columns_1])
    
    # PCA 2
    data["Monthly_Inhand_Salary"] = scaler_Monthly_Inhand_Salary.transform(np.asarray(data["Monthly_Inhand_Salary"]).reshape(-1,1))[0]
    data["Monthly_Balance"] = scaler_Monthly_Balance.transform(np.asarray(data["Monthly_Balance"]).reshape(-1,1))[0]
    data["Amount_invested_monthly"] = scaler_Amount_invested_monthly.transform(np.asarray(data["Amount_invested_monthly"]).reshape(-1,1))[0]
    data["Total_EMI_per_month"] = scaler_Total_EMI_per_month.transform(np.asarray(data["Total_EMI_per_month"]).reshape(-1,1))[0]
    
    df[["pc2_1", "pc2_2"]] = pca_2.transform(data[pca_numerical_columns_2])
    
    return df