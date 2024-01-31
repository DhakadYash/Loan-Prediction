from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'index.html')

def calculator(request):
    return render(request, 'calculator.html')

def result(request):
    if request.method == "POST":
        import numpy as np
        import pandas as pd
        from sklearn import svm
        import joblib

        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import StandardScaler

        # Define a mapping for converting categorical values to numerical labels
        categorical_mapping = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'mstatus': {'Yes': 1, 'No': 0},
            'education': {'Graduate': 1, 'Not Graduate': 0},
            'profession': {'Yes': 1, 'No': 0},
            'parea': {'Urban': 1, 'Rural': 0},
            'loan': {'N': 0, 'Y': 1}
        }

        lis = [
            categorical_mapping['gender'].get(request.POST.get('gender', 'Male'), 0),
            categorical_mapping['mstatus'].get(request.POST.get('mstatus', 'No'), 0),
            float(request.POST.get('dependents', 0)),
            categorical_mapping['education'].get(request.POST.get('education', 'Not Graduate'), 0),
            float(request.POST.get('income', 0)),
            float(request.POST.get('cincome', 0)),
            float(request.POST.get('amount', 0)),
            float(request.POST.get('term', 0)),
            float(request.POST.get('chistory', 0)),
            ]

        df = pd.DataFrame([lis], columns=['Gender', 'Married', 'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'])

        df['loanamount_log']=np.log(df['LoanAmount'])

        df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']

        df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
        df['Married'].fillna(df['Married'].mode()[0],inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

        df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
        df.loanamount_log = df.loanamount_log.fillna(df.loanamount_log.mean())

        x = df.iloc[:,np.r_[0:4,7:11]].values
        
        LabelEncoder_X = LabelEncoder()

        for i in range(0,6):
            x[:,i]= LabelEncoder_X.fit_transform(x[:,i])

        ss = StandardScaler()

        x = ss.fit_transform(x)

        rf_mod = joblib.load('rf_model.sav')
        nb_mod = joblib.load('nb_model.sav')
        dt_mod = joblib.load('dt_model.sav')

        rfa = nb_mod.predict(x)
        nba = nb_mod.predict(x)
        dta = nb_mod.predict(x)

        return render(request, 'result.html', {'rfa': rfa, 'nba': nba, 'dta': dta})

    return render(request, 'result.html')