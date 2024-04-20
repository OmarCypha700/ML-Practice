
import pickle
## Load the model
print('loading the model...')
input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f_in:
    dv,model = pickle.load(f_in)

customer = {'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'yes',
 'phoneservice': 'yes',
 'multiplelines': 'no',
 'internetservice': 'dsl',
 'onlinesecurity': 'yes',
 'onlinebackup': 'no',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'mailed_check',
 'monthlycharges': 49.95,
 'tenure': 13,
 'totalcharges': 587.45}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
print(f'input = {customer}')
print(f"Churn probability: {y_pred:.2f}")