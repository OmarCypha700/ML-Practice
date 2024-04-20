print('Importing libraries....')
# Import required libraries
import numpy as np
import pandas as pd 
# For cross-validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# For fitting the model
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# For model evaluation
from sklearn.metrics import roc_auc_score
# Save the model
import pickle

# Parameters
C=1.0
n_split = 5
output_file = f'model_C={C}.bin'

# Data preparatio
print('Running data preparation...')
df = pd.read_csv('../data/Telco_Customer_Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(df.totalcharges.mean())
df.churn = (df.churn == 'yes').astype(int)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['monthlycharges', 'tenure', 'totalcharges']

categorical = ['gender',
               'seniorcitizen', 
               'partner', 
               'dependents', 
               'phoneservice', 
               'multiplelines', 
               'internetservice', 
               'onlinesecurity', 
               'onlinebackup', 
               'deviceprotection', 
               'techsupport', 
               'streamingtv', 
               'streamingmovies', 
               'contract', 
               'paperlessbilling', 
               'paymentmethod']

# Training
print('Training model...')
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model 

def predict(dv, df, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Validation
print(f'Running validation with C={C}...')
kfold = KFold(n_splits=n_split, shuffle=True, random_state=1)

scores = []
fold = 0

for train_index, validation_index in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_index]
    df_validation = df_train_full.iloc[validation_index]

    y_train = df_train.churn
    y_val = df_validation.churn

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(dv, df_validation, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold}= {auc:.3f}')
    fold += 1
print('Validation results')
print('C=%s %.3f +- %.3f'%(C, np.mean(scores), np.std(scores)))

# Train final model
print('Training final model...')
dv, model = train(df_train_full, df_train_full.churn.values, C=1.0)
y_pred = predict(dv, df_test, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc
print(f'auc of final model: %.3f' % auc)

# Save model
print('Saving model...')
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f'Model saved to {output_file}')





