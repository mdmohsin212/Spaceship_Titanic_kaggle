import pandas as pd
import pickle
from xgboost import XGBClassifier
import yaml

train_data = pd.read_csv('data/features/train.csv')

x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Load Parameter
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
model_params = params['xgboost_params']

# Train Model
model = XGBClassifier(**model_params)
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))