import pandas as pd
import pickle
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

test_data = pd.read_csv('data/features/test.csv')
model = pickle.load(open('model.pkl', 'rb'))

x_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)


metrics_dict = {
    'accuracy' : accuracy,
    'precision' : precision,
    'recall' : recall,
    'auc' : auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)