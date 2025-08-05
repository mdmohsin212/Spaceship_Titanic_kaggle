import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv('data/features/train.csv')

x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

model = RandomForestClassifier(n_estimators=145, criterion='entropy')
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))