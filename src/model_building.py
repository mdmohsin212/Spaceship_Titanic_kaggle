import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv('data/features/train.csv')

x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

model = RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=50)
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))