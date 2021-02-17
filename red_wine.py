from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pandas import ConfusionMatrix

# loading Dataset

dataset = pd.read_csv("winequality-red.csv", sep=';', header=0)

feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']

# define array

X = dataset[feature_names]
y = dataset.quality

# separate data
'''separates data in 80% for training
 and 20% for validation'''

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=0)


# evaluate model performance

def get_score(n_estimators):
    ''' get mean of scores for five fold cross validation for ten different values
        for the number of trees in Random Forest Regressor '''

    scores = -1 * cross_val_score(RandomForestRegressor(n_estimators, random_state=0), X, y,
                                  cv=5, scoring='neg_mean_absolute_error')

    return scores.mean()


# fit model

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

# make prediction
predictions = model.predict(X)
MAE = mean_absolute_error(y, predictions)
print('MAE for Random Forest regression is : {}'.format(MAE))

confusion_matrix = ConfusionMatrix(y, predictions)
print("Confusion matrix:\n%s" % confusion_matrix)


