import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import sys
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
import shap 
import pickle
import joblib


#import and prepare data
data=pd.read_excel('~/Desktop/projecten/voetbalspelers/testfile_fifa.xlsx')
data=data[data['player_positions']=='CB']
data.dropna(axis = 1, how = 'any', inplace = True)
data.dropna(axis = 0, how = 'any', inplace = True)
data=data[data['value_eur']!=0]
X=data.drop(['value_eur','player_positions','defending'], axis=1)
y=data['value_eur']
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)
print(X_train.shape)

#Train random forest regressor
rf = RandomForestRegressor(max_depth=3, random_state=0)
rf = rf.fit(X_train, y_train)

# Fit Shap Kernel Explainer
explainer = shap.Explainer(rf.predict, X_train)
shap_values = explainer(X_train)

#Save model as pickle 
filename = 'fifa_model.sav'
pickle.dump(rf, open(filename, 'wb')) #write binary

#Save explainer as pickle
ex_filename = 'explainer.bz2'
joblib.dump(explainer, filename=ex_filename, compress=('bz2', 9))
