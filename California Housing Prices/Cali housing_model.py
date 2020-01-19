# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:13:24 2020

@author: Omkar R
"""

import pandas as pd
import pickle
dat=sklearn.datasets.fetch_california_housing()

d=pd.DataFrame(dat.data,columns=dat.feature_names)
d['Target']=pd.Series(dat.target)

d2=d[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
        'Target']]


X = d2.iloc[:, :6]

y = d2.iloc[:, -1]


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
     
print(model.predict([[8.325,41.0,6.9841,1.0238,322.0,2]]))