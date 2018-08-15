import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataset = pd.read_csv("train.csv" , low_memory = False)

X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
y = dataset.iloc[:,17].values

info = dataset.info()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3:4] = labelencoder_X.fit_transform(X[:, 3:4]).fillna('0')

temp = np.array(X[:, 3:4])
for i in range(84191):
    X[:, 3:4][i] == X[:, 3:4][i][0]
    
    
    
    dfsdfss