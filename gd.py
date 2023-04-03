#gold detection prediction machine learning

#import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading the csv data to a pandas DataFrame

data = pd.read_csv("gld_price_data.csv")
print(data)

print(data.head())
print(data.tail())
print(data.shape)

print(data.info())

n = data.isnull().sum()
print(n)

print(data.describe())

# correlation

corr = data.corr()
print(corr)

# heatmap data visualization for understanding

#plt.figure(figsize = (8,8))
#sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot = True, annot_kws={'size':8}, cmap='Blues')

#correlation values of gld
print(corr['GLD'])

# checking the distribution of GLF price
d = sns.distplot(data['GLD'], color='orange')
print(d)

# features and target
X = data.drop(['Date','GLD'],axis=1)
Y = data['GLD']

print(X)
print(Y)

#sptlitting into trianing data and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

# model traingin Randome Forest Regressor
model = RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)

# prediction on test data
pred = model.predict(x_test)
print(pred)

y_test = list(y_test)

plt.plot(y_test, color='blue', label = 'Actual Value')
plt.plot(pred, color='green', label='Prediction Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()