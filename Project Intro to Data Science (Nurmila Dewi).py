#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys


import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

%config InlineBackend.figure_format ='svg'



#read data 
USA_Cars = pd.read_csv('USA_cars_datasets.csv')

#display
print(USA_Cars)
USA_Cars.head()

#last row, to make sure 
USA_Cars.tail()

#check shape 
USA_Cars.shape

#display the las 5 rows 
USA_Cars.tail()



#untuk Linear regression dipisahkan dulu
feature_cols=['year','lot','mileage']

#
x = USA_Cars[feature_cols]

#
x = USA_Cars[['year','lot','mileage']]
print(type(x))
x.shape

y = USA_Cars['year']
y = USA_Cars.price
print(type(y))
y.shape


#train data 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

print (lr.intercept_)
print (lr.coef_)

#pair featurename coef 
list(zip(feature_cols, lr.coef_))

#making prediction
y_pred = lr.predict(x_test)




#calculate 
import numpy as np 
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#feature selection 
feature_cols = ['year','lot']

x = USA_Cars[feature_cols]
y = USA_Cars.price


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



#data visualization with seaborn 
%matplotlib inline
sns.pairplot( USA_Cars, x_vars=['year','lot','mileage'],
                                 y_vars=['price'], size= 7, aspect = 0.7, kind='reg')



trainDataset = USA_Cars.sample(frac = 0,8, random_state =0)
testDataset = USA_Cars.dro(trainDataset.index)

sns.pairplot(trainDataset[['brand','state', 'country']],diag_kind = 'kde'


#matplotlib visualization
USA_Cars.model.hist(bins = 20)

USA_Cars.year.hist(bins = 20)




#tensorflow LR

nmh = np.random

mileage = USA_Cars['mileage'].values
price = USA_Cars ['price'].values


learning_rate = 0.01
training_epochs = 1000

display_step = 50

x_train = np.asarray(mileage)
y_train = np.asarray(price)


n_samples = x_train[0]

x = tf.plceholder('float')
y = tf.plceholder('float')

w = tf.Variable(rng.randn(), name = "weight")
b = tf.Variable(rng.randn(), name = "bias")

pred = tf.add(tf.multiply(x,w),b)
error = tf.reduce_sum(tf.pow(pred-y,2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

init = tf.global_variables_initializer()

x = np.linspace(1/100, 1, 100)
fig, ax = plt.subplots(1, figsize=(4.7, 3))
ax.plot(x, np.log(x), label='$\ln(x)$')
ax.legend()
plt.show()
