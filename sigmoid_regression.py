#!/usr/bin/env python


import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
feature=10
x_train=values[0:N_TRAIN,feature]
x_test=values[N_TRAIN:,feature]
bias=True
mu=[100,10000]

s=2000
(w,tr_err)=a1.linear_regression(x_train, t_train, basis='sigmoid', reg_lambda=0, degree=3, mu=mu, s=s,bias_term=bias)
(test_preds,test_err)=a1.evaluate_regression(x_test,t_test,w,degree=3,bias_term=bias,basis='sigmoid',mu=mu,s=s)

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev=np.asmatrix(x_ev).T
phi = a1.design_matrix(x_ev,3,basis='sigmoid',bias_term=bias,mu=mu,s=s)
train_preds=phi@w
plt.plot(x_train,t_train ,'c*')
plt.plot(x_ev,train_preds,'ro')
plt.plot(x_test,t_test ,'b*')
plt.xlabel('X')
plt.ylabel('RMSE')
plt.legend(['Train data','Polynomial','Test data'])
title='Polynomial for feature 11 using sigmoid basis function'
plt.title(title)
plt.show()
labels=['train_error','test_error']
errors=[tr_err,test_err]
print(errors)
n_groups = 2
fig, ax = plt.subplots()
index = np.arange(n_groups)
plt.bar(index, errors)
plt.xlabel('train/test')
plt.ylabel('RMSE')
plt.xticks(index, labels)
plt.title('Sigmoid basis function train/test error')
plt.show()
plt.tight_layout()
plt.legend()
