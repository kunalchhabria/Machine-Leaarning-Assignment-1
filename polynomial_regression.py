#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
normalize_arr=[False,True]

for normalize in normalize_arr:
	if normalize:
		x = a1.normalize_data(x)
	N_TRAIN = 100
	x_train = x[0:N_TRAIN,:]
	x_test = x[N_TRAIN:,:]
	t_train = targets[0:N_TRAIN]
	t_test = targets[N_TRAIN:]
	train_error={}
	test_error={}
	for i in range(1,6+1):
		#if i==3:continue
		bias=True
		(w,tr_err)=a1.linear_regression(x_train, t_train, basis='polynomial', reg_lambda=0, degree=i, mu=0, s=1,bias_term=bias)
		(test_preds,test_err)=a1.evaluate_regression(x_test,t_test,w,degree=i,bias_term=bias,basis='polynomial')
		train_error[i]=tr_err
		test_error[i]=test_err
	print(train_error,test_error)

		# Produce a plot of results.
	plt.rcParams.update({'font.size': 15})
	plt.plot(list(train_error.keys()), list(train_error.values()))
	plt.plot(list(test_error.keys()), list(test_error.values()))
	plt.ylabel('RMSE')
	plt.legend(['Training error','Testing error'])
	title='Fit with polynomials of degree 1 to 6 and regularization= ' + str(normalize)
	plt.title(title)
	plt.xlabel('Polynomial degree')
	plt.show()
