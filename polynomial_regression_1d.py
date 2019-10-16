#!/usr/bin/env python


import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
bias_arr=[True,False]
for bias in bias_arr:
	train_error=[]
	test_error=[]
	for i in range(7,15):
	    x_train=values[0:N_TRAIN,i]
	    x_test=values[N_TRAIN:,i]
	    
	    (w,tr_err)=a1.linear_regression(x_train, t_train, basis='polynomial', reg_lambda=0, degree=3, mu=0, s=1,bias_term=bias)
	    (test_preds,test_err)=a1.evaluate_regression(x_test,t_test,w,degree=3,bias_term=bias,basis='polynomial')
	    train_error.append(tr_err)
	    test_error.append(test_err)
	    print(i, tr_err,test_err)

	n_groups = 8

	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index,train_error, bar_width,
	alpha=opacity,
	color='b',
	label='train_error')

	rects2 = plt.bar(index + bar_width, test_error, bar_width,
	alpha=opacity,
	color='g',
	label='test_error')

	plt.xlabel('feature')
	plt.ylabel('RMSE')
	title ='RMSE by feature with bias = ' + str(bias)
	plt.title(title)
	plt.xticks(index + bar_width, [8,9,10,11,12,13,14,15])
	plt.legend()

	plt.tight_layout()
	plt.show()


for i in range(10,13):
	for bias in bias_arr:
		x_train=values[0:N_TRAIN,i]
		x_test=values[N_TRAIN:,i]

		t_train = targets[0:N_TRAIN]
		t_test = targets[N_TRAIN:]
		#bias=True
		
		(w,tr_err)=a1.linear_regression(x_train, t_train, basis='polynomial', reg_lambda=0, degree=3, mu=0, s=1,bias_term=bias)

		(test_preds,test_err)=a1.evaluate_regression(x_test,t_test,w,degree=3,bias_term=bias,basis='polynomial')
		# Use linspace to get a set of samples on which to evaluate
		x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
		x_ev=np.asmatrix(x_ev).T

		phi = a1.design_matrix(x_ev,3,basis='polynomial',bias_term=bias)
		train_preds=phi@w

		plt.plot(x_train,t_train ,'c*')
		plt.plot(x_ev,train_preds,'ro')
		plt.plot(x_test,t_test ,'b*')
		plt.xlabel('X')
		plt.ylabel('RMSE')
		plt.legend(['Train data','Polynomial','Test data'])
		title='Polynomial for feature '+ str(i) + ' with bias ='+str(bias)
		plt.title(title)
		plt.show()
