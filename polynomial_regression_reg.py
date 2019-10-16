#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100

x_data= x[0:N_TRAIN,:]

t_data = targets[0:N_TRAIN]
lambda_arr = [0,0.01, 0.1, 1, 10, 10**2, 10**3, 10**4]
k=10
train_error_dict={}
validation_error_dict={}
for lambda_value in lambda_arr:
	train_error_arr=[]
	valiation_error_arr=[]
	for i in range(k):
	    if i==0:
	        x_train=x_data[10:,:]
	        x_validation=x_data[:10,:]
	        t_train=t_data[10:]
	        t_validation=t_data[:10]
	    elif i==k-1:
	        x_train=x_data[:i*10,:]
	        x_validation=x_data[i*10:,:]
	        t_train=t_data[:i*10]
	        t_validation=t_data[i*10:]
	    else:
	        pos=10*i
	        x_train=np.concatenate((x_data[:pos,:],x_data[pos+10:,:]),axis=0)
	        x_validation=x_data[pos:pos+10,:]
	        t_train=np.concatenate((t_data[:pos],t_data[pos+10:]),axis=0)
	        t_validation=t_data[pos:pos+10]
	    (w,tr_err)=a1.linear_regression(x_train, t_train, basis='polynomial', reg_lambda=lambda_value, degree=2, mu=0, s=1,bias_term=True)

	    (test_preds,test_err)=a1.evaluate_regression(x_validation,t_validation,w,degree=2,bias_term=True,basis='polynomial')
	    train_error_arr.append(tr_err)
	    valiation_error_arr.append(test_err)
	train_error_dict[lambda_value]=np.mean(train_error_arr)
	validation_error_dict[lambda_value]=np.mean(valiation_error_arr)
print('\n\n',validation_error_dict)

plt.rcParams.update({'font.size': 15})
del(validation_error_dict[0])
del(train_error_dict[0])
plt.semilogx(list(validation_error_dict.keys()), list(validation_error_dict.values()))
plt.ylabel('validation set error')
plt.title('Average validation set error versus λ')
plt.xlabel('lambda')
plt.show()
