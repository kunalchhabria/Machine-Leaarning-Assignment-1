#!/usr/bin/env python


"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import copy
from scipy import nanmean
def load_unicef_data():
 
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    #print(inds[1][:5])
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(data):

    x=copy.deepcopy(data)
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1,bias_term=None):


  phi = design_matrix(x,degree,basis=basis,bias_term=bias_term,mu=mu,s=s)
  phi_transpose=phi.T 
  if reg_lambda >0:
    XTX=phi_transpose@phi
    phi_inverse=np.linalg.inv(XTX + np.eye(len(XTX))*reg_lambda) 
    w=(phi_inverse@phi_transpose)@t
    
  else:
    w=np.linalg.pinv(phi)@t

  preds= phi@w
  train_err=np.sqrt((np.power((preds-t),2)).mean())

  return (w, train_err)



def design_matrix(data,degree,basis="polynomial",bias_term=None,mu=None,s=None):


  if basis == 'polynomial':
    phi=copy.deepcopy(data)
    if (degree>1):
      for i in range(2,degree+1):
        
        new_values=np.power(data,i)
        phi=np.concatenate((phi,new_values),1)

    if bias_term:
      phi=np.concatenate((np.ones((len(phi),1)),phi),1)
   
  elif basis == 'sigmoid':

    for i in mu:

      a= (data-i)/s
      new_values=1/(1+np.exp(-a))
      #phi=new_values
      if i==mu[0]:
        phi=new_values
      else:
        phi=np.concatenate((phi,new_values),1)
    if bias_term:
      phi=np.concatenate((np.ones((len(phi),1)),phi),1)

  else: 
      assert(False), 'Unknown basis %s' % basis

  return phi


def evaluate_regression(x_test,t_test,w,degree,bias_term=None,basis=None,mu=None,s=None):

  data=copy.deepcopy(x_test)
  t=t_test
  phi = design_matrix(data,degree,basis=basis,bias_term=bias_term,mu=mu,s=s)
  preds= phi@w
  err=np.sqrt((np.power((preds-t),2)).mean())
  t_est = preds
  return (t_est, err)
