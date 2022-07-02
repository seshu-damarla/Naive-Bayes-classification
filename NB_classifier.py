# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:00:26 2022

@author: Seshu Kumar Damarla
"""

import numpy as np
import pandas as pd
import math
# data
data = pd.read_csv('data.csv',header=None)
xydata = np.array(data)
test_x=np.array([[6,130,8]])

xdata=xydata[:,1:]
N=xydata.shape[0]    # no. of examples

dd=xydata[:,0] == 1

C=np.unique(xydata[:,0])
C=C.astype(int)
C=np.array([C])

c1xdata=xydata[dd==True,1:]
c2xdata=xydata[dd==False, 1:]
n=c1xdata.shape[1]    # no. of. features

mean_features_c1=np.mean(c1xdata, axis=0, keepdims=True, dtype=np.float)
var_features_c1=(np.var(c1xdata, axis=0, keepdims=True, dtype=np.float)) *(c1xdata.shape[0]/(c1xdata.shape[0]-1))

mean_features_c2=np.mean(c2xdata, axis=0, keepdims=True, dtype=np.float)
var_features_c2=(np.var(c2xdata, axis=0, keepdims=True, dtype=np.float)) *(c2xdata.shape[0]/(c2xdata.shape[0]-1))

#mean_features = np.array([[mean_features_c1],[mean_features_c2]])
#std_features = np.array([[std_features_c1],[std_features_c2]])

prior=np.zeros([1,C.shape[1]])
prior[:,0] = c1xdata.shape[0] / N
prior[:,1] = c2xdata.shape[0] / N

likelihood = np.zeros([C.shape[1], n])

for j in range(n):
    #print(math.pi)
    a1 = np.sqrt(2*(var_features_c1[:,j]**2)*math.pi)
    a2 = (test_x[:,j]-mean_features_c1[:,j])**2
    a3 = 2*(var_features_c1[:,j]**2)
    a4 = np.exp(-a2/a3)
    likelihood[0,j] = (a4/a1)
    #print(likelihood[0,j])
    
for j in range(n):
    b1 = np.sqrt(2*(var_features_c2[:,j]**2)*math.pi)
    b2 = (test_x[:,j]-mean_features_c2[:,j])**2
    b3 = 2*(var_features_c2[:,j]**2)
    b4 = np.exp(-b2/b3)
    likelihood[1,j] = (b4/b1)
    print(likelihood[1,j])

def prod1(X,nn):
    ab=1
    X=np.array([X])   
    for i in range(nn):
        ab=ab*X[0,i]
    
    return ab
 
print(prod1(np.array([1,2,3]),3))
# p(x)    
evidence = prior[:,0] * prod1(likelihood[0,0:],n) + prior[:,1] * prod1(likelihood[1,0:],n)
#print(likelihood)
#print(prod1(likelihood[0,:],n))
#print(prod1(likelihood[1,:],n))

posterior_c1 = (prior[:,0] * prod1(likelihood[0,:],n)) / evidence
posterior_c2 = (prior[:,1] * prod1(likelihood[1,:],n)) / evidence

print(posterior_c1)
print(posterior_c2)

if posterior_c1>posterior_c2:
    class_label_testx = 1
else:
    class_label_testx = 0

print(class_label_testx)


    

    


