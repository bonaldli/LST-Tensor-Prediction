# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:02:44 2019

@author: zlibn
"""
# In[0]: import packages, define fuctions
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def MSE (A, B):
    from sklearn.metrics import mean_squared_error
    # matrix A and B are rquired to be same length
    mse = mse = mean_squared_error(A, B)
    return mse

# In[0.1] Input data
A = inflow_Tensor_J31_F28_M28[0:51] + 0.1 #Otherwise the original zero will be treated as missing
A = A.transpose((2,1,0)) # dim (90,247,51)
# In[1.0]: Method 1
##############################################################################
############### Method 1: Nuclear Norm Low-Rank Tensor Completion #############
###############################################################################
# Note that Method 1 ill-performs, so we didn't choose it eventually
    
# In[1.1]: Create the observed matrix A_Omega(observed still the 74th timestamp 
# of 50th day) and the to_be_predicted matrix A_test (from 75th timestamp at 50th day)

# A_Omega[:,74:247,50] are supposed to be 0
A_Omega = A.copy()
zero = np.zeros((90,173))
A_Omega[:,74:247,50] = zero
sub_d0 = 90*247*50+90*74
sub = np.zeros((90,247,51))
for i2 in range(50):
    for i1 in range(247):
        for i0 in range(90):
            sub[i0,i1,i2] = i0
        
#A_Test[:,74:247,50] are supposed to be non-zero
A_Test = np.zeros((90,247,51))
A_Test[:,74:247,50] = A[:,74:247,50].copy()

# save into matlab readable format
scipy.io.savemat('C:/Users/zlibn/Desktop/A.mat', mdict={'A': A})
scipy.io.savemat('C:/Users/zlibn/Desktop/A_Omega.mat', mdict={'A_Omega': A_Omega})
scipy.io.savemat('C:/Users/zlibn/Desktop/A_Test.mat', mdict={'A_Test': A_Test})

# In[1.2]: To get the sample indice, say (1,1,1) (since it's observed)
sub = np.zeros((sub_d0,3))
#fill 1st colum    
v0 = np.arange(1, 91, 1)
ind = np.arange(len(sub[:,0]))
np.put(sub[:,0], ind, v0)
#fill 2nd colum 
v1 = np.arange(1,248,1)

v1 = np.repeat(v1, 90)
ind = np.arange(len(sub[:,1]))
np.put(sub[:,1], ind, v1)
#fill 3rd colum 
v2 = np.arange(1,52,1)

v2 = np.repeat(v2, 22230)
ind = np.arange(len(sub[:,2]))
np.put(sub[:,2], ind, v2)
scipy.io.savemat('C:/Users/zlibn/Desktop/subs.mat', mdict={'subs': sub})

temp = sub[1111500:1118160,:]

# In[1.3]: To get the test indice, say (1,75,51) (since it's missing)
sub_Test = np.zeros((90*173,3))
#fill 1st colum    
v0 = np.arange(1, 91, 1)
ind = np.arange(len(sub_Test[:,0]))
np.put(sub_Test[:,0], ind, v0)
#fill 2nd colum 
v1 = np.arange(75,248,1)

v1 = np.repeat(v1, 90)
ind = np.arange(len(sub_Test[:,1]))
np.put(sub_Test[:,1], ind, v1)
#fill 3rd colum 
sub_Test[:,2] = sub_Test[:,2] + 51
scipy.io.savemat('C:/Users/zlibn/Desktop/subs_Test.mat', mdict={'subs_Test': sub_Test})
# In[1.4]: tensor completion in matlab

# In[1.5]: plot and evaluate the completion result from matlab
#just drag the .m file into "Variable Explorer"

#a = real_1day[0,:,0]
#b = pred_1day[0,:,0]
#e = pred_1day_mixed[0,:,0]
cpl = Full_X_d[0,:,50]

x = np.linspace(1, 247, 247)
fig = plt.figure(figsize=(6,4))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
ax.set_xticks([74], minor=False)
ax.plot(x, a, color = 'r', label= 'Real Data')
ax.plot(x, e, color = 'k', label= '30% New Data 70% Old Pred')
ax.plot(x, cpl, color = 'b', label= 'Tensor Completion')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel="Passenger InFlow at STN1 at Day51")
plt.legend(loc=2)
plt.draw()
# In[2.0]: Method 2
##############################################################################
################# Method 2: Bayesian Low-Rank Tensor Completion ##############
###############################################################################
# In[2.1]: to get the observation indictor matrix O for BFTC
O = np.zeros((90,247,51))+1 # 1 means "observed"
zero = np.zeros((90,173)) # 0 means "missing"
O[:,74:247,50] = zero
ratio = np.count_nonzero(O)/O.size #count the observation rate
# save into matlab readable format
scipy.io.savemat('C:/Users/zlibn/Desktop/O.mat', mdict={'O': O})

############################### Within cluster2 ###############################
# In[2.2] Since BTC within a cluster is tried first, so the cluster info is import
########  to get the observation indictor matrix O for BFTC within cluster 2
#pick the cluster2
C2 = list(x for x in range(71,85) if x != 83)
A2 = A[C2,:,:]
O2 = np.zeros((13,247,51))+1
zero = np.zeros((13,173))
O2[:,74:247,50] = zero
ratio = np.count_nonzero(O2)/O2.size
# save into matlab readable format
scipy.io.savemat('C:/Users/zlibn/Desktop/O2.mat', mdict={'O2': O2})
scipy.io.savemat('C:/Users/zlibn/Desktop/A2.mat', mdict={'A2': A2})
# In[2.3]: tensor completion in matlab

# In[2.4]: check the completion result 
stn = 0

plt.plot(X_hat[stn,:,50],'b')
plt.plot(A2[stn,:,50],'r')
MSE(A2[stn,:,50], X_hat[stn,:,50])

############################# in mixed station set #############################
# In[2.5]  Observation indicator matrix O_m, Complete Real Tensor A_m in mixed cluster
#import random
#C_m = random.sample(range(1, 90), 15)
C_m = [17,56,57,38,64,35,11,54,84,55,51,87,65,37,16]

#C_m = list(i for j in (range(5), range(71,85)) for i in j)
A_m = A[C_m,:,:]
O_m = np.zeros((15,247,51))+1
zero = np.zeros((15,173))
O_m[:,74:247,50] = zero
# if we wanna two days empty, as following
#O_m[:,74:247,49] = zero 
#O_m[:,:,50] = np.zeros((15,247))
print(np.count_nonzero(O_m))
print(O_m.size)
ratio = np.count_nonzero(O2)/O2.size
# save into matlab readable format
scipy.io.savemat('C:/Users/zlibn/Desktop/O_m.mat', mdict={'O_m': O_m})
scipy.io.savemat('C:/Users/zlibn/Desktop/A_m.mat', mdict={'A_m': A_m})
# In[2.6] BTC in matlab

# In[2.7]  check the completion result #############
stn = 2
plt.plot(A_m[stn,:,50],'r')
plt.plot(X_hat[stn,:,50],'b')

MSE(A_m[stn,:,50], X_hat[stn,:,50])
