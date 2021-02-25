
# coding: utf-8
###############################################################################
#Here
# In[146]:
# Define the decomposition function

import tensorly as tl 
from tensorly.decomposition import parafac, randomised_parafac
import numpy as np
from tensorly.decomposition import partial_tucker
import matplotlib.pyplot as plt


class parafacPCA:
    def __init__(self, dimension,nsample=1000):
        self.dimension = dimension
        self.factors = [None for i in range(3)]
        self.nsample = nsample
    def fit(self,data):
        factors = randomised_parafac(data,n_samples=self.nsample, rank=self.dimension,verbose=0)
        for k in range(0,3):
            self.factors[k] = factors[k].dot(np.diag(1/np.sqrt(np.sum(factors[k]**2,0))))

    def transform(self,data):
        ndim = data.shape[1]*data.shape[2]
        X = np.zeros((ndim,self.dimension))
        for k in range(self.dimension):
            X[:,k] = tl.kruskal_to_tensor([self.factors[1][:,[k]],self.factors[2][:,[k]]]).reshape(-1)
        beta = data.reshape(-1,ndim).dot(X).dot(np.linalg.inv(X.T.dot(X)))
        return beta
        
    def inverse_transform(self,beta):
        self.factors[0] = beta
        return tl.kruskal_to_tensor(self.factors)
                                              
class tuckerPCA:
    def __init__(self, dimension):
        self.dimension = dimension
        self.factors = None
    def fit(self,data):
        core, factors = partial_tucker(data,modes=[1,2],rank=[self.dimension,self.dimension])
        self.factors = factors
    def transform(self,data):
        return tl.tenalg.multi_mode_dot(data,[self.factors[0].T,self.factors[1].T],modes = [1,2])
    def inverse_transform(self,core):
        return tl.tenalg.multi_mode_dot(core,self.factors,modes = [1,2])

def superlong(flow_tensor, d_start, d_end, stn):
    temp = []
    for t in range(d_start-1, d_end):
        a = flow_tensor[t,:,stn]
        temp = np.hstack((temp, a))
    return temp

def real_vs_pred(xtest, xprediction, stn, dayto):
    for i in range(dayto):
        a = xtest[stn,:,i]
        b = xprediction[stn,:,i]
        x = np.linspace(1, 247, 247)
        fig = plt.figure(figsize=(6,4))
        fig.show()
        ax = fig.add_subplot(111)
        ax.plot(x, a, color = 'r', label= 'Real Data')
        ax.plot(x, b, color = 'b', label= 'Fitted Data')
        
        ax.set_facecolor('w')
        ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)
    
        ax.set(xlabel="Time", ylabel=f'Passenger InFlow at STN {stn} at Day{i+51}')
        plt.legend(loc=2)
        plt.draw()

def real_vs_recon(xtest, train_recon, stn, day):
    a = xtest[stn,:,day]
    b = train_recon[stn,:,day]
    x = np.linspace(1, 247, 247)
    fig = plt.figure(figsize=(6,4))
    fig.show()
    ax = fig.add_subplot(111)
    ax.plot(x, a, color = 'r', label= 'Real Data')
    ax.plot(x, b, color = 'b', label= 'Recon Data')
        
    ax.set_facecolor('w')
    ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)
    
    ax.set(xlabel="Time", ylabel=f'Passenger InFlow at STN {stn} at Day{day}')
    plt.legend(loc=2)
    plt.draw()

def MSE (A, B):
    from sklearn.metrics import mean_squared_error
    # matrix A and B are rquired to be same length
    mse = mse = mean_squared_error(A, B)
    return mse

# In[1]: The Tensor to be decomposed

xtest = inflow_Tensor_J31_F28_M28[54:60]
xtrain = inflow_Tensor_J31_F28_M28[4:60]
# In[1]: The Tensor to be decomposed

xtest = inflow_Tensor_J31_F28_M28[54:60].transpose((2,1,0))
xtrain = inflow_Tensor_J31_F28_M28[4:60].transpose((2,1,0))

# In[1]:
#xtest = inflow_Tensor[50:56].transpose((2,1,0))
xtrain = inflow_Tensor_W[0:8]

# In[]:
pred_1day = xprediction[:,:,0:1]
xtrain = pred_1day
# In[2.1]: CP Decomposition: Decomposition on training test [90,247,56]

#cppca = parafacPCA(90,nsample=90)
cppca = parafacPCA(50,nsample=90)
cppca.fit(xtrain)
train_code = cppca.transform(xtrain)
train_recon = cppca.inverse_transform(cppca.transform(xtrain))
train_recon_score = np.sum(np.sum((xtrain-train_recon)**2,2),1)
train_score = np.sum(train_code**2,1)

a = xtrain[10,:,50]
b = train_recon[10,:,50]

plt.plot(a,'r')
plt.plot(b,'b')

MSE(a,b)
# In[2.2] CP Decomposition: the 3 modes factors

cpfactor0 = cppca.factors[0]
cpfactor1 = cppca.factors[1]
cpfactor2 = cppca.factors[2]

plt.plot(cppca.factors[1])
plt.figure()
plt.plot(cppca.factors[2])

# In[2.3] CP Decomposition: take a glance how the factors look like
f = plt.figure()
for i in range(0,3):
    plt.plot(cpfactor2[:,i])

f = plt.figure()
for i in range(3,6):
    plt.plot(cpfactor2[:,i])

f = plt.figure()
for i in range(0,3):
    plt.plot(cpfactor1[:,i])

f = plt.figure()
for i in range(3,6):
    plt.plot(cpfactor1[:,i])
    
# In[2.4] how does the flatten T_mode factor look like

a = cpfactor2[:,0:30].transpose().flatten()

x = np.linspace(1, len(a), len(a))
fig = plt.figure(figsize=(20,5))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
ax.plot(x, a, color = 'b', label= 'Inflow')
ax.set(xlabel="Rank", ylabel="Day Matrix Column of Each Day")
plt.legend(loc=2)
plt.draw()


# In[]
#a = superlong(inflow_Tensor, 1,56,0)
#x = np.linspace(1, 13832, 13832)
#fig = plt.figure(figsize=(25,5))
#fig.show()
#ax = fig.add_subplot(111)
##ax.set_facecolor('w')
#ax.plot(x, a, color = 'r', label= 'Inflow')
#ax.set_facecolor('w')
#ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)
#
#ax.set(xlabel="Time", ylabel="Inflow in Station1 from 1-Jan to 25-Feb")
#plt.legend(loc=2)
#plt.draw()
#
## In[1] to check the period of the tensor data
#a = superlong(inflow_Tensor_Jan, 1,31,0)
#b = superlong(inflow_Tensor_Feb1_25, 1,25,0)
#superlong_STN0 = np.concatenate((a,b))
#
#x = np.linspace(1, 13832, 13832)
#fig = plt.figure(figsize=(30,5))
#fig.show()
#ax = fig.add_subplot(111)
##ax.set_facecolor('w')
#ax.plot(x, superlong_STN0, color = 'r', label= 'Inflow')
#
#ax.set_facecolor('w')
#ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)
#
#ax.set(xlabel="Time", ylabel="Inflow in Station1 from 1-Jan to 25-Feb")
#plt.legend(loc=2)
#plt.draw()
#
## In[]
#a = train_recon.transpose((2,1,0))
#x = np.linspace(1, 2964, 2964)
#fig = plt.figure(figsize=(20,5))
#fig.show()
#ax = fig.add_subplot(111)
##ax.set_facecolor('w')
#ax.plot(x, a, color = 'b', label= 'Inflow')
#ax.set_facecolor('w')
#ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)
#
#ax.set(xlabel="Rank", ylabel="recon of 01 Feb to 12 Feb at STN1")
#plt.legend(loc=2)
#plt.draw()

# In[3.1] Prediction based on CP Decomposiiton

k = 6 #we are doing k-step prediction
#prediction = prediction.reshape(1,-1) #for one-step prediction, converting row into column
#prediction = prediction_for_T_mode_6_days_pred_50_rank #load the prediction result
pred_factors = [np.zeros((90, 50)),np.zeros((247, 50)),np.zeros((k, 50))]
pred_factors[0] = cpfactor0
pred_factors[1] = cpfactor1
pred_factors[2] = pred #or "= prediction"
xprediction = tl.kruskal_to_tensor(pred_factors)
#xtest = inflow_Tensor[24:25].transpose((2,1,0))

# In[3.2]: how does the CP prediction perform
real_vs_pred(xtest, xprediction, stn=0, dayto=1)

# In[4.1]: Tucker Decomposition

tucker_pca = tuckerPCA(50)
tucker_pca.fit(xtrain)
train_code = tucker_pca.transform(xtrain)
train_score = np.sum(np.sum(train_code**2,2),1)
train_recon = tucker_pca.inverse_transform(tucker_pca.transform(xtrain))
train_recon_score = np.sum(np.sum((xtrain-train_recon)**2,2),1)

# In[4.2] Tucker Decomposition: how accurate the reconstruction 
a = xtrain[38,:,50]
b = train_recon[38,:,50]

plt.plot(a,'r')
plt.plot(b,'b')

MSE(a,b)

# In[4.3]: Collect all the factors
tfactor0 = tucker_pca.factors[0]
tfactor1 = tucker_pca.factors[1]
core = train_code

plt.plot(tfactor0[:,0])
plt.figure()
plt.plot(tfactor1[:,0]) #T_mode

# In[5.1] Tucker Decomposition Prediction
k = 6 #we are doing k-step prediction
#prediction = prediction.reshape(1,-1) #for one-step prediction, converting row into column
#prediction = prediction_for_T_mode_6_days_pred_50_rank #load the prediction result
pred_tfactors = [np.zeros((247, 50)),np.zeros((k, 50))]
pred_tfactors[0] = tfactor0
pred_tfactors[1] = tprediction_fit3_complete_data #tprediction
xprediction_t =  tl.tenalg.multi_mode_dot(core,pred_tfactors,modes = [1,2])

# In[3.2]: how does the prediction perform
real_vs_pred(xtest, xprediction_t, stn=38, dayto=1)
day = 0
stn = 38
a = xtest[stn,:,day]
b = xprediction_t[stn,:,day]
MSE(a,b)
