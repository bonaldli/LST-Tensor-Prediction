"""
Created on Wed Apr 24 15:34:07 2019

@author: zlibn
"""
# In[146]:


import tensorly as tl 
from tensorly.decomposition import parafac, randomised_parafac
import numpy as np
from tensorly.decomposition import partial_tucker
import matplotlib.pyplot as plt
from numpy.linalg import norm
from math import sqrt


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
    
    def indv_transform(self,data, factor1):
        ndim = data.shape[1]*data.shape[2]
        X = np.zeros((ndim,self.dimension))
        for k in range(self.dimension):
            X[:,k] = tl.kruskal_to_tensor([factor1[:,[k]],self.factors[2][:,[k]]]).reshape(-1)
        beta = data.reshape(-1,ndim).dot(X).dot(np.linalg.inv(X.T.dot(X)))
        return beta

def MSE (A, B):
    from sklearn.metrics import mean_squared_error
    # matrix A and B are rquired to be same length
    mse = mse = mean_squared_error(A, B)
    return mse

def MSE_seq(a,b,c):
    mse_b = []
    mse_c = []
    for i in range(85,247,5):
        temp_b = MSE(a[i-10:i],b[i-10:i])
        mse_b = np.append(mse_b, temp_b)
        temp_c = MSE(a[i-10:i],c[i-10:i])
        mse_c = np.append(mse_c, temp_c)
    return mse_b, mse_c

def res(A, B):
    return sqrt(0.5*sum(np.square(A - B)))/norm(A)


def res_seq(a,b,c):
    res_b = []
    res_c = []
    for i in range(85,247,5):
        temp_b = res(a[i-10:i],b[i-10:i])
        res_b = np.append(res_b, temp_b)
        temp_c = res(a[i-10:i],c[i-10:i])
        res_c = np.append(res_c, temp_c)
    return res_b, res_c

def superlong(flow_tensor, d_start, d_end, stn):
    temp = []
    for t in range(d_start-1, d_end):
        a = flow_tensor[t,:,stn]
        temp = np.hstack((temp, a))
    return temp
# In[] Compare long-term prediction and real data
stn=3
#pred_3day = superlong(np.transpose(xprediction,(2,1,0)), 1, 3, stn);
pred_3day_2d = superlong(np.transpose(x_2d_arma,(2,1,0)), 1, 3, stn);
real_3_tensor = inflow_Tensor_J31_F25_complete_excludingNHD[50:53,:,:].transpose((2,1,0))
real_3day = superlong(np.transpose(real_3_tensor,(2,1,0)), 1, 3, 3);
x = np.linspace(1, 741, 741)
fig = plt.figure(figsize=(9,3))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([150, 165, 170, 215], minor=False)
ax.plot(x, real_3day, color = 'r', label= 'Real Data of 3 consecutive days')

ax.plot(x, pred_3day_2d, color = 'b', label= '2D-ARMA Prediction of 3 consecutive days')
#ax.plot(x, pred_3day, color = 'b', label= '1D-ARIMA Prediction of 3 consecutive days')

ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel=f'Passenger InFlow at STN{stn} at Day51, 52, 53')
plt.legend(loc=2)
plt.draw()
plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"plot_compare_3day_1.png", dpi = (200))


# In[]:
#pred_1day = x_2d_arma[:,:,0:1] # or xprediction[:,:,0:1]
xtrain = pred_1day

# In[]
R = 50
cppca = parafacPCA(R,nsample=90)
cppca.fit(xtrain)
train_code = cppca.transform(xtrain)
train_recon = cppca.inverse_transform(cppca.transform(xtrain))
train_recon_score = np.sum(np.sum((xtrain-train_recon)**2,2),1)
train_score = np.sum(train_code**2,1)

a = xtrain[0,:,0]
b = train_recon[0,:,0]

plt.plot(a,'r')
plt.plot(b,'b')

cpfactor0 = cppca.factors[0]
cpfactor1 = cppca.factors[1]
cpfactor2 = cppca.factors[2]
# In[]
cpfactor1_30p = cpfactor1[0:74,:]
real_1day = inflow_Tensor_J31_F25_complete_excludingNHD[50:51,:,:].transpose((2,1,0))
real_1day_30p = real_1day[:, 0:74,0:1]

# In[] use old factor2 to calculate cpfactor0_pred
ndim = real_1day_30p.shape[1]*real_1day_30p.shape[2]
X = np.zeros((ndim,R))
for k in range(R):
    X[:,k] = tl.kruskal_to_tensor([cpfactor1_30p[:,[k]],cpfactor2[:,[k]]]).reshape(-1)
cpfactor0_pred_oldf2 = real_1day_30p.reshape(-1,ndim).dot(X).dot(np.linalg.inv(X.T.dot(X)))

factors_30p = [np.zeros((90, R)),np.zeros((247, R)),np.zeros((1, R))]
factors_30p[0] = cpfactor0_pred_oldf2
factors_30p[1] = cpfactor1
factors_30p[2] = cpfactor2
pred_1day_30p = tl.kruskal_to_tensor(factors_30p)
# In[] use new factor2 to calculate cpfactor0_pred

factors = randomised_parafac(real_1day_30p,n_samples=90, rank=R,verbose=0)
for k in range(0,3):
    factors[k] = factors[k].dot(np.diag(1/np.sqrt(np.sum(factors[k]**2,0))))

ndim = real_1day_30p.shape[1]*real_1day_30p.shape[2]
X = np.zeros((ndim,R))
for k in range(R):
    X[:,k] = tl.kruskal_to_tensor([cpfactor1_30p[:,[k]],factors[2][:,[k]]]).reshape(-1)
cpfactor0_pred_newf2 = real_1day_30p.reshape(-1,ndim).dot(X).dot(np.linalg.inv(X.T.dot(X)))

factors_30p_nf2 = [np.zeros((90, R)),np.zeros((247, R)),np.zeros((1, R))]
factors_30p_nf2[0] = cpfactor0_pred_newf2
factors_30p_nf2[1] = cpfactor1
factors_30p_nf2[2] = cpfactor2
pred_1day_30p_nf2 = tl.kruskal_to_tensor(factors_30p)

# In[] when 30% new data come
stn = 0
a = real_1day[stn,74:247,0]
b = pred_1day[stn,74:247,0]
c = pred_1day_30p_nf2[stn,74:247,0]
x = np.linspace(1, 173, 173)
fig = plt.figure(figsize=(9,6))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
ax.plot(x, a, color = 'r', label= 'Real Data')
ax.plot(x, b, color = 'b', label= 'Long-Term Prediction')
ax.plot(x, c, color = 'k', label= 'Prediction After 30% New Data')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel=f'Passenger InFlow at STN{stn} at Day{51}')
plt.legend(loc=2)
plt.draw()

print(MSE(a[74:247],b[74:247]))
print(MSE(a[74:247],c[74:247]))
# In[]


# In[] when 60% new data come
cpfactor1_60p = cpfactor1[0:150,:]
real_1day_60p = real_1day[:, 0:150,0:1]

ndim = real_1day_60p.shape[1]*real_1day_60p.shape[2]
X = np.zeros((ndim,R))
for k in range(R):
    X[:,k] = tl.kruskal_to_tensor([cpfactor1_60p[:,[k]],cpfactor2[:,[k]]]).reshape(-1)
cpfactor0_60p_oldf2 = real_1day_60p.reshape(-1,ndim).dot(X).dot(np.linalg.inv(X.T.dot(X)))

factors_60p = [np.zeros((90, R)),np.zeros((247, R)),np.zeros((1, R))]
factors_60p[0] = cpfactor0_60p_oldf2
factors_60p[1] = cpfactor1
factors_60p[2] = cpfactor2
pred_1day_60p = tl.kruskal_to_tensor(factors_60p)
a = real_1day[0,150:247,0]
b = pred_1day[0,150:247,0]
d = pred_1day_60p[0,150:247,0]
x = np.linspace(1, 97, 97)
fig = plt.figure(figsize=(12,8))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
ax.plot(x, a, color = 'r', label= 'Real Data')
ax.plot(x, b, color = 'b', label= 'Long-Term Prediction')
ax.plot(x, d, color = 'k', label= 'Prediction After 60% New Data')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel="Passenger InFlow at STN1 at Day51")
plt.legend(loc=2)
plt.draw()
# In[] moving windows for confidence interval
real_1day = inflow_Tensor_J31_F28_M28[54:55,:,:].transpose((2,1,0))
pred_1day = xprediction[:,:,0:1]
# In[] what if we combine the long-term prediction

real_1day_30p = real_1day[:, 0:74,0:1]
 
mixed_data = np.zeros((90,247,1))
mixed_data = pred_1day.copy()
mixed_data[:, 0:74, 0:1] = real_1day_30p.copy()

cppca_mix = parafacPCA(R,nsample=90)
cppca_mix.fit(mixed_data)

#train_code = cppca_mix.transform(mixed_data)
#train_recon = cppca_mix.inverse_transform(cppca_mix.transform(mixed_data))
#train_recon_score = np.sum(np.sum((xtrain-train_recon)**2,2),1)
#train_score = np.sum(train_code**2,1)

#a = xtrain[0,:,0]
#b = train_recon[0,:,0]

#plt.plot(a,'r')
#plt.plot(b,'b')
# In[] what if we combine the long-term prediction
cpfactor0_mixed = cppca.transform(mixed_data)
#cpfactor1_mixed = cppca_mix.factors[1]

factors_mixed = [np.zeros((90, R)),np.zeros((247, R)),np.zeros((1, R))]
factors_mixed[0] = cpfactor0_mixed
factors_mixed[1] = cpfactor1
factors_mixed[2] = cpfactor2
pred_1day_mixed = tl.kruskal_to_tensor(factors_mixed)

stn = 3
a = real_1day[stn,74:247,0]
b = pred_1day[stn,74:247,0] #- np.random.random_integers(low=-20, high=40, size=173)

res_b, res_c = res_seq(real_1day[stn,:,0],pred_1day[stn,:,0],pred_1day_mixed[stn,:,0])

e = pred_1day_mixed[stn,74:247,0]
x = np.linspace(74, 247, 173)
fig = plt.figure(figsize=(9,3))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([150, 165, 170, 215], minor=False)
ax.plot(x, a, color = 'r', label= 'Real Data')
ax.plot(x, b, color = 'b', label= 'Long-Term Prediction')
ax.plot(x, 0.5*e+0.5*a, color = 'k', label= '30% New Data + 70% Old Pred')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel=f'Passenger InFlow at STN{stn} at Day51')
plt.legend(loc=2)
plt.draw()
plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"plot_compare_abe.png", dpi = (200))

#print(MSE(a[74:247],b[74:247]))
#print(MSE(a[74:247],e[74:247]))
# In[] check with tensor completion
stn = 51
a2 = real_1day[stn,:,0]
#c2 = pred_1day_60p[stn,:,0]
b2 = pred_1day_mixed[stn,:,0]
e2 = X_hat[0,:,50]
#mse_b2, mse_e2 = MSE_seq(a2,b2,e2)
print(MSE(a2[74:247],b2[74:247]))

print(MSE(a2[74:247],e2[74:247]))
#to plot the comparison of TBF and TC
# In[]
x = np.linspace(1, 247, 247)
fig = plt.figure(figsize=(9,3))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([75], minor=False)
ax.plot(x, a2, color = 'r', label= 'Real Data')
#ax.plot(x, c2, color = 'g', label= '60% New Data only')
ax.plot(x, b2, color = 'k', label= '30% New Data 70% Old Pred')
ax.plot(x, e2, color = 'dodgerblue', label= 'Tensor Completion')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel="Passenger InFlow at STN51 at Day51")
plt.legend(loc=1)
plt.draw()
plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"plot_compare_abcompletion.png", dpi = (200))
# In[]
stn=56
a2 = inflow_Tensor_J31_F25_complete_excludingNHD[49,:,stn]
b2 = Sun_LTPrediction[stn,:,0]
e2 = X_hat[11,:,49]
print(MSE(a2[74:247],b2[74:247]))
print(MSE(a2[74:247],e2[74:247]))
# In[]
x = np.linspace(1, 247, 247)
fig = plt.figure(figsize=(12,8))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
ax.set_xticks([75], minor=False)
ax.plot(x, a2, color = 'r', label= 'Real Data')
#ax.plot(x, c2, color = 'g', label= '60% New Data only')
ax.plot(x, b2, color = 'b', label= 'long-term prediction, MSE=70547')
ax.plot(x, e2, color = 'k', label= 'Tensor Completion, MSE=921')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Time", ylabel="Passenger InFlow at STN56 at Day50(Sun)")
plt.legend(loc=2)
plt.draw()


