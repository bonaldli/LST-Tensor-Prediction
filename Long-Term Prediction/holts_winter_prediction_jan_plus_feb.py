# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:48:42 2019

@author: zlibn
"""
# packages to be installed
# the reference link: https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
# the reference link: https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd #data manipulation and analysis in numerical tables and time series
import numpy as np
import matplotlib.pyplot as plt

def plot(df, r):
    x = np.linspace(1, len(df), len(df))
    fig = plt.figure(figsize=(6,5))
    fig.show()
    ax = fig.add_subplot(111)
    #ax.set_facecolor('w')
    ax.plot(x, df, color = 'r', label= f'R{r+1}')
    ax.set_facecolor('w')
    ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

    ax.set(xlabel="T-Mode")
    plt.legend(loc=2)
    plt.draw()

# In[] Import the T_mode data
r=1
data = pd.DataFrame(cpfactor2.transpose()) #tfactor1
data.columns = ["day{0}".format(i) for i in range(1, 57)]
real = data.iloc[r]
df = data.iloc[r][0:50]
plot(df,r)
# In[] First Try Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(df).fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.forecast(6).rename(r'$\alpha=0.2$')

fit2 = SimpleExpSmoothing(df).fit(smoothing_level=0.6,optimized=False)
fcast2 = fit2.forecast(6).rename(r'$\alpha=0.6$')

fit3 = SimpleExpSmoothing(df).fit()
fcast3 = fit3.forecast(6).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

# plot
ax = real.plot(marker='o', color='black', figsize=(12,8))
fcast1.plot(marker='o', color='blue', legend=True)
fit1.fittedvalues.plot(marker='o',  color='blue')

fcast2.plot(marker='o', color='red', legend=True)
fit2.fittedvalues.plot(marker='o', color='red')

fcast3.plot(marker='o', color='green', legend=True)
fit3.fittedvalues.plot(marker='o', color='green')
plt.show()
# The prediction result is terrible, so abandon it
# In[] Second Try Simple Holt Linear 
fit1 = Holt(df).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(6).rename("Holt's linear trend")

#fit2 = Holt(df, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
#fcast2 = fit2.forecast(6).rename("Exponential trend")

fit3 = Holt(df, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(6).rename("Additive damped trend")

ax = real.plot(color="black", marker="o", figsize=(12,8))
fit1.fittedvalues.plot(marker="o", color='blue')
fcast1.plot(color='blue', marker="o", legend=True)
#fit2.fittedvalues.plot(marker="o", color='red')
#fcast2.plot(color='red', marker="o", legend=True)
fit3.fittedvalues.plot(marker="o", color='green')
fcast3.plot(color='green', marker="o", legend=True)

plt.show()
# The prediction result is terrible, so abandon it
# In[] Finally try Holt Winters Method with trend and seasonal
a = -real.min() + 1e-010
df_temp = 10*(df + a)
real_temp = 10*(real + a)
fit1 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='add').fit(use_boxcox=True)
fit2 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='mul').fit(use_boxcox=True)
fit3 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
fit4 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']

results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
results["Additive Dam"]   = [fit3.params[p] for p in params] + [fit3.sse]
results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]


ax = real_temp.plot(color="black", marker="o", figsize=(12,8), title=f'Forecasts for Rank{r+1} from Holt-Winters method' )
ax.set_ylabel("Factors Value")
ax.set_xlabel("Day")

fit1.fittedvalues.plot(ax=ax, style='--', color='red')
fit2.fittedvalues.plot(ax=ax, style='--', color='green')
fit3.fittedvalues.plot(ax=ax, style='--', color='blue')
fit4.fittedvalues.plot(ax=ax, style='--', color='purple')

fit1.forecast(6).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)
fit2.forecast(6).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)
fit3.forecast(6).rename('Holt-Winters (add-add-seasonal-damped)').plot(ax=ax, style='--', marker='o', color='blue', legend=True)
fit4.forecast(6).rename('Holt-Winters (add-mul-seasonal-damped)').plot(ax=ax, style='--', marker='o', color='purple', legend=True)
plt.show()
# The result so far is not bad, so the following T-mode prediction is based on this method
# And so far it is observed that add-mul-seasonal-damped offers the most amount of accurate prediction
# In[] loop to solve all the 50 models since we have 50 rank
##### Input: T_mode Factor Matrix (say, cpfactor2)

pred = []
R = data.shape[0]
mse = np.zeros(R)
k = 6 # k-step prediction
#path_img = "C:\\Users\\zlibn\\Desktop\\T_mode_Prediction_cpfactor\\"
R_names = ["R{0}.png".format(i) for i in range(0, R)]

for i in range(R):
    df = data.iloc[i][0:50]# the training set is the previous 50 data
    real = data.iloc[i]
    mini = real.min()
    a = - mini + 1e-010 # the fitting method requires strict positive
    df_temp = 10*(df + a)
    real_temp = 10*(real + a)

    fit1 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='add').fit(use_boxcox=True) # nan is not gonna happen by fit1
    #fit3 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
    #fit4 = ExponentialSmoothing(df_temp, seasonal_periods=7, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)

    #print(f'success to predict Rank-{i}')
    #plt.figure()
    #ax = real_temp.plot(color="black", marker="o", figsize=(12,8), title=f'Forecasts for Rank{i+1} from Holt-Winters method' )
    #ax.set_ylabel("Factors Value")
    #ax.set_xlabel("Day")
    #fit3.fittedvalues.plot(style='--', color='purple')
    #fit3.forecast(k).rename('Holt-Winters (add-mul-seasonal-damped)').plot(style='--', marker='o', color='purple', legend=True)
#    plt.figure()
#    fit3.fittedvalues.plot(style='--', color='blue', label=f'Fitted Value of Rank{i+1}')
#    k_step_pred = fit3.forecast(k)
#    k_step_pred.plot(style='--', marker='o', color='blue', legend=True, label=f'Prediction') #deduct the added value
#    plt.plot(full_temp, '--k')
    #plt.savefig(path_img + R_names[i], dpi = (200))
    #print(f'success to save the plot of Rank-{i}')
    #mse[i] = sum((fit4.forecast(k)-real_temp[50:56])**2)/k
    #print(f'success to evaluate Rank-{i} prediction')
    temp = 0.1*fit1.forecast(k).values - a #convert series into array, and convert the prediction back to original value
    pred = np.hstack((pred, temp))
    #print(f'success to stack Rank-{i}')

pred = pred.reshape((R, k))
pred = pred.transpose()

# In[]
# now the result 'pred' can be substituted back to 'tensor_decomposition_pca_Jan_plus_Feb'

    
