"""
Replication of Francis Galton study on parent-child heights.

@author: arnulf
"""
#%% MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

#%% DATA IMPORT
fpath = r'C:\Users\jquintero\Downloads'
fname = r'\galton_data.csv'
data = pd.read_csv(fpath+fname)

#%% DATA MGMT

# Repsonse and regressor separation
y = data['Height']
X = data['Father']
var_y = y.mean()
var_x = X.mean()

# OLS Estimators by construction
Sxy = np.sum((X-var_x)*(y-var_y))
Sxx = np.sum((X-var_x)**2)

beta1 = Sxy/Sxx
beta0 = var_y-var_x*beta1

# OLS by model
lr = LR()
lr.fit(X.to_numpy().reshape(-1,1),y.to_numpy().reshape(-1,1))
lr.coef_
lr.intercept_

# Plot optimal equation
hat_y = beta0 + beta1*X
plt.figure(figsize=(10,7))
ax = plt.axes()
ax.scatter(X, y, c=X, alpha=0.5)
ax.plot(X,hat_y,color='red')
ax.set_xlabel('Altura ancestro')
ax.set_ylabel('Altura descendiente')
plt.tight_layout; plt.show()

#%% DATA VIZ

# Scatter plot y vs X
plt.figure(figsize=(10,7))
ax = plt.axes()
ax.scatter(X, y, c=X, alpha=0.5)

ax.axvline(x=63.8, c='purple', alpha = 0.4)
ax.axvline(x=64.2, c='purple', alpha = 0.4)

ax.axvline(x=74.8, c='green', alpha = 0.4)
ax.axvline(x=75.2, c='green', alpha = 0.4)

ax.axhline(y=y[X.apply(lambda x: x>63.8 and x<64.2)].mean(), c='purple', 
           linestyle='--', alpha=0.6)
ax.axhline(y=y[X.apply(lambda x: x>74.8 and x<75.2)].mean(), c='green', 
           linestyle='--', alpha=0.6)

ax.set_xlabel('Altura ancestro')
ax.set_ylabel('Altura descendiente')
plt.tight_layout; plt.show()

#%% LIN REG

# Figuring out linear equation
b0,b1 = y.mean()-y.std()/X.std()*np.corrcoef(X,y)[1,0]*X.mean(), y.std()/X.std()*np.corrcoef(X,y)[1,0]
lin1 = b0+b1*X
lin2 = y.mean()-1.6*y.std()/X.std()*np.corrcoef(X,y)[1,0]*X.mean()+1.6*b1*X
plt.figure(figsize=(10,7))
ax = plt.axes()
ax.scatter(X, y, c='gray', alpha=0.5)
ax.plot(X,lin1)
ax.plot(X,lin2)
ax.set_xlabel('Altura ancestro')
ax.set_ylabel('Altura descendiente')
plt.tight_layout; plt.show()

# Optimal parameters by OLS

