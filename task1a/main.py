import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

trainset = pd.read_csv('train.csv')
trainset.pop('Id')
y = trainset.pop('y').values
X = trainset.values

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

alphas = [0.01, 0.1, 1, 10, 100]

res = []
for alpha in alphas:
    RMSE = 0
    
    model = Ridge(alpha=alpha)
    MSEs = cross_val_score(model, X_scaled, y, scoring='neg_root_mean_squared_error', cv=10)
    RMSE = -np.mean(MSEs)
    print(RMSE)
    res = np.append(res, RMSE)
    

submission = pd.DataFrame(res)
submission.to_csv('submission.csv', header=False, index=False)
