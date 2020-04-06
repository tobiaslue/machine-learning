import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

trainset = pd.read_csv('train.csv')
trainset.pop('Id')
y = trainset.pop('y').values
X = trainset.values

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

kf = KFold(10)

res = []
for alpha in alphas:
    RMSEs = []
    RMSE = 0
    model = Ridge(alpha=alpha, fit_intercept=False)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RMSEs.append(mean_squared_error(y_test, y_pred)**0.5)

    RMSE = np.mean(RMSEs)
    print(RMSE)
    res = np.append(res, RMSE)
    

submission = pd.DataFrame(res)
submission.to_csv('submission.csv', header=False, index=False)
