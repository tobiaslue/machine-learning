import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

trainset = pd.read_csv('train.csv')
y = trainset['y'].values
X = trainset[['x1', 'x2', 'x3', 'x4', 'x5']].values

def fun(x):
    xx = np.append(x, [x**2, np.exp(x), np.cos(x)])
    return np.append(xx, 1)

def preprocess(X):
    return np.apply_along_axis(fun, 1, X)

X_preprocessed = preprocess(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=1)

model = linear_model.LassoCV(fit_intercept=False, max_iter=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print((mean_squared_error(y_test, y_pred))**0.5)
print(model.coef_)

# df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
# df.plot()
# plt.show()

submission = pd.DataFrame(model.coef_)
submission.to_csv('submission.csv', header=False, index=False)
