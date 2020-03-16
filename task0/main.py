import pandas as pd
from matplotlib import pyplot
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import numpy as np

trainset = pd.read_csv('./train.csv')
testset = pd.read_csv('./test.csv')

Y_train = trainset.pop('y').values
trainset.pop('Id')
X_train = trainset.values

Id_test = testset.pop('Id')
X_test = testset.values

Y = np.mean(X_test, axis=1)

models = []
models.append(linear_model.LinearRegression(fit_intercept=False))


for model in models:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    RMSE = mean_squared_error(Y, Y_pred)**0.5
    print(RMSE, model.coef_)

submission = pd.DataFrame(Y_pred)
submission = pd.merge(Id_test, submission, left_index=True, right_index=True)
submission.to_csv('./submission.csv', header=['Id', 'y'], index=False)