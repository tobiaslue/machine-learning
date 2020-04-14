import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from deep import Classifier
import torch

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


# data
train_features = pd.read_csv('train_features.csv')
train_labels= pd.read_csv('train_labels.csv', index_col='pid').sort_values(by='pid')
test_features = pd.read_csv('test_features.csv')

pids = test_features['pid'].drop_duplicates().sort_values().reset_index(drop=True)

# functions
def get_score(df_true, df_submission):
    """calculates the score according to the project guideline

    Parameters
    ----------
    df_true (Dataframe):
        A dataframe containing the reference labels
    df_submission (Dataframe): 
        A dataframe containing the calculated labels

    Returns
    -------
    score 
        The overall score
    """
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    print("Task1: ", task1, "\nTask2: ", task2, "\nTask3: ", task3)
    return score

def preprocess(X_train, X_test):
    """preprocesses the data

    Parameters
    ----------
    X_train (Dataframe): 
        A dataframe containing the train dataset to be preprocessed
    X_test (Dataframe):
        A dataframe containing the test dataset to be preprocessed

    Returns
    -------
    preprocessed (Dataframe):
        A dataframe containing preprocessed data (without NaNs)

    TODO implement data imputation and find a way to group data per pacient
    """
    X_train = (X_train.groupby('pid').mean()).fillna(X_train.median())
    X_train = X_train.drop('Time', axis=1).sort_values(by='pid')

    X_test = (X_test.groupby('pid').mean()).fillna(X_test.median())
    X_test = X_test.drop('Time', axis=1).sort_values(by='pid')

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train)
    return pd.DataFrame(scaled, index=X_train.index), pd.DataFrame(scaler.transform(X_test), index=X_test.index)

def subtask1(X_train, y_train, X_test):
    # y_pred = []

    # for test in TESTS:
    #     y = y_train[test]

    #     model = RandomForestClassifier(random_state=42, class_weight='balanced').fit(X_train, y)
    #     y_pred.append(model.predict_proba(X_test)[:,1])
    X = torch.Tensor(X_train.values)
    y = torch.Tensor(y_train[TESTS].values)
    X_pred = torch.Tensor(X_test.values)
    model = Classifier(35, 10)
    model.train(X, y)
    y_pred = model.predict(X_pred)
    return pd.DataFrame(y_pred.detach().numpy(), columns=TESTS, index=X_test.index)

def subtask2(X_train, y_train, X_test):
    X = torch.Tensor(X_train.values)
    y = torch.Tensor([y_train['LABEL_Sepsis'].values]).transpose(0, 1)
    X_pred = torch.Tensor(X_test.values)
    model = Classifier(35, 1)
    model.train(X, y)
    y_pred = model.predict(X_pred)
    return pd.DataFrame(y_pred.detach().numpy(), columns=['LABEL_Sepsis'], index=X_test.index)

def subtask3(X_train, y_train, X_test):
    y_pred = []

    for vital in VITALS:
        y = y_train[vital]

        model = LassoCV(random_state=42).fit(X_train, y)
        y_pred.append(model.predict(X_test))

    return pd.DataFrame(np.transpose(y_pred), columns=VITALS, index=X_test.index)

def evaluate_performance(X, y):
    """evaluates the performance dependant on the scoring

    Parameters
    ----------
    X (Dataframe):
        test dataset
    y (Dataframe):
        test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    labels_tests = subtask1(X_train, y_train, X_test)
    label_sepsis = subtask2(X_train, y_train, X_test)
    labels_vitals = subtask3(X_train, y_train, X_test)

    result = pd.concat([labels_tests, label_sepsis, labels_vitals], axis=1)

    score = get_score(y_test, result)
    print('Evaluated score: ', score)

def make_submission(X_train, y_train, X_test):
    labels_tests = subtask1(X_train, y_train, X_test)
    label_sepsis = subtask2(X_train, y_train, X_test)
    labels_vitals = subtask3(X_train, y_train, X_test)

    result = pd.concat([labels_tests, label_sepsis, labels_vitals], axis=1)

    result.to_csv('prediction.zip', float_format='%.3f', compression='zip')

   
X_train, X_test = preprocess(train_features, test_features)
y_train = train_labels

evaluate_performance(X_train, y_train)

#make_submission(X, y, X_submission)
