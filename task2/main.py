import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

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

def feature_engineer(data):
    """preprocesses the data

    Parameters
    ----------
    data (Dataframe): 
        A dataframe containing the dataset to be preprocessed

    Returns
    -------
    preprocessed (Dataframe):
        A dataframe containing preprocessed data (without NaNs)

    TODO implement data imputation and find a way to group data per pacient
    """
    data = (data.groupby('pid').mean()).fillna(data.median())
    data = data.drop('Time', axis=1).sort_values(by='pid')

    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), index=data.index)

def subtask1(X_train, y_train, X_test):
    y_pred = []

    for test in TESTS:
        y = y_train[test]

        model = LogisticRegression(random_state=42).fit(X_train, y)
        y_pred.append(model.predict_proba(X_test)[:,1])

    return pd.DataFrame(np.transpose(y_pred), columns=TESTS, index=X_test.index)

def subtask2(X_train, y_train, X_test):

    y = y_train[('LABEL_Sepsis')]

    model = LogisticRegression(random_state=42).fit(X_train, y)
    y_pred = model.predict_proba(X_test)[:,1]

    return pd.DataFrame(np.transpose(y_pred), columns=['LABEL_Sepsis'], index=X_test.index)

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

   
X = feature_engineer(train_features)
y = train_labels

evaluate_performance(X, y)

X_submission = feature_engineer(test_features)
#make_submission(X, y, X_submission)
