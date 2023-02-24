from sklearn.model_selection import KFold
from sklearn.metrics import (recall_score, accuracy_score, confusion_matrix, f1_score, precision_score,
                             auc, roc_auc_score, roc_curve, precision_recall_curve, matthews_corrcoef)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
X_train = np.load("../data/splits/X_train.npy")
y_train = np.load("../data/splits/y_train.npy")
y_train = y_train.ravel()
X_test= np.load("../data/splits/X_test.npy")
y_test=np.load("../data/splits/y_test.npy")
y_test = y_test.ravel()

# best parameters
GBDT_tuned_params = {"max_features":'auto',
                  "min_samples_leaf" : 6,
                  "max_depth" : 39,
                  'learning_rate': 0.02,
                  'n_estimators': 1203,
                  "subsample": 0.60
                  }

RF_tuned_params = {'max_depth': 72,
                   'max_features': 'auto',
                   'min_samples_leaf': 2,
                   'n_estimators': 785,
                   'n_jobs': -1,
                   'random_state': 42
                  }
LR_tuned_params = {'C': 158,
                   'penalty': 'l2',
                   'solver': 'saga'
                  }
SVM_tuned_params = {'C': 2689,
                    'gamma': 0.08,
                    'kernel': 'rbf',
                     'probability': True
                    }
XGB_tuned_params={"booster":'gbtree',
                  "verbosity": 0,
                  "objective": "binary:logistic",
                  'reg_lambda': 0.00015428425554129491,
                  'reg_alpha': 0.002879984126740577,
                  'colsample_bytree': 0.6,
                  'subsample': 0.65,
                  "gamma":0,
                  'learning_rate': 0.02,
                  'n_estimators': 3100,
                  'max_depth': 11,
                  'random_state': 42,
                  'min_child_weight': 1}

# setting up models using tuned hyperparameters from Grid Search.
GBDT = GradientBoostingClassifier(random_state=42).set_params(**GBDT_tuned_params)
RF = RandomForestClassifier(random_state=42).set_params(**RF_tuned_params)
LR = LogisticRegression(random_state=42).set_params(**LR_tuned_params)
SVM = SVC(random_state=42).set_params(**SVM_tuned_params)
XGB = XGBClassifier(random_state=42).set_params(**XGB_tuned_params)

# K-fold validation function will take six arguments:
# 1. model, 2. train set, 3.test set, 4.# of fold, 5.shuffle boolean, 6.random_state.
def kf_cv(clf, X, y, folds=10, shuf=True, random_state=42):
    cv = KFold(n_splits=folds, random_state=42, shuffle=shuf)
    f1, prec, recall, acc, ROC_AUC, MCC, conf = ([], [], [], [], [], [], [])
    metric_cols = ['F1', 'precision', 'recall', 'accuracy', 'ROC_AUC', 'MCC','confusion_matrix']
    print('Classifier ##{0}## will be used with {1} folds, and shuffle mode is {2}'.format(clf.__class__.__name__,
                                                                                           folds,
                                                                                           'on' if shuf else 'off')
          )

    for train_index, test_index in cv.split(X):
        # turn on the below code will print train_index/test_index and may help understand the details and debug.
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # if sampling method is used, use this to avoid warnings: clf.fit(X_train.values, y_train.values.ravel())
        clf.fit(X_train, y_train.ravel())

        pred_test = clf.predict(X_test)
        pred_test_probs = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pred_test)

        f1.append(f1_score(y_test, pred_test))
        prec.append(precision_score(y_test, pred_test))
        recall.append(recall_score(y_test, pred_test))
        acc.append(accuracy_score(y_test, pred_test))
        ROC_AUC.append(roc_auc_score(y_test, pred_test_probs[:, 1]))
        MCC.append(matthews_corrcoef(y_test, pred_test))
        conf.append(confusion_matrix(y_test, pred_test))

    # return a dataframe consisting five metrics and confusion matrix.
    return pd.DataFrame(zip(f1, prec, recall, acc, ROC_AUC, MCC,conf), columns=metric_cols)

def train_kfold_csv():
    # create dictionary to store k-fold CV results
    cv_score_dict = {}
    # create a dataframe to store the mean and standard error of the mean (SEM) from k-fold CV
    cv_score_mean = pd.DataFrame()
    cv_score_sem = pd.DataFrame()

    # run k-fold CV
    for model_name in ['RF', 'GBDT', 'LR', 'SVM', 'XGB']:
        clf = eval(model_name)
        cv_score_dict[model_name] = kf_cv(clf, X_train, y_train, 10, True)
        cv_score_mean[model_name] = cv_score_dict[model_name].mean()[:-1]
        cv_score_sem[model_name] = cv_score_dict[model_name].sem()

    cv_score_mean = cv_score_mean.transpose().astype(float).round(2)
    cv_score_sem = cv_score_sem.transpose().round(2)

    print('\n\ndetail results in dictionary "cv_score_dict" using model_name as key')
    print('Performace summary in df "cv_score_mean" and "cv_score_sem"')

    # check mean performance of each model.
    cv_score_mean.to_csv("../results/kfold/cv_score_mean.csv",index=False)
    cv_score_sem.to_csv("../results/kfold/cv_score_sem.csv",index=False)
    # plot k-fold CV by methods
    cv_score_mean.plot(ylim=(0, 1), kind='bar',
                                         rot=0, stacked=False,
                                         yerr=cv_score_sem, capsize=4,
                                         figsize=(10, 10), fontsize=24)
    plt.legend(loc=3, prop={'size': 15})
    plt.title('CV_plot_methods', fontsize=20)
    plt.savefig("../results/kfold/CV_plot_methods.png")

    # plot k-fold CV by metrics
    cv_score_mean.transpose().plot(kind='bar', ylim=(0, 1),
                                                     rot=0, stacked=False,
                                                     yerr=cv_score_sem.transpose(), capsize=4,
                                                     figsize=(10, 10), fontsize=22)
    plt.legend(loc=2, prop={'size': 20})
    plt.title('CV_plot_metrics', fontsize=20)
    plt.savefig("../results/kfold/CV_plot_metrics.png")

# run k-fold CV on test set
# create a dataframe to store test set performance
def test_kfold_csv():
    test_set_result = pd.DataFrame()
    f1, prec, recall, acc, ROC_AUC, MCC, conf = ([], [], [], [], [], [],[])

    # evaluate the performance of the different classifiers
    for model_name in ['RF', 'GBDT', 'LR', 'SVM','XGB']:
        clf = eval(model_name)

        clf.fit(X_train, y_train.ravel())

        pred_test = clf.predict(X_test)
        pred_test_probs = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pred_test)

        f1.append(f1_score(y_test, pred_test))
        prec.append(precision_score(y_test, pred_test))
        recall.append(recall_score(y_test, pred_test))
        acc.append(accuracy_score(y_test, pred_test))
        ROC_AUC.append(roc_auc_score(y_test, pred_test_probs[:, 1]))
        MCC.append(matthews_corrcoef(y_test, pred_test))
        conf.append(confusion_matrix(y_test, pred_test))

    test_set_scores = zip(f1, prec, recall, acc, ROC_AUC, MCC, conf)

    test_set_result = pd.DataFrame(test_set_scores,
                                   columns=['F1', 'precision', 'recall',
                                            'accuracy', 'ROC_AUC', 'MCC','confusion_matrix'],
                                   index=['RF', 'GBDT', 'LR', 'SVM','XGB'])
    test_set_result.iloc[:, :-1].transpose().plot(kind='bar', ylim=(0, 1),
                                                                          rot=0, stacked=False, legend=False,
                                                                          figsize=(10, 10), fontsize=22)
    plt.legend(loc=3, prop={'size': 15})
    plt.title('test_set_plot_metrics', fontsize=20)
    plt.savefig("../results/kfold/test_set_plot_metrics.png")

if __name__ == "__main__":
    # make picture
    train_kfold_csv()
    test_kfold_csv()
