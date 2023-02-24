# this script is used  for generating ecfp4 kfold results and external set
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
import joblib
import os
plt.style.use("seaborn-deep")

# load data
X_train = np.load("../data/splits/ecfp/X_train.npy")
y_train = np.load("../data/splits/ecfp/y_train.npy")
y_train = y_train.ravel()
X_test= np.load("../data/splits/ecfp/X_test.npy")
y_test=np.load("../data/splits/ecfp/y_test.npy")
y_test = y_test.ravel()
# load external test set
X_Berry = np.load("../data/splits/ecfp/X_Berry.npy")
y_Berry = np.load("../data/splits/ecfp/y_Berry.npy")
X_Oat = np.load("../data/splits/ecfp/X_Oat.npy")
y_Oat = np.load("../data/splits/ecfp/y_Oat.npy")

# read study from different model
def get_best_LR_model(study_dir):
    study = joblib.load(study_dir)
    best_param=study.best_trial.params
    best_param['penalty']='l2'
    best_param['C']=best_param.pop('logreg_c')
    model=LogisticRegression(random_state=42).set_params(**best_param)
    return model

def get_best_RF_model(study_dir):
    study = joblib.load(study_dir)
    best_param=study.best_trial.params
    best_param['max_features'],best_param['n_jobs']='auto',-1;
    model=RandomForestClassifier(random_state=42).set_params(**best_param)
    return model

def get_best_SVM_model(study_dir):
    study = joblib.load(study_dir)
    best_param=study.best_trial.params
    best_param['probability']= True
    model=SVC(random_state=42).set_params(**best_param)
    return model

def get_best_GBDT_model(study_dir):
    study = joblib.load(study_dir)
    best_param=study.best_trial.params
    best_param['max_features'] = 'auto';
    model=GradientBoostingClassifier(random_state=42).set_params(**best_param)
    return model

def get_best_XGB_model(study_dir):
    # XGB 寻优使用的方法是xgb.DMatrix 方法和直接使用的xgboost方法不一样,需要对lambda 以及alpha进行改名
    study = joblib.load(study_dir)
    best_param=study.best_trial.params
    best_param["reg_lambda"]=best_param.pop("lambda")
    best_param["reg_alpha"] = best_param.pop("alpha")
    best_param["booster"],best_param["objective"],best_param["verbosity"],best_param["n_jobs"]= 'gbtree',"binary:logistic",0,-1

    model=XGBClassifier(random_state=42).set_params(**best_param)
    return model
# path is where to save best study

path="../data/optuna_study/ecfp/"
fileList = os.listdir(path)
for file in fileList:
    fileName,extension=os.path.splitext(file)
    if extension == '.pkl':
        seed = fileName.split('_')[-1]
        if 'LR' in fileName:
            LR=get_best_LR_model(path+file)
        if 'RF' in fileName:
            RF=get_best_RF_model(path+file)
        if 'SVM' in fileName:
            SVM=get_best_SVM_model(path+file)
        if 'GBDT' in fileName:
            GBDT=get_best_GBDT_model(path+file)
        if 'XGB' in fileName:
            XGB=get_best_XGB_model(path+file)
print("all models have been test")

# K-fold validation function will take six arguments:
# 1. model, 2. train set, 3.test set, 4.# of fold, 5.shuffle boolean, 6.random_state.
def kf_cv(clf, X, y, folds=10, shuf=True, random_state=42):
    cv = KFold(n_splits=folds, random_state=42, shuffle=shuf)
    f1, prec, recall, acc, ROC_AUC, MCC, conf = ([], [], [], [], [], [], [])
    metric_cols = [ 'Accuracy', 'Precision', 'Recall', 'F1','ROC_AUC', 'MCC','Confusion_matrix']
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
    return pd.DataFrame(zip(acc, prec, recall, f1, ROC_AUC, MCC,conf), columns=metric_cols)

def train_kfold_csv():
    # create dictionary to store k-fold CV results
    cv_score_dict = {}
    # create a dataframe to store the mean and standard error of the mean (SEM) from k-fold CV
    cv_score_mean = pd.DataFrame()
    cv_score_sem = pd.DataFrame()

    # run k-fold CV
    for model_name in ['LR','SVM','RF', 'GBDT', 'XGB']:
        clf = eval(model_name)
        cv_score_dict[model_name] = kf_cv(clf, X_train, y_train, 10, True)
        cv_score_mean[model_name] = cv_score_dict[model_name].mean()
        cv_score_sem[model_name] = cv_score_dict[model_name].sem()

    cv_score_mean = cv_score_mean.transpose().astype(float) # round(3)
    cv_score_sem = cv_score_sem.transpose()

    print('\n\ndetail results in dictionary "cv_score_dict" using model_name as key')
    print('Performace summary in df "cv_score_mean" and "cv_score_sem"')

    # check mean performance of each model.
    cv_score_mean.to_csv("../results/kfold/ecfp/cv_score_mean.csv")
    cv_score_sem.to_csv("../results/kfold/ecfp/cv_score_sem.csv")
    # plot k-fold CV by methods
    cv_score_mean.plot(ylim=(0, 1), kind='bar',
                                         rot=0, stacked=False,
                                         yerr=cv_score_sem, capsize=4,
                                         figsize=(12, 12), fontsize=24)
    plt.legend(loc=3, prop={'size': 18})
    plt.title('CV_plot_methods', fontsize=20)
    plt.savefig("../results/kfold/ecfp/CV_plot_methods.png")

    # plot k-fold CV by metrics
    cv_score_mean.transpose().plot(kind='bar', ylim=(0, 1),
                                                     rot=0, stacked=False,
                                                     yerr=cv_score_sem.transpose(), capsize=4,
                                                     figsize=(12, 12), fontsize=22)
    plt.legend(loc=3, prop={'size': 18})
    plt.title('CV_plot_metrics', fontsize=20)
    plt.savefig("../results/kfold/ecfp/CV_plot_metrics.png")

# run k-fold CV on test set
# create a dataframe to store test set performance
def test_kfold_csv():
    test_set_result = pd.DataFrame()
    acc, prec, recall, f1,ROC_AUC, MCC, conf = ([], [], [], [], [], [],[])

    # evaluate the performance of the different classifiers
    for model_name in ['LR','SVM','RF', 'GBDT', 'XGB']:
        clf = eval(model_name)
        clf.fit(X_train, y_train.ravel())
        pred_test = clf.predict(X_test)
        pred_test_probs = clf.predict_proba(X_test)
        f1.append(f1_score(y_test, pred_test))
        prec.append(precision_score(y_test, pred_test))
        recall.append(recall_score(y_test, pred_test))
        acc.append(accuracy_score(y_test, pred_test))
        ROC_AUC.append(roc_auc_score(y_test, pred_test_probs[:, 1]))
        MCC.append(matthews_corrcoef(y_test, pred_test))
        conf.append(confusion_matrix(y_test, pred_test))

    test_set_scores = zip(acc, prec, recall, f1, ROC_AUC, MCC, conf)
    test_set_result = pd.DataFrame(test_set_scores,
                                   columns=['Accuracy','Precision', 'Recall','F1',
                                             'ROC_AUC', 'MCC','Confusion_matrix'],
                                   index=['LR','SVM','RF', 'GBDT', 'XGB'])
    test_set_result.iloc[:, :-1].transpose().plot(kind='bar', ylim=(0, 1),
                                                                          rot=0, stacked=False, legend=False,
                                                                          figsize=(12, 12), fontsize=22)
    test_set_result.to_csv("../results/kfold/ecfp/innner_test_set_plot_metrics.csv")
    plt.legend(loc=3, prop={'size': 18})
    plt.title('test_set_plot_metrics', fontsize=20)
    plt.savefig("../results/kfold/ecfp/test_set_plot_metrics.png")

def test_auc():
    color_dict={'LR':'deepskyblue','SVM':'red','RF':'darkorange','GBDT':'green','XGB':'purple'}
    plt.figure()
    plt.figure(figsize=(12, 12))
    for model_name in ['LR','SVM','RF', 'GBDT', 'XGB']:
        clf = eval(model_name)

        clf.fit(X_train, y_train.ravel())

        #pred_test = clf.predict(X_test)
        pred_test_probs = clf.predict_proba(X_test)
        pred_test = pred_test_probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred_test)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color_dict[model_name], label= str(model_name) +" " + 'ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--',linewidth=2.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC curves for test set',fontsize=22)
    plt.legend(loc="lower right",prop={'size': 18})
    plt.savefig("../results/kfold/ecfp/test_set_plot_auc_roc.png")

# test model on external test set berry and oat
def external_test_set(X_external,y_external,name):
    test_set_result = pd.DataFrame()
    acc, prec, recall, f1, ROC_AUC, MCC, conf = ([], [], [], [], [], [], [])

    # evaluate the performance of the different classifiers
    for model_name in ['LR','SVM','RF', 'GBDT', 'XGB']:
        clf = eval(model_name)

        clf.fit(X_train, y_train.ravel())

        pred_test = clf.predict(X_external)
        pred_test_probs = clf.predict_proba(X_external)
        fpr, tpr, thresholds = roc_curve(y_external, pred_test)

        f1.append(f1_score(y_external, pred_test))
        prec.append(precision_score(y_external, pred_test))
        recall.append(recall_score(y_external, pred_test))
        acc.append(accuracy_score(y_external, pred_test))
        ROC_AUC.append(roc_auc_score(y_external, pred_test_probs[:, 1]))
        MCC.append(matthews_corrcoef(y_external, pred_test))
        conf.append(confusion_matrix(y_external, pred_test))

    test_set_scores = zip(acc, prec, recall, f1, ROC_AUC, MCC, conf)

    test_set_result = pd.DataFrame(test_set_scores,
                                   columns=['Accuracy', 'Precision', 'Recall', 'F1',
                                            'ROC_AUC', 'MCC', 'Confusion_matrix'],
                                   index=['LR','SVM','RF', 'GBDT',  'XGB'])
    test_set_result.iloc[:, :-1].transpose().plot(kind='bar', ylim=(0, 1),
                                                  rot=0, stacked=False, legend=False,
                                                  figsize=(12, 12), fontsize=22)
    test_set_result.to_csv("../results/kfold/ecfp/%s_external_test_set_plot_metrics.csv" % (name))
    plt.legend(loc=3, prop={'size': 18})
    plt.title(' %s external test set results ' % (name), fontsize=20)
    plt.savefig("../results/kfold/ecfp/%s_external_test_set_plot_metrics.png" % (name))

if __name__ == "__main__":
    # make picture
    test_auc()
    train_kfold_csv()
    test_kfold_csv()
    external_test_set(X_external=X_Berry, y_external=y_Berry, name="Berry")
    external_test_set(X_external=X_Oat, y_external=y_Oat, name="Oat")

