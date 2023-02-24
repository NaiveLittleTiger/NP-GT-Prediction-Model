'''This script needs to  finish
1、get optuna study from three different seed
2、concat these three results into one dataframe
3、draw picture to show this result
'''
import optuna
import joblib
import os
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
plt.style.use("seaborn-deep")

# load data using  maccs as small molecular representation
dataDir="../../data/splits/maccs/"

X_train = np.load(dataDir+"X_train.npy")
y_train = np.load(dataDir+"y_train.npy")
y_train = y_train.ravel()
X_test= np.load(dataDir+"X_test.npy")
y_test=np.load(dataDir+"y_test.npy")
y_test = y_test.ravel()

# load external test set
X_Berry = np.load(dataDir+"X_Berry.npy")
y_Berry = np.load(dataDir+"y_Berry.npy")
X_Oat = np.load(dataDir+"X_Oat.npy")
y_Oat = np.load(dataDir+"y_Oat.npy")

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

def model_evaluation(clf,model_name,seed,X_test,y_test):
    test_set_result = pd.DataFrame()
    acc, prec, recall, f1, ROC_AUC, MCC, conf,Seed = ([], [], [], [], [], [], [], [])
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
    Seed.append(seed)
    test_set_scores = zip(acc, prec, recall, f1, ROC_AUC, MCC, conf,Seed)
    test_set_result = pd.DataFrame(test_set_scores,
                                   columns=['Accuracy','Precision', 'Recall','F1',
                                             'ROC_AUC', 'MCC','Confusion_matrix','Seed'],
                                   index=[model_name])
    return test_set_result

def test_concat_seed_csv(path,X_test,y_test):
    # get all study
    df_empty= pd.DataFrame(columns=['Accuracy','Precision', 'Recall','F1','ROC_AUC', 'MCC','Confusion_matrix','Seed'])
    fileList = os.listdir(path)
    for file in fileList:
        fileName,extension=os.path.splitext(file)
        if extension == '.pkl':
            seed = fileName.split('_')[-1]
            if 'LR' in fileName:
                model=get_best_LR_model(path+file)
                df = model_evaluation(model,model_name='LR',seed=seed, X_test=X_test,y_test=y_test)
                df_empty=df_empty.append(df)
            if 'RF' in fileName:
                model=get_best_RF_model(path+file)
                df = model_evaluation(model,model_name='RF',seed=seed,X_test=X_test,y_test=y_test)
                df_empty=df_empty.append(df)
            if 'SVM' in fileName:
                model=get_best_SVM_model(path+file)
                df = model_evaluation(model,model_name='SVM',seed=seed,X_test=X_test,y_test=y_test)
                df_empty=df_empty.append(df)
            if 'GBDT' in fileName:
                model=get_best_GBDT_model(path+file)
                df = model_evaluation(model,model_name='GBDT',seed=seed,X_test=X_test,y_test=y_test)
                df_empty=df_empty.append(df)
            if 'XGB' in fileName:
                model=get_best_XGB_model(path+file)
                df = model_evaluation(model,model_name='XGB',seed=seed,X_test=X_test,y_test=y_test)
                df_empty=df_empty.append(df)
    print("all models have been test")
    return df_empty

def draw_picture(df,name=""):
    df=df.drop(columns='Seed')
    df=df.reset_index();
    df=df.rename(columns={'index': 'model'});
    df_mean = df.groupby('model').mean()
    df_sem = df.groupby('model').sem()
    df_mean = df_mean.transpose()
    df_sem =  df_sem.transpose()
    #save
    df_mean.to_csv("../../results/test_set_seed/maccs/df_mean.csv")
    df_sem.to_csv("../../results/test_set_seed/maccs/df_sem.csv")
    # plot three seed  CV by methods
    df_mean.plot(ylim=(0, 1), kind='bar',
                       rot=0, stacked=False,
                       yerr=df_sem, capsize=4,
                       figsize=(12, 12), fontsize=24)
    plt.legend(loc=3, prop={'size': 18})
    plt.title('%s_test_set_methods' %name, fontsize=20)
    plt.savefig("../../results/test_set_seed/maccs/%s_test_seed_plot_methods.png" %name)

    # plot three seed CV by metrics
    df_mean.transpose().plot(kind='bar', ylim=(0, 1),
                                   rot=0, stacked=False,
                                   yerr=df_sem.transpose(), capsize=4,
                                   figsize=(12, 12), fontsize=22)
    plt.legend(loc=3, prop={'size': 18})
    plt.title('%s_test_set_metrics' %name, fontsize=20)
    plt.savefig("../../results/test_set_seed/maccs/%s_test_seed_plot_metrics.png" %name)

if __name__ == "__main__":
    # make picture maccs
    study_path="../../data/optuna_study/maccs-batch/"
    output_data_path = "../../results/test_set_seed/maccs/"

    df = test_concat_seed_csv(study_path,X_test=X_test,y_test=y_test)
    df.to_csv(output_data_path + "Inner_test.csv")
    draw_picture(df=df,name="Inner")
    #Berry
    df = test_concat_seed_csv(study_path, X_test=X_Berry, y_test=y_Berry)
    df.to_csv(output_data_path + "Berry_test.csv")
    draw_picture(df=df,name="Berry")
    #Oat
    df = test_concat_seed_csv(study_path, X_test=X_Oat, y_test=y_Oat)
    df.to_csv(output_data_path + "Oat_test.csv")
    draw_picture(df=df,name="Oat")

