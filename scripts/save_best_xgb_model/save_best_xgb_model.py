# 本脚本在于保存最好的XGBoost模型和GBDT模型，并计划部署上线
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
import pickle

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

path="../../data/optuna_study/ecfp/"
fileList = os.listdir(path)
for file in fileList:
    fileName,extension=os.path.splitext(file)
    if extension == '.pkl':
        seed = fileName.split('_')[-1]
        if 'GBDT' in fileName:
            GBDT=get_best_GBDT_model(path+file)
        if 'XGB' in fileName:
            XGB=get_best_XGB_model(path+file)
pickle.dump(XGB, open("./XGBoost.pkl", "wb"))
pickle.dump(GBDT, open("./GBDT.pkl", "wb"))

print("best model has been saved")