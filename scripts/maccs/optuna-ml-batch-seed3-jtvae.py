import pandas as pd
import numpy as np
import random
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import optuna
import joblib
from optuna.samplers import TPESampler


#dedfine some variable
seed_list=[41,42,43];
dataDir='../../data/splits/jtvae/'
optuna_outDir='../../data/optuna_study/jtvae-batch/'
seed_model=42;

X_train = np.load(dataDir+"X_train.npy")
y_train = np.load(dataDir+"y_train.npy")
y_train = y_train.ravel()
X_test = np.load(dataDir+"X_test.npy")
y_test = np.load(dataDir+"y_test.npy")
y_test = y_test.ravel()

def objective_LR(trial):
    logreg_c=trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
    solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
    clf =  LogisticRegression(C=logreg_c,solver=solver,random_state = seed_model)
    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=5,scoring='roc_auc').mean()

def objective_RF (trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 2000)
    max_depth = trial.suggest_int('max_depth', 2, 128, log=True)
    min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 32, log=True)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_leaf= min_samples_leaf,random_state=seed_model)
    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=5,scoring='roc_auc').mean()

def objective_GBDT(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 2000)
    max_depth = trial.suggest_int('max_depth', 2, 128, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 64, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    subsample = trial.suggest_float("subsample", 0.4, 1.0, step=0.05)
    clf = GradientBoostingClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                     learning_rate= learning_rate,subsample=subsample,random_state=seed_model)
    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=5,scoring='roc_auc').mean()

def objective_SVM(trial):
    svc_c = trial.suggest_float("C", 1e-10, 1e10, log=True)
    svc_gamma = trial.suggest_float("gamma", 1e-15, 1e5, log=True)
    svc_kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf","sigmoid"])
    clf = SVC(C=svc_c, kernel=svc_kernel, gamma=svc_gamma,probability=True,random_state=seed_model)
    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=5,scoring='roc_auc').mean()

def objective_XGB(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {
        "verbosity": 0,
        # this parameter means using the GPU when training our model to speedup the training process
        "objective": "binary:logistic",
        "eval_metric": "auc",
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.45, 0.50,0.55, 0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': trial.suggest_int('n_estimators', 0, 5000,step=100),
        'max_depth': trial.suggest_categorical('max_depth', [1, 3, 5, 7, 9, 11, 13, 15, 17]),
        'random_state': trial.suggest_categorical('random_state',[42]),
        'min_child_weight': trial.suggest_int('min_child_weight',1,400),
        'gamma': trial.suggest_int('gamma', 0, 10)
    }
    #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    history = xgb.cv(param, dtrain, num_boost_round=500,early_stopping_rounds=50, nfold=5,seed=seed_model)
    mean_auc = history["test-auc-mean"].values[-1]
    return mean_auc

if __name__ == "__main__":
    for seed_sample in seed_list:

        print("searching parameters for LR and seed is {}".format(seed_sample))
        study_LR = optuna.create_study(direction= "maximize",sampler=TPESampler(seed=seed_sample))
        study_LR.optimize(objective_LR, n_trials=100)
        joblib.dump(study_LR, optuna_outDir+"study_LR_seed_%s.pkl" %seed_sample)

        #RF study save
        print("searching parameters for RF and seed is {}".format(seed_sample))
        study_RF = optuna.create_study(direction= "maximize",sampler=TPESampler(seed=seed_sample))
        study_RF.optimize(objective_RF, n_trials=100)
        joblib.dump(study_RF, optuna_outDir+"study_RF_seed_%s.pkl" %seed_sample)

        print("searching parameters for SVM and seed is {}".format(seed_sample))
        study_SVM = optuna.create_study(direction= "maximize",sampler=TPESampler(seed=seed_sample))
        study_SVM.optimize(objective_SVM, n_trials=100)
        joblib.dump(study_SVM, optuna_outDir+"study_SVM_seed_%s.pkl" %seed_sample)

        print("searching parameters for GBDT and seed is {}".format(seed_sample))
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study_GBDT = optuna.create_study(pruner=pruner,direction="maximize", sampler=TPESampler(seed=seed_sample))
        study_GBDT.optimize(objective_GBDT, n_trials=100)
        joblib.dump(study_GBDT, optuna_outDir+"study_GBDT_seed_%s.pkl" %seed_sample)

        print("searching parameters for XGBoost and seed is {}".format(seed_sample))
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study_XGB = optuna.create_study(pruner=pruner, direction="maximize", sampler=TPESampler(seed=seed_sample))
        study_XGB.optimize(objective_XGB, n_trials=100)
        joblib.dump(study_XGB, optuna_outDir+"study_XGBoost_seed_%s.pkl" %seed_sample)
    print("all seed have been finished")