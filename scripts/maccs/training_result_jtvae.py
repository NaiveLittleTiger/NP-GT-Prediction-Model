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
X_train = np.load("../data/splits/jtvae/X_train.npy")
y_train = np.load("../data/splits/jtvae/y_train.npy")
y_train = y_train.ravel()
X_test= np.load("../data/splits/jtvae/X_test.npy")
y_test=np.load("../data/splits/jtvae/y_test.npy")
y_test = y_test.ravel()

# load external test set


# best parameters
GBDT_tuned_params = {"max_features":'auto',
                  "min_samples_leaf" : 44,
                  "max_depth" : 8,
                  'learning_rate': 0.018,
                  'n_estimators': 1786,
                  "subsample": 0.65
                  }

RF_tuned_params = {'max_depth': 37,
                   'max_features': 'auto',
                   'min_samples_leaf': 2,
                   'n_estimators': 1557,
                   'n_jobs': -1,
                   'random_state': 42
                  }
LR_tuned_params = {'C': 2288,
                   'penalty': 'l2',
                   'solver': 'lbfgs'
                  }
SVM_tuned_params = {'C': 121770,
                    'gamma': 0.51,
                    'kernel': 'rbf',
                     'probability': True
                    }
XGB_tuned_params={"booster":'gbtree',
                  "verbosity": 0,
                  "objective": "binary:logistic",
                  'reg_lambda': 5.3793849219104123e-08,
                  'reg_alpha': 6.950390994058419e-07,
                  'colsample_bytree': 0.4,
                  'subsample': 0.7,
                  "gamma":0,
                  'learning_rate': 0.016,
                  'n_estimators': 2100,
                  'max_depth': 15,
                  'random_state': 42,
                  'min_child_weight': 1}

# setting up models using tuned hyperparameters from Grid Search.
GBDT = GradientBoostingClassifier(random_state=42).set_params(**GBDT_tuned_params)
RF = RandomForestClassifier(random_state=42).set_params(**RF_tuned_params)
LR = LogisticRegression(random_state=42).set_params(**LR_tuned_params)
SVM = SVC(random_state=42).set_params(**SVM_tuned_params)
XGB = XGBClassifier(random_state=42).set_params(**XGB_tuned_params)