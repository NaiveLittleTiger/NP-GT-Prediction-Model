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

# load external test set



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