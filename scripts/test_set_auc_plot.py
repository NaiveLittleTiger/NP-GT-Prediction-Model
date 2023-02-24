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
plt.style.use("seaborn-deep")

# load data
X_train = np.load("../../data/splits/X_train.npy")
y_train = np.load("../../data/splits/y_train.npy")
y_train = y_train.ravel()
X_test= np.load("../../data/splits/X_test.npy")
y_test=np.load("../../data/splits/y_test.npy")
y_test = y_test.ravel()
# load external test set
X_Berry = np.load("../../data/splits/X_Berry.npy")
y_Berry = np.load("../../data/splits/y_Berry.npy")
X_Oat = np.load("../../data/splits/X_Oat.npy")
y_Oat = np.load("../../data/splits/y_Oat.npy")

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
                  'n_jobs':-1,
                  'max_depth': 11,
                  'random_state': 42,
                  'min_child_weight': 1}

# setting up models using tuned hyperparameters from Grid Search.
GBDT = GradientBoostingClassifier(random_state=42).set_params(**GBDT_tuned_params)
RF = RandomForestClassifier(random_state=42).set_params(**RF_tuned_params)
LR = LogisticRegression(random_state=42).set_params(**LR_tuned_params)
SVM = SVC(random_state=42).set_params(**SVM_tuned_params)
XGB = XGBClassifier(random_state=42).set_params(**XGB_tuned_params)

def test_auc():
    color_dict={'LR':'deepskyblue','RF':'darkorange','GBDT':'green','SVM':'red','XGB':'purple'}
    plt.figure()
    plt.figure(figsize=(12, 12))
    for model_name in ['LR','RF', 'GBDT', 'SVM','XGB']:
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
    plt.savefig("../results/test/test_set_plot_auc_roc.png")
    plt.show()

def external_test_set(X_external,y_external,name):
    test_set_result = pd.DataFrame()
    acc, prec, recall, f1, ROC_AUC, MCC, conf = ([], [], [], [], [], [], [])

    # evaluate the performance of the different classifiers
    for model_name in ['LR', 'RF', 'GBDT', 'SVM', 'XGB']:
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
                                   index=['LR', 'RF', 'GBDT', 'SVM', 'XGB'])
    test_set_result.iloc[:, :-1].transpose().plot(kind='bar', ylim=(0, 1),
                                                  rot=0, stacked=False, legend=False,
                                                  figsize=(12, 12), fontsize=22)
    test_set_result.to_csv("../results/kfold/%s_external_test_set_plot_metrics.csv" % (name))
    plt.legend(loc=3, prop={'size': 18})
    plt.title(' %s external test set results ' % (name), fontsize=20)
    plt.savefig("../results/kfold/%s_external_test_set_plot_metrics.png" % (name))


if __name__ == "__main__":
    # make picture
    #test_auc();
    external_test_set(X_external=X_Oat, y_external=y_Oat, name="Oat")
    external_test_set(X_external=X_Berry, y_external=y_Berry, name="Berry")