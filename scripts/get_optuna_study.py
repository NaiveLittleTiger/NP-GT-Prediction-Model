import optuna
import joblib
import os

study_dir="../data/optuna_study/jtvae-seed-40/"

def read_study():
    datanames = os.listdir(study_dir)
    for dataname in datanames:
        if os.path.splitext(dataname)[1] == '.pkl':
            print(dataname)
            study=joblib.load(study_dir+ dataname)
            # print(study.best_trial.params)
            print("Best trial until now:")
            print(" Value: ", study.best_trial.value)
            print(" Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

if __name__ == "__main__":
    read_study()
