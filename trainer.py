'''
This file handles the training of the student model
'''
import optuna
import torch.nn as nn


def objective(trial):
    pass

# trains the model for a single epoch and returns loss/acc
def train_epoch():
    pass

# TODO:
'''
- ReduceLROnPlateau
- ADAM
- Use Optuna For hyperparameter tuning


'''
if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="student-1-resnet-18-simple"
    )
    timeout = 12*60*60 # 12 hrs
    study.optimize(objective, n_trials=2000, timeout=timeout)
    print(f"Best value: {study.best_value} (params: {study.best_params})")