'''
This file handles the training of the student model
'''
import optuna
import torch.nn as nn
import torch
from teachers import get_teachers
from student import get_student
from get_data import get_loaders
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_model_metrics, distillation_loss


TUNE_EPOCHS = 15

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get student and teacher models
    t1, t2 = get_teachers()
    s1 = get_student()
    t1, t2, s1 = t1.to(device), t2.to(device), s1.to(device) 
    s1 = s1.train(True)
    t1, t2 = t1.eval(), t2.eval()
    # get ciphar-10 data
    train_dl, val_dl = get_loaders(128, 6)
    # setup optimizers and scheduler
    learning_rate_init = trial.suggest_float(
        "learning_rate_init", 1e-4, 1e-2
    )
    optimizer = Adam(s1.parameters(), lr=learning_rate_init)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # setup inital teacher
    switch_counter = 3 # switch teacher every 3 epochs
    t1_select = False
    teacher = t2
    epoch_val_acc = -1
    for i in range(1, TUNE_EPOCHS+1):
        # lets assume for now simple schedule
        # teachers switch every 3 epochs
        # TODO move this logic to a teacher_schedule function
        switch_counter -= 1
        if switch_counter == 0:
            switch_counter = 3
            if t1_select: # t1 was previous teacher
                teacher = t2
                t1_select = False
            else: # t2 was previous teacher
                teacher = t1
                t1_select = True
        _, acc = train_epoch(s1, teacher, optimizer, scheduler, train_dl, device)
        _, epoch_val_acc = get_model_metrics(s1, val_dl, device=device)
        trial.report(epoch_val_acc, i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epoch_val_acc


# trains the model for a single epoch and returns loss/acc
def train_epoch(student: nn.Module, teacher: nn.Module, 
                optimizer, scheduler, 
                train_dl: torch.utils.data.DataLoader, device='cpu'):
    student.train()
    teacher.eval()
    total_loss = 0
    total_correct = 0
    for idx, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(device), labels.to(device)
        t_logits = None
        with torch.no_grad():
            t_logits = teacher(imgs)
        s_logits = student(imgs)
        # calculate acc and loss
        loss = distillation_loss(s_logits, labels, t_logits, 0, 1)
        total_correct += torch.sum(s_logits.argmax(1) == labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    num_samples = len(train_dl.dataset)
    num_batches = len(train_dl)
    loss, acc = total_loss/num_batches, total_correct/num_samples
    scheduler.step(loss)
    return loss, acc


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="student-1-resnet-18-simple",
        direction='maximize',
        load_if_exists=True
    )
    timeout = 12*60*60 # 12 hrs
    study.optimize(objective, n_trials=20, timeout=timeout)
    print(f"Best value: {study.best_value} (params: {study.best_params})") 
    # So far: 0.002 with val acc=0.817, 15 epochs