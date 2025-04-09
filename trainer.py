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
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
from utils import DistillationLoss
import os

TRAIN_EPOCHS = 100

def train(writer, cpt_path='student_res_18.cpt', patience=12, dist_alpha=0, dist_temp=1):
    distill_loss = DistillationLoss(dist_alpha, dist_temp) # alpha=0, Temp=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=cpt_path)
    # get student and teacher models
    t1, t2 = get_teachers()
    s1 = get_student()
    t1, t2, s1 = t1.to(device), t2.to(device), s1.to(device) 
    # set correct modes for model
    s1.train(True)
    t1.eval()
    t2.eval()
    # get ciphar-10 data
    train_dl, val_dl = get_loaders(128, 6)
    # setup optimizers and scheduler
    learning_rate_init = 0.002
    optimizer = Adam(s1.parameters(), lr=learning_rate_init)
    scheduler = ReduceLROnPlateau(optimizer, 'min') # minimizing loss
    # setup inital teacher
    switch_counter = 3 # switch teacher every 3 epochs
    t1_select = False
    teacher = t2
    epoch_val_acc = -1
    for epoch in range(0, TRAIN_EPOCHS):
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
        loss, acc = train_epoch(s1, teacher, optimizer, 
                                scheduler, train_dl, writer, distill_loss, epoch, device)
        epoch_val_loss, epoch_val_acc = get_model_metrics(s1, val_dl, criterion=distill_loss,
                                                          teacher=teacher, device=device)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        print_info(epoch, loss, acc.item(), epoch_val_loss, epoch_val_acc)
        # check if we should stop early
        early_stopping(epoch_val_loss, s1)
        if early_stopping.early_stop:
            print("Early stopping..")
            break
        
# trains the model for a single epoch and returns loss/acc
def train_epoch(student: nn.Module, teacher: nn.Module, 
                optimizer, scheduler, 
                train_dl: torch.utils.data.DataLoader, writer, distill_loss, epoch, device='cpu'):
    student.train()
    teacher.eval()
    total_loss = 0
    total_correct = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        t_logits = None
        with torch.no_grad():
            t_logits = teacher(imgs)
        s_logits = student(imgs)
        # calculate acc and loss
        loss = distill_loss(s_logits, labels, t_logits)
        total_correct += torch.sum(s_logits.argmax(1) == labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    num_samples = len(train_dl.dataset)
    num_batches = len(train_dl)
    loss, acc = total_loss/num_batches, total_correct/num_samples
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    curr_lr = scheduler.get_last_lr()[0]
    writer.add_scalar("LR", curr_lr, epoch)
    scheduler.step(loss)
    return loss, acc

def print_info(epoch, loss, acc, epoch_val_loss, epoch_val_acc):
    print_str = f"Epoch {epoch} | Train(loss/acc)={round(loss,4)}/{round(acc,4)} | "
    print_str += f"Val(loss/acc)={round(epoch_val_loss,4)}/{round(epoch_val_acc,4)}"
    print(print_str)

def get_exp_alpha_name(dist_alpha):
    str_a = str(dist_alpha)
    if "." in str_a:
        # check if first digit is 0
        if str_a[0] == "0":
            return int(str_a.split('.')[1])
        else:
            return 1
    else:
        return dist_alpha

# TODO
# add command args to run on slurm
# So far: 0.002 with val acc=0.817, 15 epochs
# include early stopping based on validation set
if __name__ == "__main__":
    checkpoints_folder = 'checkpoints/'
    os.makedirs(checkpoints_folder, exist_ok=True)
    base_log_dir = './logs'
    # experiment params
    dist_temp = 1
    dist_alpha = 0.4
    assert dist_alpha <= 1, "Distillation alpha should be <= 1"
    assert dist_alpha >= 0, "Distillation alpha should be >= 1"
    assert dist_temp == int(dist_temp), "Distillation temp should an integer"
    
    exp_alpha_name = get_exp_alpha_name(dist_alpha)

    exp_name = f'stu_resnet18_a_{exp_alpha_name}_t_{dist_temp}'
    log_dir = base_log_dir + '/' + exp_name
    writer = SummaryWriter(log_dir=log_dir)
    early_stop_cpt_path = checkpoints_folder+exp_name+'.cpt'
    train(writer, cpt_path=early_stop_cpt_path, patience=18, dist_alpha=dist_alpha, dist_temp=dist_temp)
    writer.close()