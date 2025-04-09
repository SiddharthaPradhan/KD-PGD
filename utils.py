import torch
import torch.nn as nn
import  torch.nn.functional as F

# Calculate metrics for the dl
def get_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      criterion=None, teacher=None, device='cpu'):
    acc = 0
    loss = 0
    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            out: torch.Tensor = model(images)
            pred = out.argmax(1)
            acc += torch.sum(pred == labels)
            if criterion is not None:
                teacher_out = teacher(images)
                loss += criterion(out, labels, teacher_out)
    acc = acc/len(dataloader.dataset) # per sample acc 
    if criterion is not None:
        loss = loss/len(dataloader) # per sample loss
        loss = loss.item()
    return loss, acc.item()



# define the loss for distillation
def distillation_loss(out, labels, teacher_logits, alpha = 0, softmax_temp=1):
    """Loss for the project

    Args:
        out (torch.Tensor): Student model output
        labels (torch.Tensor): Ground truth label
        teacher_logits (torch.Tensor): Teacher model output
        alpha (float): Controls the tradeoff between teacher knowledge and student self-learning. Defaults to 0.
        softmax_temp (int, optional): _description_. Defaults to 1.
    Returns:
        loss: returns a*(hard_loss) + (1-a)*soft_loss
    """
    student_log_probs = F.log_softmax(out / softmax_temp, dim=1)
    teacher_probs = F.softmax(teacher_logits / softmax_temp, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (softmax_temp ** 2)
    hard_loss =  nn.CrossEntropyLoss(reduction='mean')(out, labels)
    loss = alpha*(hard_loss) + (1-alpha)*soft_loss
    return loss


# simple class to store dist_loss hparams
class DistillationLoss:
    def __init__(self, alpha = 0, softmax_temp=1):
        self.alpha = alpha
        self.softmax_temp = softmax_temp

    def __call__(self, out, labels, teacher_logits):
        return distillation_loss(out, labels, teacher_logits, alpha=self.alpha, softmax_temp=self.softmax_temp)