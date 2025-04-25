import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from student import get_student
from teachers import get_teachers
from blackbox import get_blackbox
from utils.get_data import get_loaders
from utils import get_model_metrics
from torch.nn import CrossEntropyLoss
from attacks import attacks
from utils import get_model_metrics_batch
import torchvision
torch.manual_seed(123)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_model = get_student(device=device, pretrained=True)
    # attack_model.load_state_dict(torch.load('checkpoints/stu_resnet18_a_0_t_3.cpt'))
    # attack_model.load_state_dict(torch.load('checkpoints/resnet-18.cpt'))
    attack_model.eval()
    criterion = CrossEntropyLoss()
    
    # load teachers
    t_resnet, t_densenet = get_teachers()
    t_resnet, t_densenet = t_resnet.to(device), t_densenet.to(device)
    
    _, val_dl = get_loaders(64, 2)
    dataiter = iter(val_dl)
    images, labels = next(dataiter) # get batch of data
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True
    out = t_resnet(images)
    loss = criterion(out, labels)
    loss.backward()
    temp = images.grad.data.clone()
    images.grad = None
    out = t_densenet(images)
    loss = criterion(out, labels)
    loss.backward()
    _grad = (temp + images.grad.data)/2
    adv_images = attacks.fgsm_attack(images, _grad, epsilon=9/255)
        
    # load blackbox
    black_box = get_blackbox()
    get_model_metrics_batch(black_box, images, adv_images, labels, device=device)
    

