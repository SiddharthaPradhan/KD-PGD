import torch.nn as nn
import torch
from models.resnet import resnet18, resnet50
from get_data import get_loaders
from utils import get_model_metrics

'''
Student Network -- Most cases this will be ResNet18
'''
def get_student(type='resnet18', device='cpu', pretrained=False):
    model = None
    if type == 'resnet18':
        model =  resnet18(pretrained)
    elif type == 'resnet50':
        model = resnet50
    else:
        raise NotImplementedError(f"Student {type} has not been configured in student.py")
    return model.to(device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, test_dl = get_loaders(32, 4)
    student =  get_student(pretrained=True) # if we are training a student --> pretrained=False
    print("Resnet18 Test Acc:", get_model_metrics(student, test_dl, device=device))