from models import resnet
from models import densenet
from get_data import get_loaders, classes
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import get_model_metrics

def get_teachers():
    resnet_t = resnet.resnet50(True) 
    densenet_t = densenet.densenet161(True)
    return resnet_t, densenet_t


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check train acc for both teachers
    resnet_t = resnet.resnet50(True) 
    densenet_t = densenet.densenet161(True)
    train_dl, test_dl = get_loaders(32, 4)
    print("ResNet50 Test Acc:", get_model_metrics(resnet_t, test_dl, device=device))
    print("DenseNet161 Test Acc:", get_model_metrics(densenet_t, test_dl, device=device))

    
            

