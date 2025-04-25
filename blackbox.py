from utils.utils import get_model_metrics
from models import googlenet
from utils.get_data import get_loaders
import torch

'''returns the "black-box" GoogLeNet Model'''
def get_blackbox():
    return googlenet.googlenet(True)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, test_dl = get_loaders(32, 4)
    blackbox =  get_blackbox()
    print("GoogLeNet Test Acc:", get_model_metrics(blackbox, test_dl, device=device))
