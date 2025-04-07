from utils import get_model_metrics
from models import googlenet
from get_data import get_loaders, classes

'''returns the "black-box" GoogLeNet Model'''
def get_blackbox():
    return googlenet.googlenet(True)
    
if __name__ == "__main__":
    train_dl, test_dl = get_loaders(32, 4)
    blackbox =  get_blackbox()
    print("GoogLeNet Test Acc:", get_model_metrics(blackbox, train_dl, 'cuda'))
    
