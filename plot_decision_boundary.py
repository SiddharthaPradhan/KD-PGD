'''
Plot decision boundaries of multiple models for a given image.
'''
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, Subset
from utils.get_data import get_loaders, classes, inv_normalize, normalize
import random
from models.teachers import get_teachers
from models.student import get_student
from models.blackbox import get_blackbox
import torchvision
from tqdm import tqdm
from itertools import product
# setup seeds and make deterministic
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_selected_images(cat_idx, plane_idx):
    # Note we are using test set as the validation set, so val --> test
    _, val_loader = get_loaders(2, 3)
    val_dataset : Dataset = val_loader.dataset
    labels  = val_dataset.targets
    dataset_cat_idx = np.where(labels == cat_idx)[0]
    dataset_plane_idx = np.where(labels == plane_idx)[0]
    cat_subset = Subset(val_dataset, dataset_cat_idx)
    cat_images = [cat_img for cat_img, _ in cat_subset]
    plane_subset = Subset(val_dataset, dataset_plane_idx)
    plane_images = [plane_img for plane_img, _ in plane_subset]
    # select a good looking image for the 2 classes
    sel_cat_img = cat_images[30]
    sel_plane_img = plane_images[98]
    return sel_cat_img.unsqueeze(0), sel_plane_img.unsqueeze(0)

def generate_adversarial_direction(model, image, label):
    """
    Generate a normalized FGSM adversarial direction.
    """
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    loss.backward()
    grad = image.grad.data
    direction = grad / torch.norm(grad)
    return direction.detach()

def orthogonalize_direction(base_dir, new_dir):
    """
    Make new_dir orthogonal to base_dir using Gram-Schmidt process.
    """
    proj = (torch.sum(new_dir * base_dir) / torch.sum(base_dir * base_dir)) * base_dir
    orth_dir = new_dir - proj
    return orth_dir / torch.norm(orth_dir)


def plot_boundary(img:torch.Tensor, true_label:int,
                  models_dict:dict[str,torch.nn.Module],
                  models_colors:dict[str, str],
                  max_range=0.5,
                  default_model='GoogLeNet(Blackbox)'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blackbox =  get_blackbox().to(device)
    # _, densenet_t = get_teachers()
    # densenet_t = densenet_t.to(device)
    # get directions
    # TODO dir1 grad direction for googlenet
    # TODO dir2 orthogonal to dir1 with grad-smith process
    label = torch.Tensor([true_label+3]).to(device)
    label = label.type(torch.int64).to(device)
    # dir1 = generate_adversarial_direction(models_dict.get(default_model), 
    dir1 = generate_adversarial_direction(blackbox, 
                                          img.clone().to(device), label)
    random_noise = torch.randn_like(img).to(device)
    dir2 = orthogonalize_direction(dir1, random_noise)
    # setup grid
    alphas = np.linspace(-max_range, max_range, 100)
    # alphas = np.linspace(-max_range, max_range, 100)
    betas = np.linspace(-max_range, max_range, 100)
    # betas = np.linspace(-max_range, max_range, 100)
    X, Y = np.meshgrid(alphas, betas) # create 2d mesh of alpha and beta values
    fig, ax = plt.subplots()
    true_count = 0 
    total = 0
    for name, model in models_dict.items():
        model=blackbox
        # check if model can correctly predict this image first
        with torch.no_grad():
            temp_img = img.to(device)
            out = model(temp_img)
            pred = torch.argmax(out, dim=1).item()
            print(name, pred == true_label)
            # collage = inv_normalize(temp_img.squeeze(0).cpu()).numpy()
            # print(temp_img.min(), temp_img.max())
            # plt.imshow(np.transpose(collage, (1, 2, 0)))
            # plt.show()
        region = np.zeros_like(X)
        for i, j in tqdm(product(range(X.shape[0]), range(X.shape[1])), total=X.shape[0]*X.shape[1]):
            perturbation = X[i, j] * dir1 + Y[i, j] * dir2
            # TODO check this logic, clamping with normalization?
            # theoretical true min= -2.0180698152
            # theoretical true max= 2.11582568807
            true_min = -2.0180698152
            true_max = 2.11582568807
            new_img = inv_normalize(img.to(device))
            # print(perturbation.min(), perturbation.max())
            # print(new_img.min(), new_img.max())
            new_img = new_img + perturbation
            new_img = torch.clamp(new_img, 0, 1)
            # collage = new_img.squeeze(0).cpu().numpy()
            # plt.imshow(np.transpose(collage, (1, 2, 0)))
            # plt.show()
            new_img = normalize(new_img)
            new_img = torch.clamp(new_img, true_min, true_max)
            
            with torch.no_grad():
                out = model(new_img)
                pred = torch.argmax(out, dim=1).item()
                total += 1
                true_count += int(pred == true_label)
            region[i, j] = (pred == true_label)

        # add contour for current model
        ax.contour(X, Y, region, levels=[0.5], colors=[models_colors[name]],
                linewidths=1.5)
    
    print(true_count, total)
    legend_lines = [Line2D([0], [0], color=color, lw=2, label=name) 
                    for name, color in models_colors.items()]
    ax.legend(handles=legend_lines)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title('Decision Boundary Visualization')
    ax.set_xlabel('Direction 1 (pixels)')
    ax.set_ylabel('Direction 2 (pixels)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # plane and cat :) for images
    cat_idx = np.argwhere(np.array(classes) == 'cat')[0,0]
    plane_idx = np.argwhere(np.array(classes) == 'plane')[0,0]
    print(f"cat index={cat_idx}, airplane index={plane_idx}")
    sel_cat_img, sel_plane_img = get_selected_images(cat_idx, plane_idx)
    
    # TODO add selected student based on the evaluation results
    resnet_t, densenet_t = get_teachers()
    blackbox =  get_blackbox()
    
    models_dict = {
        # 'ResNet-50(Teacher-1)': resnet_t.to(device),
        'DenseNet-161(Teacher-2)': densenet_t.to(device),
        # 'GoogLeNet(Blackbox)': blackbox.to(device)
    }
    
    models_colors = {
        # 'ResNet-50(Teacher-1)': 'blue',
        'DenseNet-161(Teacher-2)': 'green',
        # 'GoogLeNet(Blackbox)': 'orange',
        # red for student
    }
    
    plot_boundary(sel_cat_img, cat_idx, models_dict, models_colors)
    
    


    

