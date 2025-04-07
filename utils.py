import torch

def get_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cpu'):
    acc = 0
    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            out: torch.Tensor = model(images)
            pred = out.argmax(1)
            acc += torch.sum(pred == labels)
    acc = acc/len(dataloader.dataset)
    return acc.item()