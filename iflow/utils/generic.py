import torch
import os

def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

