# src/utils.py
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    if targets.dim() == 3: 
        targets = targets.view(targets.size(0), -1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total