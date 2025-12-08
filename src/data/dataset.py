# src/data/dataset.py
import torch
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self, data_path):
        # [수정] 보안 경고 제거 옵션 추가
        self.data = torch.load(data_path, weights_only=True)
        self.problems = self.data["problems"]
        self.solutions = self.data["solutions"]
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        return self.problems[idx], self.solutions[idx]