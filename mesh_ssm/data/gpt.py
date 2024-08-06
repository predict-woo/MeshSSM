import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = torch.tensor(sequence[:-1], dtype=torch.float32)
        targets = torch.tensor(sequence[1:], dtype=torch.float32)
        return inputs, targets
