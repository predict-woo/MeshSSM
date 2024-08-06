from torch.optim import Adam
from transformers import GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader

# dataset


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


# model

config = GPT2Config(
    vocab_size=1,  # Not using a vocab
    n_positions=1024,  # max length of your sequences, adjust as necessary
    n_ctx=1024,
    n_embd=192,
    n_layer=12,
    n_head=12,
)

model = GPT2LMHeadModel(config)

# Modify the model's embedding layer to pass through the 192-dimensional input
model.transformer.wte = torch.nn.Identity()

# training

optimizer = Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.MSELoss()  # since you might be predicting continuous values


def train(model, data_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# Assuming 'sequences' is your dataset
dataset = TokenDataset()
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

train(model, data_loader, optimizer, criterion, epochs=10)
