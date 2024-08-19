import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from mesh_ssm.models.mesh_gpt import MeshGPT
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from transformers import GPT2Config


class FolderDataset(Dataset):
    def __init__(self, folder, max_length=800 * 3 * 2 + 2):
        self.files = os.listdir(folder)
        self.folder = folder
        self.max_length = max_length
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        print(f"FolderDataset: init complete, {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load data
        input_ids = np.load(f"{self.folder}/{self.files[idx]}")

        # adjust data
        input_ids[:, 1] += 192  # Adjusting the second column as described
        input_ids = input_ids.flatten()
        input_ids += 3  # avoid using 0, 1, 2 as tokens

        # add start and end tokens
        input_ids = np.concatenate(
            [
                np.array([self.start_token], dtype=np.int64),
                input_ids,
                np.array([self.end_token], dtype=np.int64),
            ]
        )

        # Truncate or pad the sequence to the desired length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            print("what the FUCK")
        else:
            padding = np.full(
                (self.max_length - len(input_ids),), self.pad_token, dtype=np.int64
            )
            input_ids = np.concatenate((input_ids, padding))

        input_ids = input_ids.astype(np.int64)
        attention_mask = (input_ids != self.pad_token).astype(
            np.int64
        )  # Create attention mask based on non-zero tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


name = "chair_gpt"
lr = 1e-4
weight_decay = 0.01
batch_size = 12

wandb_logger = WandbLogger(
    log_model=False,
    project="mesh-ssm-gpt",
    name=f"{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}]",
)


trainer = L.Trainer(
    accelerator="gpu",
    devices=[6, 7, 8, 9],
    logger=wandb_logger,
    max_epochs=1000,
    log_every_n_steps=1,
)


config = GPT2Config(
    vocab_size=192 * 2 + 3,
    n_embd=384,
    n_layer=6,
    n_head=6,
    n_positions=800 * 3 * 2 + 2,
)


model = MeshGPT(config, lr=lr, weight_decay=weight_decay)


train_dataset = FolderDataset("chair_encoded_dataset_np", max_length=800 * 2 * 3 + 2)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=30
)

trainer.fit(model, train_dataloader)
