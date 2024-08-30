import os
import torch
from torch.utils.data import DataLoader
from mesh_ssm.models.mesh_gpt import MeshGPT
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import GPT2Config

from data import EncodedChairDataset

name = "chair_gpt_global_codebook"
lr = 1e-3
weight_decay = 0.01
batch_size = 12

wandb_logger = WandbLogger(
    log_model=False,
    project="mesh-ssm-gpt",
    name=f"{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}]",
)

# check if checkpoints/{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}] exists
if not os.path.exists(
    f"checkpoints/{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}]"
):
    os.makedirs(f"checkpoints/{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}]")

# Checkpoint callback to save the model every 10 epochs
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    every_n_epochs=1,
    dirpath=f"checkpoints/{name}-lr[{lr}]-wd[{weight_decay}]-bs[{batch_size}]",
    filename="{epoch:02d}-{val_loss:.2f}",
    verbose=True,
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=[5, 6, 7, 8],
    # devices=[2],
    logger=wandb_logger,
    max_epochs=1000,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
)

config = GPT2Config(
    vocab_size=1024 + 3,
    n_embd=384,
    n_layer=6,
    n_head=6,
    n_positions=800 * 3 * 2 + 2,
)


model = MeshGPT(config, lr=lr, weight_decay=weight_decay)


train_dataset = EncodedChairDataset(
    "encoded_chair_dataset.npy",
    max_length=800 * 2 * 3 + 2,
    train_size=0.8,
    split="train",
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=30
)

val_dataset = EncodedChairDataset(
    "encoded_chair_dataset.npy",
    max_length=800 * 2 * 3 + 2,
    train_size=0.8,
    split="val",
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=30
)

trainer.fit(model, train_dataloader, val_dataloader)
