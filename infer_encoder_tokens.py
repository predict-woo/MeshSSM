from tools import Tools
from data import SimplifiedChairDataset
from tqdm import tqdm
import torch
import numpy as np

tools = Tools("cuda:8")

tools.init_autoencoder(
    "checkpoints/chair-final-bs[128]-lr[0.001]-ks[3]/epoch=987-val_loss=0.0000.ckpt"
)

dataset = SimplifiedChairDataset(
    "chair_dataset",
    10000,
)

# Add tqdm progress bar
array_list = []
for i in tqdm(range(10000), desc="Processing meshes"):
    mesh = dataset[i]
    tokens = tools.encode_mesh(mesh)
    array_list.append(tokens.cpu().numpy())


np.save(
    "encoded_chair_dataset.npy", np.array(array_list, dtype=object), allow_pickle=True
)
