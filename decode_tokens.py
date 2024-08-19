import os
import random
from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from mesh_ssm.utils.augment import augment_mesh
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch.utils.data import Dataset
import torch
from tqdm import tqdm  # Import tqdm for the progress bar
from mesh_ssm.utils.render import reconstruct_mesh
from pytorch3d.io import save_obj


class FolderDataset(Dataset):
    def __init__(self, folder):
        self.files = os.listdir(folder)
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(f"{self.folder}/{self.files[idx]}")


model = MeshAutoencoder.load_from_checkpoint(
    "mesh-ssm-chair/0854fxck/checkpoints/epoch=499-step=70500.ckpt"
)
model.to("cuda:3")
model.setup()
model.eval()


dataset = FolderDataset(
    "chair_encoded_dataset",
)

# Add tqdm progress bar
for i in tqdm(range(len(dataset)), desc="decoding meshes"):
    all_min_encoding_indices = dataset[i]
    print(all_min_encoding_indices)
    all_min_encoding_indices = all_min_encoding_indices.to("cuda:3")

    z_s = []
    for j, quantizer in enumerate(model.rq.vq_layers):
        # Get the quantized vectors for the i-th quantizer
        indices = all_min_encoding_indices[:, j]
        z_res = quantizer.get_codebook_entry(indices)
        # Accumulate the quantized vectors
        z_s.append(z_res)

    z_s = torch.stack(z_s, dim=1)
    z_q = torch.sum(z_s, dim=1)

    # encode the mesh
    recons = model.decode(z_q)

    mesh = reconstruct_mesh(recons)
    save_obj(
        f"decoded_meshes/{i}.obj",
        mesh.verts_list()[0],
        mesh.faces_list()[0],
    )
