import os
from torch.utils.data import Dataset, DataLoader
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from mesh_ssm.utils.augment import augment_mesh


class AugmentedHorseMeshDataset(Dataset):
    def __init__(self, obj_path, num_samples):
        self.obj_path = obj_path
        self.num_samples = num_samples
        self.verts, self.faces, _ = load_obj(obj_path)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces.verts_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # augmented_mesh = augment_mesh(self.mesh)
        augmented_mesh = self.mesh
        return augmented_mesh


def collate_meshes(batch):
    verts = []
    faces = []
    for mesh in batch:
        verts.append(mesh.verts_list()[0])
        faces.append(mesh.faces_list()[0])
    return Meshes(verts=verts, faces=faces)
