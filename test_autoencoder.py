from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
import lightning as L
from torch.utils.data import Dataset, DataLoader


class SingleObjDataset(Dataset):
    def __init__(self, obj_path, num_samples):
        self.obj_path = obj_path
        self.num_samples = num_samples
        self.verts, self.faces, _ = load_obj(obj_path)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces.verts_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        augmented_mesh = self.mesh
        return augmented_mesh


def collate_meshes(batch):
    verts = []
    faces = []
    for mesh in batch:
        verts.append(mesh.verts_list()[0])
        faces.append(mesh.faces_list()[0])
    return Meshes(verts=verts, faces=faces)


trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=1,
    log_every_n_steps=1,
)

model = MeshAutoencoder()
test_dataset = SingleObjDataset("tet.obj", 1)
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_meshes)

trainer.test(model, test_dataloader)
