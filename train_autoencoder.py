from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from pytorch3d.io import load_obj
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.structures import Meshes
from mesh_ssm.utils.augment import augment_mesh, normalize_mesh
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.profilers import PyTorchProfiler
import torch


class AugmentedHorseMeshDataset(Dataset):
    def __init__(self, obj_path, num_samples):
        self.obj_path = obj_path
        self.num_samples = num_samples
        self.verts, self.faces, _ = load_obj(obj_path)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces.verts_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        augmented_mesh = augment_mesh(self.mesh)
        # augmented_mesh = self.mesh
        return augmented_mesh


def collate_meshes(batch):
    verts = []
    faces = []
    for mesh in batch:
        verts.append(mesh.verts_packed())
        faces.append(mesh.faces_packed())
    return Meshes(verts=verts, faces=faces)


wandb_logger = WandbLogger(log_model="all", project="mesh-ssm")
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    trace_memory=True,
    schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=0, active=1),
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=[1],
    logger=wandb_logger,
    max_epochs=3,
    log_every_n_steps=1,
    profiler=profiler,
)
model = MeshAutoencoder()

train_dataset = AugmentedHorseMeshDataset("tea.obj", 1600)
train_dataloader = DataLoader(
    train_dataset, batch_size=16, collate_fn=collate_meshes, num_workers=30
)

val_dataset = AugmentedHorseMeshDataset("tea.obj", 1)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_meshes)


print("Dataset and dataloader created")


trainer.fit(model, train_dataloader, val_dataloader)
