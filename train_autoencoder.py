from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from pytorch3d.io import load_obj
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.structures import Meshes
from mesh_ssm.utils.augment import augment_mesh, normalize_mesh
from data import AugmentedHorseMeshDataset, collate_meshes
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

TEST = False

if TEST:
    verts, faces, aux = load_obj("Horse.obj", load_textures=False)

    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    meshes = normalize_mesh(meshes)
    autoencoder = MeshAutoencoder()
    autoencoder.setup()
    autoencoder.training_step(meshes, 0)
    exit()

wandb_logger = WandbLogger(log_model="all", project="mesh-ssm")

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    logger=wandb_logger,
    max_epochs=100,
    log_every_n_steps=1,
)
model = MeshAutoencoder()

train_dataset = AugmentedHorseMeshDataset("cone.obj", 100)
train_dataloader = DataLoader(
    train_dataset, batch_size=1, collate_fn=collate_meshes, num_workers=10
)

val_dataset = AugmentedHorseMeshDataset("cone.obj", 1)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_meshes)


print("Dataset and dataloader created")


trainer.fit(model, train_dataloader, val_dataloader)
