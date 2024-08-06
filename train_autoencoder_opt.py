from lightning.pytorch.loggers import WandbLogger
from mesh_ssm.models.mesh_autoencoder_opt import MeshAutoencoderOpt
from mesh_ssm.utils.augment import augment_mesh
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader, Dataset
import lightning as L


class TestAugmentedDataset(Dataset):
    def __init__(self, obj_path, num_samples):
        self.obj_path = obj_path
        self.num_samples = num_samples
        self.verts, self.faces, _ = load_obj(obj_path)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces.verts_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        augmented_mesh = augment_mesh(self.mesh)
        return augmented_mesh


feature_extractor = FaceFeatureExtractor()


def collate_meshes(batch):
    verts_list = []
    faces_list = []
    for augmented_mesh in batch:
        faces = augmented_mesh.faces_packed()
        verts = augmented_mesh.verts_packed()
        verts_list.append(verts)
        faces_list.append(faces)

    batched_mesh = Meshes(verts=verts_list, faces=faces_list)

    faces = batched_mesh.faces_packed()
    verts = batched_mesh.verts_packed()
    features = feature_extractor.get_derived_face_features(batched_mesh)
    edges = feature_extractor.get_face_feature_graph(batched_mesh)
    return faces, verts, features, edges


wandb_logger = WandbLogger(log_model="all", project="mesh-ssm-opt")

trainer = L.Trainer(
    accelerator="gpu",
    devices=[3],
    logger=wandb_logger,
    max_epochs=100,
    log_every_n_steps=10,
)

model = MeshAutoencoderOpt()

train_dataset = TestAugmentedDataset("Horse.obj", 6400)
train_dataloader = DataLoader(
    train_dataset, batch_size=64, collate_fn=collate_meshes, num_workers=90
)

val_dataset = TestAugmentedDataset("Horse.obj", 1)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_meshes)


print("Dataset and dataloader created")


trainer.fit(model, train_dataloader, val_dataloader)
