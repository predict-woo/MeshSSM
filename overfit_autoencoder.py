from lightning.pytorch.loggers import WandbLogger
from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from mesh_ssm.utils.augment import augment_mesh, normalize_mesh, sort_mesh_faces
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader, Dataset
import lightning as L


class TrainAugmentedDataset(Dataset):
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


class ValidationNormalizedDataset(Dataset):
    def __init__(self, obj_path, num_samples):
        self.obj_path = obj_path
        self.num_samples = num_samples
        self.verts, self.faces, _ = load_obj(obj_path)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces.verts_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        normalized_mesh = normalize_mesh(self.mesh)
        sorted_mesh = sort_mesh_faces(normalized_mesh)
        return sorted_mesh


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


# create one example tuple for test infernce with collate_meshes
def create_test_inference_tuple(obj_path):
    verts, faces, _ = load_obj(obj_path)
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    mesh = normalize_mesh(mesh)

    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    features = feature_extractor.get_derived_face_features(mesh)
    edges = feature_extractor.get_face_feature_graph(mesh)
    return faces, verts, features, edges


lr = 0.001
ks = 9
path = "results"
name = "final-single"
devices = [3]


wandb_logger = WandbLogger(
    log_model=False, project="mesh-ssm-opt", name=f"{name}-{lr}-{ks}"
)


trainer = L.Trainer(
    accelerator="gpu",
    devices=devices,
    logger=wandb_logger,
    max_epochs=100,
    log_every_n_steps=10,
)

model = MeshAutoencoder(
    lr=lr,
    ks=ks,
    path=path,
    name=name,
    test_inf_mesh=create_test_inference_tuple("Horse.obj"),
)

train_dataset = TrainAugmentedDataset("Horse.obj", 6400)
train_dataloader = DataLoader(
    train_dataset, batch_size=64, collate_fn=collate_meshes, num_workers=30
)

val_dataset = ValidationNormalizedDataset("Horse.obj", 1)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_meshes)

print("Dataset and dataloader created")

trainer.fit(model, train_dataloader, val_dataloader)
