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


class ChairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_size,
    ):
        self.data_dir = data_dir
        self.target_size = target_size

        self.obj_files = [f for f in os.listdir(data_dir) if f.endswith(".obj")]
        self.num_obj_files = len(self.obj_files)

        if self.num_obj_files == 0:
            raise ValueError(f"No OBJ files found in {data_dir}")

        # Shuffle and split the obj_files
        random.shuffle(self.obj_files)

        self.meshes = []
        for obj_file in self.obj_files:
            verts, faces, _ = load_obj(os.path.join(data_dir, obj_file))
            self.meshes.append(Meshes(verts=[verts], faces=[faces.verts_idx]))

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        obj_idx = idx % self.num_obj_files
        mesh = self.meshes[obj_idx]
        augmented_mesh = augment_mesh(mesh)
        return augmented_mesh


feature_extractor = FaceFeatureExtractor()


def collate_meshes(batch):
    batched_mesh = join_meshes_as_batch(batch)

    num_verts_per_mesh = (
        batched_mesh.num_verts_per_mesh()
    )  # Number of vertices in each mesh
    num_faces_per_mesh = (
        batched_mesh.num_faces_per_mesh()
    )  # Number of faces in each mesh

    faces = batched_mesh.faces_packed()
    verts = batched_mesh.verts_packed()
    features = feature_extractor.get_derived_face_features(batched_mesh)
    edges = feature_extractor.get_face_feature_graph(batched_mesh)
    return faces, verts, features, edges, num_verts_per_mesh, num_faces_per_mesh


model = MeshAutoencoder.load_from_checkpoint(
    "mesh-ssm-chair/0854fxck/checkpoints/epoch=499-step=70500.ckpt"
)
model.to("cuda:3")
model.setup()
model.eval()


dataset = ChairDataset(
    "chair_dataset",
    10000,
)

# Add tqdm progress bar
for i in tqdm(range(10000), desc="Processing meshes"):
    mesh = dataset[i]
    inp = collate_meshes([mesh])
    inp = tuple(map(lambda x: x.to("cuda:3"), inp))
    # encode the mesh
    tokens = model.encode(inp)
    torch.save(tokens, f"chair_encoded_dataset/{i}.pt")
