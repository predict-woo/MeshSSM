import os
import random
from mesh_ssm.utils.augment import augment_mesh
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

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


class SimplifiedChairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_size,
        augmentation=True,
    ):
        self.data_dir = data_dir
        self.target_size = target_size
        self.augmentation = augmentation

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
        if self.augmentation:
            augmented_mesh = augment_mesh(mesh)
        else:
            augmented_mesh = mesh
        return augmented_mesh


class EncodedChairDataset(Dataset):
    def __init__(
        self,
        path,
        max_length=800 * 3 * 2 + 2,
        split="train",
        train_size=0.8,
        seed=42,
    ):
        self.max_length = max_length
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2

        self.dataset = np.load(path, allow_pickle=True)
        # shuffle
        np.random.shuffle(self.dataset)

        # Split the files into training and validation sets
        train_files, val_files = train_test_split(
            self.dataset, train_size=train_size, random_state=seed
        )

        if split == "train":
            self.dataset = train_files
        elif split == "val":
            self.dataset = val_files
        else:
            raise ValueError("split should be either 'train' or 'val'")

        print(f"FolderDataset ({split}): init complete, {len(self.dataset)} files")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # load data

        input_ids = self.dataset[idx]

        # Check if input_ids has the expected shape and dtype
        if len(input_ids.shape) != 2 or input_ids.shape[1] < 2:
            raise ValueError(f"Unexpected input shape for file")

        # adjust data
        input_ids = input_ids.flatten()
        input_ids += 3  # avoid using 0, 1, 2 as tokens

        # add start and end tokens
        input_ids = np.concatenate(
            [
                np.array([self.start_token], dtype=np.int64),
                input_ids,
                np.array([self.end_token], dtype=np.int64),
            ]
        )

        # Truncate or pad the sequence to the desired length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            print("Truncated input_ids to max_length")
        else:
            padding = np.full(
                (self.max_length - len(input_ids),), self.pad_token, dtype=np.int64
            )
            input_ids = np.concatenate((input_ids, padding))

        input_ids = input_ids.astype(np.int64)
        attention_mask = (input_ids != self.pad_token).astype(
            np.int64
        )  # Create attention mask based on non-zero tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = EncodedChairDataset(
        "encoded_chair_dataset.npy",
        max_length=800 * 2 * 3 + 2,
        train_size=0.8,
        split="train",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=30
    )
    for batch in train_dataloader:
        print(batch)
        break
