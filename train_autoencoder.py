import lightning as L
import os
import random
import argparse
from lightning.pytorch.loggers import WandbLogger
from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from mesh_ssm.utils.augment import augment_mesh
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import ModelCheckpoint


class AugmentedChairDataset(Dataset):
    def __init__(self, data_dir, target_size, val_split=0.1, is_train=True):
        self.data_dir = data_dir
        self.target_size = target_size
        self.val_split = val_split
        self.is_train = is_train

        self.obj_files = [f for f in os.listdir(data_dir) if f.endswith(".obj")]
        self.num_obj_files = len(self.obj_files)

        if self.num_obj_files == 0:
            raise ValueError(f"No OBJ files found in {data_dir}")

        # Shuffle and split the obj_files
        random.shuffle(self.obj_files)
        split_idx = int(self.num_obj_files * (1 - self.val_split))

        print(f"Split index: {split_idx}")

        if self.is_train:
            self.obj_files = self.obj_files[:split_idx]
        else:
            self.obj_files = self.obj_files[split_idx:]

        self.num_obj_files = len(self.obj_files)

        self.meshes = []
        for obj_file in self.obj_files:
            verts, faces, _ = load_obj(os.path.join(data_dir, obj_file))
            self.meshes.append(Meshes(verts=[verts], faces=[faces.verts_idx]))

        # Adjust target size based on split
        if self.is_train:
            self.target_size = int(target_size * (1 - val_split))
        else:
            self.target_size = int(target_size * val_split)

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


def main(args):
    wandb_logger = WandbLogger(
        log_model=False,
        project=args.project,
        name=f"{args.name}-bs[{args.batch_size}]-lr[{args.lr}]-ks[{args.ks}]",
    )

    if not os.path.exists(
        f"checkpoints/{args.name}-bs[{args.batch_size}]-lr[{args.lr}]-ks[{args.ks}]"
    ):
        os.makedirs(
            f"checkpoints/{args.name}-bs[{args.batch_size}]-lr[{args.lr}]-ks[{args.ks}]"
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation reconstruction loss",
        mode="min",
        save_top_k=3,
        every_n_epochs=1,
        dirpath=f"checkpoints/{args.name}-bs[{args.batch_size}]-lr[{args.lr}]-ks[{args.ks}]",
        filename="{epoch:02d}-{val_loss:.4f}",
        verbose=True,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger,
        max_epochs=1000,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    model = MeshAutoencoder(
        lr=args.lr,
        ks=args.ks,
        path=args.path,
        name=args.name,
    )

    train_dataset = AugmentedChairDataset("chair_dataset", 10000)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_meshes,
        num_workers=30,
    )

    val_dataset = AugmentedChairDataset("chair_dataset", 10000, is_train=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_meshes,
        num_workers=30,
    )

    print("Dataset and dataloader created")

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Autoencoder Training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--ks", type=int, default=3, help="Kernel size")
    parser.add_argument("--path", type=str, default="results", help="Results path")
    parser.add_argument(
        "--name",
        type=str,
        default="chair-batch-recons-multigpu",
        help="Experiment name",
    )
    parser.add_argument(
        "--project", type=str, default="project-name", help="Project name"
    )
    parser.add_argument(
        "--devices", nargs="+", type=int, default=[2], help="GPU devices to use"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()
    main(args)
