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
from torch import Tensor

import inspect
import uuid
import numpy as np


class Tools:
    def __init__(self, device: str = "cuda:1"):
        self.device = device
        self.mesh_autoencoder = None
        self.feature_extractor = FaceFeatureExtractor()

        self.start_token = 1
        self.end_token = 2
        self.pad_token = 0

        self.max_length = 800 * 3 * 2 + 2

    def init_autoencoder(self, path: str):
        if self.mesh_autoencoder is None:
            self.log("loading decoder")
            with torch.cuda.device(self.device):
                self.mesh_autoencoder = MeshAutoencoder.load_from_checkpoint(
                    path,
                    map_location=self.device,
                )
                self.mesh_autoencoder.setup()
                self.mesh_autoencoder.eval()

    def log(self, message: str):
        caller_name = inspect.currentframe().f_back.f_code.co_name
        print(f"[{caller_name}] {message}")

    def infer_sequence_ssm(self, sequence: Tensor):

        pass

    def collate_meshes(self, batch):
        batched_mesh = join_meshes_as_batch(batch)

        num_verts_per_mesh = (
            batched_mesh.num_verts_per_mesh()
        )  # Number of vertices in each mesh
        num_faces_per_mesh = (
            batched_mesh.num_faces_per_mesh()
        )  # Number of faces in each mesh

        faces = batched_mesh.faces_packed()
        verts = batched_mesh.verts_packed()
        features = self.feature_extractor.get_derived_face_features(batched_mesh)
        edges = self.feature_extractor.get_face_feature_graph(batched_mesh)
        return faces, verts, features, edges, num_verts_per_mesh, num_faces_per_mesh

    def encode_mesh(self, mesh: Tensor):
        with torch.cuda.device(self.device):
            inp = self.collate_meshes([mesh])
            inp = tuple(map(lambda x: x.to(self.device), inp))
            # encode the mesh
            tokens = self.mesh_autoencoder.encode(inp)
            return tokens

    def save_tokens(self, tokens: Tensor, path: str):
        # save tokens as numpy array
        np.save(path, tokens.cpu().numpy())

    def load_tokens(self, path: str):
        # load tokens as numpy array
        return np.load(path)

    def tokens_to_sequence(self, tokens: np.ndarray):
        if len(tokens.shape) != 2 or tokens.shape[1] < 2:
            raise ValueError(f"Unexpected input shape for file")

        tokens = tokens.flatten()
        tokens += 3  # avoid using 0, 1, 2 as tokens

        # add start and end tokens
        tokens = np.concatenate(
            [
                np.array([self.start_token], dtype=np.int64),
                tokens,
                np.array([self.end_token], dtype=np.int64),
            ]
        )

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            self.log("Truncated tokens to max_length")
        else:
            padding = np.full(
                (self.max_length - len(tokens),), self.pad_token, dtype=np.int64
            )
            tokens = np.concatenate((tokens, padding))

        sequence = tokens.astype(np.int64)
        return sequence

    def sequence_to_tokens(self, sequence: np.ndarray):
        # remove start and end tokens
        seq = torch.tensor(sequence)
        end_idx = (seq == self.end_token).nonzero()
        self.log(f"end_idx: {end_idx}")
        if end_idx.shape[0] == 0:
            end_idx = len(seq[0])
        else:
            end_idx = end_idx[0][1]

        self.log(f"end_idx: {end_idx}")
        seq = seq[0][1:end_idx]

        # if first dimension is not multiple of 6, cut off the last few so it is
        if seq.shape[0] % 12 != 0:
            seq = seq[: -(seq.shape[0] % 12)]

        tokens = seq.reshape(-1, 2)

        tokens = tokens - 3
        return tokens

    def decode_mesh(self, tokens: Tensor):
        meshes = []
        with torch.cuda.device(self.device):
            # tokens: [num_features, 2]
            codebook = self.mesh_autoencoder.rq.codebooks
            # codebook: [2, vocab_size, feature_dim]
            # get first codebook (second one is the same)
            codebook = codebook[0]
            # codebook: [vocab_size, feature_dim]

            z_s = codebook[tokens]
            # z_s = [num_features, 2, feature_dim]

            print(z_s.shape)

            # for j, quantizer in enumerate(self.mesh_autoencoder.rq.codebooks):
            #     # Get the quantized vectors for the i-th quantizer
            #     indices = seq[:, j]
            #     z_res = quantizer.get_codebook_entry(indices)
            #     # Accumulate the quantized vectors
            #     z_s.append(z_res)

            z_q = torch.sum(z_s, dim=1)
            # z_q = [num_features, feature_dim]

            print(z_q.shape)

            # decode the mesh
            recons = self.mesh_autoencoder.decode(z_q)

            mesh = reconstruct_mesh(recons)
            meshes.append(mesh)

        return join_meshes_as_batch(meshes)

    def save_mesh(self, mesh: Meshes, path: str):
        for i in range(mesh.num_meshes()):
            save_obj(
                f"{path}/{i}.obj",
                mesh.verts_list()[i],
                mesh.faces_list()[i],
            )


if __name__ == "__main__":
    device = "cuda:1"
    tools = Tools(device)

    inp = torch.tensor(
        [[1, 13, 293, 57, 211, 48, 267, 13, 267, 156, 342, 34, 364]],
        device=device,
    )
    tools.decode_mesh(inp)

# # Add tqdm progress bar
# for i in tqdm(range(len(dataset)), desc="decoding meshes"):
#     all_min_encoding_indices = dataset[i]
#     print(all_min_encoding_indices)
#     all_min_encoding_indices = all_min_encoding_indices.to("cuda:3")

#     z_s = []
#     for j, quantizer in enumerate(model.rq.vq_layers):
#         # Get the quantized vectors for the i-th quantizer
#         indices = all_min_encoding_indices[:, j]
#         z_res = quantizer.get_codebook_entry(indices)
#         # Accumulate the quantized vectors
#         z_s.append(z_res)

#     z_s = torch.stack(z_s, dim=1)
#     z_q = torch.sum(z_s, dim=1)

#     # encode the mesh
#     recons = model.decode(z_q)

#     mesh = reconstruct_mesh(recons)
#     save_obj(
#         f"decoded_meshes/{i}.obj",
#         mesh.verts_list()[0],
#         mesh.faces_list()[0],
#     )
