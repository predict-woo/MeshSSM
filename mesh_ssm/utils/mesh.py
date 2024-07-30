import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.structures.utils import packed_to_list
from .positional_encoder import PositionalEncoder
from einops import rearrange
import igl
import numpy as np
from torch_geometric.data import Data, Batch
import torch.nn as nn


def derive_angle(x, y, eps=1e-5):
    x_norm = F.normalize(x, dim=-1, p=2)
    y_norm = F.normalize(y, dim=-1, p=2)
    z = torch.sum(x_norm * y_norm, dim=-1)
    return z.clamp(-1 + eps, 1 - eps).arccos()


class FaceFeatureExtractor:
    def __init__(self):
        self.positional_encoder = PositionalEncoder(
            in_dim=3, embed_level=10, include_input=True
        )

    def get_derived_face_features(self, meshes: Meshes) -> torch.Tensor:

        # pack into single tensor
        verts = meshes.verts_packed()  # [num_vertices, 3]
        faces = meshes.faces_packed()  # [num_faces, 3]
        face_verts = verts[faces]  # [num_faces, 3, 3]

        # angles
        rotated_face_verts_1 = torch.roll(face_verts, 1, dims=1)
        rotated_face_verts_2 = torch.roll(face_verts, 2, dims=1)
        diff_1 = rotated_face_verts_1 - face_verts
        diff_2 = rotated_face_verts_2 - face_verts
        angles = derive_angle(diff_1, diff_2)

        # normals
        cross = torch.cross(diff_1[:, 0, :], diff_2[:, 0, :], dim=-1)
        normals = F.normalize(cross, dim=-1, p=2)

        # area
        area = 0.5 * torch.unsqueeze(torch.norm(cross, dim=-1), -1)

        # concat each face features into 9 vertice coordinates, 1 area, 3 angles, 3 normals -> 16 features
        # [B, 3, 3] -> [B*3, 3]
        face_verts = rearrange(face_verts, "b n v -> (b n) v")
        face_verts = self.positional_encoder.encode(face_verts)
        face_verts = rearrange(face_verts, "(b n) v -> b (n v)", n=3)

        # concat face features
        face_features = torch.cat([face_verts, area, angles, normals], dim=-1)

        return face_features

    def get_face_feature_graph(self, meshes: Meshes):
        faces = meshes.faces_packed()
        face_np = faces.cpu().numpy()

        # Triangle-Triangle Adjacency Matrix
        TT, _ = igl.triangle_triangle_adjacency(face_np)

        # back to tensor
        TT = torch.tensor(TT)

        # create index matrix of same size
        ind = torch.arange(TT.shape[0]).unsqueeze(-1).repeat(1, TT.shape[1])

        # append ind to each element in TT
        TT_combined = torch.stack((ind, TT), dim=-1)

        # flatten and reshape
        edges = TT_combined.reshape(-1, 2)
        edges = edges[edges[:, 1] != -1]
        edges = rearrange(edges, "n b -> b n")

        # move to GPU
        edges = edges.to(meshes.verts_packed().device)
        return edges

    def get_data_batch(self, meshes: Meshes):
        edges = self.get_face_feature_graph(meshes)
        features = self.get_derived_face_features(meshes)
        # PyTorch Geometric Data object
        datas = []
        for i in range(len(edges)):
            data = Data(x=features[i], edge_index=edges[i])
            datas.append(data)
        batch = Batch.from_data_list(datas)
        return batch


class Transforms:
    def __init__(self, device):
        self.kernel_size = 9
        self.device = device
        self.kernel = self.gaussian_kernel(
            size=self.kernel_size, sigma=self.kernel_size / 6.0
        )
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            bias=False,
            padding=self.kernel_size // 2,
        )
        self.conv1d.weight.data = self.kernel
        self.conv1d.weight.requires_grad = False

    def avg_fv_feat(self, faces, face_features, num_vertices=None):
        r"""Given features assigned for every vertex of every face, computes per-vertex features by
        averaging values across all faces incident each vertex.

        Args:
        faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
                shape :math:`(\text{num_faces}, \text{face_size})`.
        face_features (torch.FloatTensor): any features assigned for every vertex of every face, of shape
                :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, N)`.
        num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

        Return:
            (torch.FloatTensor): of shape (B, V, 3)
        """
        if num_vertices is None:
            num_vertices = int(faces.max()) + 1

        B = face_features.shape[0]
        V = num_vertices
        F = faces.shape[0]
        FSz = faces.shape[1]
        Nfeat = face_features.shape[-1]
        vertex_features = torch.zeros(
            (B, V, Nfeat), dtype=face_features.dtype, device=face_features.device
        )
        counts = torch.zeros(
            (B, V), dtype=face_features.dtype, device=face_features.device
        )

        faces = faces.unsqueeze(0).repeat(B, 1, 1)
        fake_counts = torch.ones(
            (B, F), dtype=face_features.dtype, device=face_features.device
        )
        #              B x F          B x F x 3
        # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
        for i in range(FSz):
            vertex_features.scatter_add_(
                1, faces[..., i : i + 1].repeat(1, 1, Nfeat), face_features[..., i, :]
            )
            counts.scatter_add_(1, faces[..., i], fake_counts)

        counts = counts.clip(min=1).unsqueeze(-1)
        vertex_normals = vertex_features / counts
        return vertex_normals

    def split_features(self, features, num_vertices=3):
        # [B, F, D] -> [B, F, n_vertices, D // n_vertices]
        features = rearrange(features, "f (n d) -> f n d", n=num_vertices)
        return features

    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        # Generate a range of values
        kernel = torch.arange(0, size, device=self.device).float()

        # Calculate the center index
        center = (size - 1) / 2

        # Apply the Gaussian formula
        kernel = torch.exp(-((kernel - center) ** 2) / (2 * sigma**2))

        # Normalize the kernel
        kernel = kernel / torch.sum(kernel)

        # Reshape to (out_channels, in_channels, kernel_size)
        kernel = kernel.view(1, 1, size)

        return kernel

    def smooth_one_hot(self, verts, num_bins=128):
        # Scale the coordinates from [0, 1] to [0, num_bins-1]
        scaled_verts = verts * (num_bins - 1)
        # Convert the scaled coordinates to integer indices
        indices = scaled_verts.long()

        # Initialize the one-hot tensor
        one_hot_verts = torch.zeros((verts.shape[0], 3, num_bins), device=self.device)
        # Scatter the indices to create one-hot encoding
        one_hot_verts.scatter_(2, indices.unsqueeze(-1), 1.0)

        one_hot_verts_reshaped = one_hot_verts.view(-1, 1, 128)
        one_hot_verts_smoothed = self.conv1d(one_hot_verts_reshaped)
        one_hot_verts_smoothed = one_hot_verts_smoothed.view(-1, 3, 128)

        return one_hot_verts_smoothed


if __name__ == "__main__":
    verts, faces, aux = load_obj("Horse.obj")
    feature_extractor = FaceFeatureExtractor()
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    features = feature_extractor.get_derived_face_features(meshes)
    edges = feature_extractor.get_face_feature_graph(meshes)
    print(edges[0].shape)
    print(features[0].shape)
    data = feature_extractor.get_data_batch(meshes)
    print(data)

    # # example triangle mesh with 4 vertices and 4 faces
    # verts = torch.tensor(
    #     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float
    # )
    # faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])

    # # another example triangle mesh with 4 vertices and 4 faces, but located at 1,1,1
    # verts2 = torch.tensor(
    #     [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=torch.float
    # )
    # faces2 = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])

    # # meshes = Meshes(verts=[verts, verts2], faces=[faces, faces2])
    # meshes = Meshes(verts=[verts], faces=[faces])

    # feature_extractor = FaceFeatureExtractor()
    # edges = feature_extractor.get_face_feature_graph(meshes)
    # features = feature_extractor.get_derived_face_features(meshes)
    # print(edges[0].shape)
    # print(features[0].shape)
