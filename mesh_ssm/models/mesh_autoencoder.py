from lightning.pytorch.utilities.types import OptimizerLRScheduler
from .mesh_encoder import MeshEncoder
from .mesh_decoder import MeshDecoder
from .residual_vector_quantize import ResidualVectorQuantizer
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from mesh_ssm.utils.mesh import FaceFeatureExtractor, Transforms
from einops import rearrange
import lightning as L
import torch
import torch.nn.functional as F


class MeshAutoencoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MeshEncoder()
        self.decoder = MeshDecoder()
        self.rq = ResidualVectorQuantizer(n_e=192, e_dim=192, num_quantizers=2)
        self.feature_extractor = FaceFeatureExtractor()
        self.transforms = Transforms()

    def training_step(self, batch: Meshes, batch_idx: int):
        # encode
        features = self.feature_extractor.get_derived_face_features(batch)
        edges = self.feature_extractor.get_face_feature_graph(batch)
        face_features = self.encoder(features, edges)

        # split face features into vertex features
        vertex_features = self.transforms.split_features(face_features)
        print(vertex_features.shape)

        # average face features
        faces = batch.faces_packed()
        vertex_features = vertex_features.unsqueeze(0)  # [F, D] -> [1, F, D]
        vertex_features = self.transforms.avg_fv_feat(faces, vertex_features)

        # put vertex features into faces
        vertex_features = vertex_features.squeeze(0)
        avg_f_feat = vertex_features[faces]  # [F, 3, D // 3]
        avg_f_feat = rearrange(avg_f_feat, "n c d -> (n c) d")  # [F * 3, D // 3]

        # residual vector quantization
        z_q, z_stack, mean_losses, all_min_encoding_indices = self.rq(avg_f_feat)

        # reshape
        rearranged_z_q = rearrange(z_q, "(n c) d -> n (c d)", c=3).unsqueeze(0)
        rearranged_z_q = rearrange(rearranged_z_q, "b n c -> b c n")

        # decode
        decoded = self.decoder(rearranged_z_q)

        # reconstruct mesh
        recons = rearrange(decoded, "b f (c n) -> b f c n", c=9)

        # loss
        verts = batch.verts_packed()
        print(verts.shape)
        mesh_one_hot_encoding = self.transforms.smooth_one_hot(verts)
        # print(mesh_one_hot_encoding.shape)

        return decoded

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
