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
from mesh_ssm.utils.render import reconstruct_mesh, render_mesh
import wandb


class MeshAutoencoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MeshEncoder()
        self.decoder = MeshDecoder()
        self.rq = ResidualVectorQuantizer(n_e=192, e_dim=192, num_quantizers=2)
        self.feature_extractor = FaceFeatureExtractor()
        self.transforms = None

    def setup(self, stage=None):
        self.transforms = Transforms(device=self.device)

    def training_step(self, batch: Meshes, batch_idx: int):
        recons_loss, commit_loss = self.common_step(batch, batch_idx)
        self.log("train reconstruction loss", recons_loss)
        self.log("train commitment loss", commit_loss)
        return recons_loss + commit_loss

    def validation_step(self, batch: Meshes, batch_idx: int):
        recons_loss, commit_loss = self.common_step(batch, batch_idx, render=True)
        self.log("validation reconstruction loss", recons_loss, batch_size=1)
        self.log("validation commitment loss", commit_loss, batch_size=1)
        return recons_loss, commit_loss

    def common_step(self, batch: Meshes, batch_idx: int, render=False):
        features = self.feature_extractor.get_derived_face_features(batch)
        edges = self.feature_extractor.get_face_feature_graph(batch)
        face_features = self.encoder(features, edges)

        # split face features into vertex features
        vertex_features = self.transforms.split_features(face_features)

        # average face features
        faces = batch.faces_packed()
        vertex_features = vertex_features.unsqueeze(0)  # [F, D] -> [1, F, D]
        vertex_features = self.transforms.avg_fv_feat(faces, vertex_features)

        # put vertex features into faces
        vertex_features = vertex_features.squeeze(0)
        avg_f_feat = vertex_features[faces]  # [F, 3, D // 3]
        avg_f_feat = rearrange(avg_f_feat, "n c d -> (n c) d")  # [F * 3, D // 3]

        # residual vector quantization
        z_q, z_stack, commit_loss, all_min_encoding_indices = self.rq(avg_f_feat)

        # reshape
        rearranged_z_q = rearrange(z_q, "(n c) d -> n (c d)", c=3).unsqueeze(0)
        rearranged_z_q = rearrange(rearranged_z_q, "b n c -> b c n")

        # decode
        decoded = self.decoder(rearranged_z_q)

        # reconstruct mesh
        recons = rearrange(decoded, "b f (c n) -> b f c n", c=9)
        recons = recons.squeeze(0)

        # render
        if render:
            reconstructed_mesh = reconstruct_mesh(recons)
            image = render_mesh(reconstructed_mesh)

            # log image to wandb
            self.logger.experiment.log(
                {"val_input_image": [wandb.Image(image, caption="val_input_image")]}
            )

        ## original loss
        # verts = batch.verts_packed()
        # mesh_one_hot_encoding = self.transforms.smooth_one_hot(verts)
        # mesh_face_vertex_encoding = mesh_one_hot_encoding[faces]
        # mesh_face_vertex_encoding = rearrange(
        #     mesh_face_vertex_encoding, "f v x i -> f (v x) i"
        # )

        # log_probs = torch.log(recons + 1e-9)
        # recons_loss = -torch.sum(log_probs * mesh_face_vertex_encoding)

        ## l2 loss for each vertex
        # recons: [F,9,128]
        # choose maximum for each vertex coordinate: [F,9,128] -> [F,9]
        recons = recons.max(dim=-1)[0]
        # recons: [F,9]
        # divide by 128
        recons = recons / 128
        # recons: [F,9]

        # get face verts
        verts = batch.verts_packed()
        face_verts = verts[faces]
        # face_verts: [F, 3, 3]
        # flatten to [F,9]
        face_verts = rearrange(face_verts, "f v c -> f (v c)")
        # face_verts: [F, 9]

        # get l2 loss
        recons_loss = torch.nn.functional.mse_loss(recons, face_verts)

        return recons_loss, commit_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
