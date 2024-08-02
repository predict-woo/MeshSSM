from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from .mesh_encoder import MeshEncoder
from .mesh_decoder import MeshDecoder
from .residual_vector_quantize import ResidualVectorQuantizer
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from mesh_ssm.utils.mesh import FaceFeatureExtractor, Transforms
from einops import rearrange
import lightning as L
import torch
from mesh_ssm.utils.render import reconstruct_mesh, render_mesh
import wandb
import gc


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


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

    def test_step(self, batch: Meshes, batch_idx: int):
        recons_loss, commit_loss = self.common_step(batch, batch_idx, render=True)
        return recons_loss + commit_loss

    def training_step(self, batch: Meshes, batch_idx: int):
        recons_loss, commit_loss = self.common_step(batch, batch_idx)
        self.log("train reconstruction loss", recons_loss)
        self.log("train commitment loss", commit_loss)
        return recons_loss + commit_loss

    def validation_step(self, batch: Meshes, batch_idx: int):
        recons_loss, commit_loss = self.common_step(batch, batch_idx, render=True)
        self.log("validation reconstruction loss", recons_loss, batch_size=1)
        self.log("validation commitment loss", commit_loss, batch_size=1)
        return recons_loss + commit_loss

    def common_step(self, batch: Meshes, batch_idx: int, render=False):
        features = self.feature_extractor.get_derived_face_features(batch)
        edges = self.feature_extractor.get_face_feature_graph(batch)
        face_features = self.encoder(features, edges)

        # split face features into vertex features
        vertex_features = rearrange(face_features, "f (n d) -> f n d", n=3)

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
            image, mesh = render_mesh(reconstructed_mesh)

            # # log image to wandb
            # if hasattr(self.logger.experiment, "log"):
            #     self.logger.experiment.log(
            #         {"val_input_image": [wandb.Image(image, caption="val_input_image")]}
            #     )
            # save_obj(
            #     f"results/tea/mesh_{self.global_step}.obj",
            #     mesh.verts_packed(),
            #     mesh.faces_packed(),
            # )

        # original loss
        verts = batch.verts_packed()
        mesh_one_hot_encoding = self.transforms.smooth_one_hot(verts)
        mesh_face_vertex_encoding = mesh_one_hot_encoding[faces]
        mesh_face_vertex_encoding = rearrange(
            mesh_face_vertex_encoding, "f v x i -> f (v x) i"
        )

        log_probs = torch.log(recons + 1e-9)
        recons_loss = -torch.mean(log_probs * mesh_face_vertex_encoding)

        # test_log_probs = torch.log(mesh_face_vertex_encoding + 1e-9)
        # test_recons_loss = -torch.mean(test_log_probs * mesh_face_vertex_encoding)

        # print(recons[0][0])
        # print(mesh_face_vertex_encoding[0][0])
        # # print(torch.sum(recons[0][0]))
        # # print(torch.sum(mesh_face_vertex_encoding[0][0]))
        # # print(rearrange(verts[faces], "f v x -> f (v x)")[0][0])
        # print(recons_loss)
        # print(test_recons_loss)

        return recons_loss, commit_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
