from .mesh_decoder import MeshDecoder
from .mesh_encoder import MeshEncoder
from .residual_vector_quantize import ResidualVectorQuantizer
from einops import rearrange
from mesh_ssm.utils.mesh import FaceFeatureExtractor, Transforms
from mesh_ssm.utils.render import reconstruct_mesh
from pytorch3d.io import save_obj
from pytorch3d.structures import packed_to_list
from torch import Tensor
import lightning as L
import torch
from typing import Tuple
import os


class MeshAutoencoder(L.LightningModule):
    def __init__(
        self,
        lr=0.001,
        ks=9,
        path="results",
        name="mesh_autoencoder",
    ):
        super().__init__()
        self.encoder = MeshEncoder()
        self.decoder = MeshDecoder()
        self.rq = ResidualVectorQuantizer(n_e=192, e_dim=192, num_quantizers=2)
        self.feature_extractor = FaceFeatureExtractor()
        self.transforms = None
        self.lr = lr
        self.ks = ks

        self.path = f"{path}/{name}_ks[{ks}]_lr[{lr}]"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.validation_step_output = None
        # self.save_hyperparameters()

    def setup(self, stage=None):
        self.transforms = Transforms(device=self.device, kernel_size=self.ks)

    def encode(self, x: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]):
        return self.common_step(*x, 0, encode=True)

    def decode(self, x: Tensor):
        return self.decode_step(x)

    def decode_step(self, z_q: Tensor):
        # reshape
        rearr_z_q = rearrange(z_q, "(F N) D -> F (N D)", N=3).unsqueeze(0)
        rearr_z_q = rearrange(rearr_z_q, "B F DN -> B DN F")

        # decode
        decoded = self.decoder(rearr_z_q)
        decoded = decoded.squeeze(0)

        # reconstruct mesh
        recons = rearrange(decoded, "F (CN O) -> F CN O", CN=9)
        return recons

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ):
        recons_loss, commit_loss, _ = self.common_step(*batch, batch_idx)
        self.log("train reconstruction loss", recons_loss)
        self.log("train commitment loss", commit_loss)
        return recons_loss + commit_loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Perform a validation step.

        Args:
            batch: A tuple containing:
                - faces: Tensor of face indices
                - verts: Tensor of vertex coordinates
                - features: Tensor of face features
                - edges: Tensor of edge indices
                - num_verts_per_mesh: Tensor of vertex counts per mesh
                - num_faces_per_mesh: Tensor of face counts per mesh
            batch_idx: Index of the current batch

        Returns:
            Tensor: Combined reconstruction and commitment loss
        """

        recons_loss, commit_loss, recons = self.common_step(*batch, batch_idx)
        self.log("validation reconstruction loss", recons_loss, batch_size=1)
        self.log("validation commitment loss", commit_loss, batch_size=1)
        self.validation_step_output = {
            "loss": recons_loss + commit_loss,
            "recons": recons,
            "num_faces_per_mesh": batch[5],
        }
        return recons_loss + commit_loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and self.validation_step_output is not None:
            pred = self.validation_step_output
            num_faces_per_mesh = pred["num_faces_per_mesh"]
            mesh = reconstruct_mesh(pred["recons"], num_faces_per_mesh)
            print(f"{self.path}/mesh_{self.global_step}.obj")
            save_obj(
                f"{self.path}/mesh_{self.global_step}.obj",
                mesh.verts_list()[0],
                mesh.faces_list()[0],
            )

    def common_step(
        self,
        faces: Tensor,
        verts: Tensor,
        features: Tensor,
        edges: Tensor,
        num_verts_per_mesh: Tensor,
        num_faces_per_mesh: Tensor,
        batch_idx: int,
        encode=False,
    ):
        r"""
        # Einops notation:

        ## Notation
        face num -------------------------- F = ?
        number of vertices per face ------- N = 3
        number of coordinates per vertex -- C = 3
        vertex num ------------------------ V = ?
        edge num -------------------------- E = ?
        vertex feature dimension ---------- D = 192
        output dimension ------------------ O = 128

        ## Combinations
        face feature dimension ------------ DN = 576
        face coordinate number ------------ CN = 9
        """

        f_feat = self.encoder(features, edges)

        # split face features into vertex features
        v_feat = rearrange(f_feat, "F (N D) -> F N D", N=3)

        # average vertex features
        avg_v_feat_list = self.transforms.avg_fv_feat(faces, v_feat)

        # put vertex features into faces
        avg_v_feat = avg_v_feat_list[faces]

        # flatten for residual vector quantization
        avg_v_feat = rearrange(avg_v_feat, "F N D -> (F N) D")

        # residual vector quantization
        z_q, _, commit_loss, all_min_encoding_indices = self.rq(avg_v_feat)

        if encode:
            return all_min_encoding_indices

        # reshape
        rearr_z_q = rearrange(z_q, "(F N) D -> F (N D)", N=3).unsqueeze(0)
        rearr_z_q = rearrange(rearr_z_q, "B F DN -> B DN F")

        # decode
        decoded = self.decoder(rearr_z_q)
        decoded = decoded.squeeze(0)

        # reconstruct mesh
        recons = rearrange(decoded, "F (CN O) -> F CN O", CN=9)
        # print(recons.shape)

        # calculate loss
        mesh_one_hot_encoding = self.transforms.smooth_one_hot(verts)
        mesh_face_vertex_encoding = mesh_one_hot_encoding[faces]
        truth = rearrange(mesh_face_vertex_encoding, "F N C O -> F (N C) O")

        # KL divergence
        log_pred = torch.log(recons + 1e-9)
        log_truth = torch.log(truth + 1e-9)
        recons_loss = torch.mean((log_truth - log_pred) * truth)

        return recons_loss, commit_loss, recons

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
