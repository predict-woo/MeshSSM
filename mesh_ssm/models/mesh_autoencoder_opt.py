from .mesh_decoder import MeshDecoder
from .mesh_encoder import MeshEncoder
from .residual_vector_quantize import ResidualVectorQuantizer
from einops import rearrange
from mesh_ssm.utils.mesh import FaceFeatureExtractor, Transforms
from mesh_ssm.utils.render import reconstruct_mesh
from pytorch3d.io import save_obj
from torch import Tensor
import lightning as L
import torch
from typing import Tuple


class MeshAutoencoderOpt(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MeshEncoder()
        self.decoder = MeshDecoder()
        self.rq = ResidualVectorQuantizer(n_e=192, e_dim=192, num_quantizers=2)
        self.feature_extractor = FaceFeatureExtractor()
        self.transforms = None

    def setup(self, stage=None):
        self.transforms = Transforms(device=self.device)

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ):
        recons_loss, commit_loss = self.common_step(*batch, batch_idx)
        self.log("train reconstruction loss", recons_loss)
        self.log("train commitment loss", commit_loss)
        return recons_loss + commit_loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ):
        recons_loss, commit_loss = self.common_step(*batch, batch_idx, render=True)
        self.log("validation reconstruction loss", recons_loss, batch_size=1)
        self.log("validation commitment loss", commit_loss, batch_size=1)
        return recons_loss + commit_loss

    def common_step(
        self,
        faces: Tensor,
        verts: Tensor,
        features: Tensor,
        edges: Tensor,
        batch_idx: int,
        render=False,
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
        avg_v_feat_list = self.transforms.avg_fv_feat(faces, v_feat, verts.shape[0])

        # put vertex features into faces
        avg_v_feat = avg_v_feat_list[faces]

        # flatten for residual vector quantization
        avg_v_feat = rearrange(avg_v_feat, "F N D -> (F N) D")

        # residual vector quantization
        z_q, _, commit_loss, _ = self.rq(avg_v_feat)

        # reshape
        rearr_z_q = rearrange(z_q, "(F N) D -> F (N D)", N=3).unsqueeze(0)
        rearr_z_q = rearrange(rearr_z_q, "B F DN -> B DN F")

        # decode
        decoded = self.decoder(rearr_z_q)
        decoded = decoded.squeeze(0)

        # reconstruct mesh
        recons = rearrange(decoded, "F (CN O) -> F CN O", CN=9)
        # print(recons.shape)

        # render
        if render:
            mesh = reconstruct_mesh(recons)
            save_obj(
                f"results/tea/mesh_{self.global_step}.obj",
                mesh.verts_packed(),
                mesh.faces_packed(),
            )

        # calculate loss
        mesh_one_hot_encoding = self.transforms.smooth_one_hot(verts)
        mesh_face_vertex_encoding = mesh_one_hot_encoding[faces]
        mesh_face_vertex_encoding = rearrange(
            mesh_face_vertex_encoding, "F N C O -> F (N C) O"
        )
        # print(mesh_face_vertex_encoding.shape)
        log_probs = torch.log(recons + 1e-9)
        recons_loss = -torch.mean(log_probs * mesh_face_vertex_encoding)

        return recons_loss, commit_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
