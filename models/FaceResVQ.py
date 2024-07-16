import torch
import torch.nn as nn
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualVQ

class FaceResVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        num_quantizers: int,
        codebook_size: int,
        num_faces: int,
        num_vertices_per_face: int = 3,
        commitment_weight: float = 1.0,
        decay: float = 0.8,
        threshold_ema_dead_code: int = 2
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.num_faces = num_faces
        self.num_vertices_per_face = num_vertices_per_face

        # Ensure num_quantizers is divisible by num_vertices_per_face
        assert num_quantizers % num_vertices_per_face == 0, "num_quantizers must be divisible by num_vertices_per_face"
        
        self.quantizers_per_vertex = num_quantizers // num_vertices_per_face

        self.residual_vq = ResidualVQ(
            dim = dim,
            num_quantizers = self.quantizers_per_vertex,
            codebook_size = codebook_size,
            shared_codebook = True,
            commitment_weight = commitment_weight,
            decay = decay,
            threshold_ema_dead_code = threshold_ema_dead_code
        )

    def forward(self, face_features: torch.Tensor):
        """
        Args:
            face_features (torch.Tensor): Tensor of shape (num_faces, dim) containing face features
        Returns:
            quantized (torch.Tensor): Quantized face features
            indices (torch.Tensor): Quantization indices
            commit_loss (torch.Tensor): Commitment loss
        """
        batch_size, num_faces, dim = face_features.shape
        assert num_faces == self.num_faces, f"Expected {self.num_faces} faces, but got {num_faces}"

        # Split face features into vertex features
        vertex_features = face_features.view(batch_size, num_faces, self.num_vertices_per_face, -1)

        # Average features for shared vertices (assuming shared vertices have the same index)
        unique_vertex_features = torch.zeros(batch_size, num_faces * self.num_vertices_per_face, dim // self.num_vertices_per_face, device=face_features.device)
        for i in range(self.num_vertices_per_face):
            unique_vertex_features[:, i::self.num_vertices_per_face] = vertex_features[:, :, i, :]

        # Quantize the unique vertex features
        quantized, indices, commit_loss = self.residual_vq(unique_vertex_features)

        # Reshape quantized features back to face structure
        quantized = quantized.view(batch_size, num_faces, self.num_vertices_per_face, -1)
        quantized = quantized.view(batch_size, num_faces, -1)

        return quantized, indices, commit_loss