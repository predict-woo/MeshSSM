from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, packed_to_list, join_meshes_as_batch
from pytorch3d.io import save_obj
from matplotlib import pyplot as plt
import torch

import einops
from .mesh import Transforms

import wandb
from PIL import Image
import numpy as np

from torch import Tensor

transforms = Transforms(device="cpu")


def _deconstruct_mesh(mesh: Meshes):
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    mesh_one_hot_encoding = transforms.smooth_one_hot(verts)
    mesh_face_vertex_encoding = mesh_one_hot_encoding[faces]
    mesh_face_vertex_encoding = einops.rearrange(
        mesh_face_vertex_encoding, "f v x i -> f (v x) i"
    )
    return mesh_face_vertex_encoding


def reconstruct_mesh(recons, num_faces_per_mesh: Tensor = None):
    if num_faces_per_mesh is None:
        num_faces_per_mesh = torch.tensor([recons.shape[0]], device=recons.device)

    recons = einops.rearrange(recons, "f (t v) n -> f t v n", t=3)

    discretization = torch.linspace(0, 1, recons.shape[-1], device=recons.device)
    peak_indices = torch.argmax(recons, dim=-1)
    recons_disc = discretization[peak_indices]

    list_recons = packed_to_list(recons_disc, num_faces_per_mesh.tolist())

    verts_list = []
    faces_list = []

    for face_verts in list_recons:

        flat_vertices = face_verts.view(-1, 3)  # Flatten vertices

        unique_vertices, inverse_indices = torch.unique(
            flat_vertices, dim=0, return_inverse=True
        )

        # Create face indices from inverse_indices
        face_indices = inverse_indices.view(-1, 3)  # Reshape to (face_num, 3)

        verts_list.append(unique_vertices)
        faces_list.append(face_indices)

    return Meshes(verts=verts_list, faces=faces_list)


def render_mesh(mesh: Meshes):
    # Initialize the camera
    R, T = look_at_view_transform(2.5, 30, 45, device=mesh.device)

    cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh.device)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object
    lights = PointLights(location=[[0.0, 0.0, 3.0]], device=mesh.device)

    # Load the mesh
    textures = TexturesVertex(
        verts_features=torch.tensor([1.0, 1.0, 0.0], device=mesh.device).expand_as(
            mesh.verts_list()[0]
        )[None]
    )

    # Create a phong renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(cameras=cameras, lights=lights, device=mesh.device),
    )

    mesh = Meshes(
        verts=mesh.verts_list()[0].unsqueeze(0),
        faces=mesh.faces_list()[0].unsqueeze(0),
        textures=textures,
    )

    # Render the mesh
    data = renderer(mesh)

    # Assuming your tensor is 'data' and in the shape [1, 512, 512, 4]
    data = data.squeeze(0)  # Remove the batch dimension, shape becomes [512, 512, 4]
    data = data.cpu().numpy()  # Move tensor to CPU and convert to numpy array

    # Convert to byte format
    data = (data * 255).astype(np.uint8)

    # Remove the alpha channel if not needed
    data = data[:, :, :3]

    # Convert numpy array to PIL Image
    image = Image.fromarray(data)
    image.save("test.png")

    return image, mesh
