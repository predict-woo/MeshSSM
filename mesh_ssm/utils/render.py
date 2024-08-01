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
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from matplotlib import pyplot as plt
import torch

import einops
from .mesh import Transforms

import wandb
from PIL import Image
import numpy as np


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


def reconstruct_mesh(recons):
    tri_recons = einops.rearrange(recons, "f (t v) n -> f t v n", t=3)
    face_num, tri, dim, disc = tri_recons.shape

    discretization = (
        torch.linspace(0, 1, disc, device=tri_recons.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    peak_indices = torch.argmax(tri_recons, dim=-1)  # Shape [face_num, tri, dim]
    vertex_coordinates = torch.gather(
        discretization.expand(face_num, tri, dim, disc),
        3,
        peak_indices.unsqueeze(-1),
    ).squeeze(
        -1
    )  # Shape [face_num, tri, dim]
    flat_vertices = vertex_coordinates.view(-1, 3)  # Flatten vertices
    unique_vertices, inverse_indices = torch.unique(
        flat_vertices, dim=0, return_inverse=True
    )

    # Create face indices from inverse_indices
    face_indices = inverse_indices.view(-1, 3)  # Reshape to (face_num, 3)

    # Create mesh using PyTorch3D Meshes structure
    mesh = Meshes(verts=unique_vertices.unsqueeze(0), faces=face_indices.unsqueeze(0))
    return mesh


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
