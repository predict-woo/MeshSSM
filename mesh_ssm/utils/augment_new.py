import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
import numpy as np


def scale_augmentation(meshes: Meshes, scale_range=(0.75, 1), origin=(0.5, 0.5, 0.5)):
    """
    Apply uniform scaling augmentation to the mesh from a specified origin point.

    Args:
    - meshes (Meshes): Input meshes to be scaled.
    - scale_range (tuple): Range of scaling factors to apply.
    - origin (tuple): Origin point for scaling (default: (0.5, 0.5, 0.5)).

    Returns:
    - Meshes: Scaled meshes.
    """
    scaled_verts_list = []
    origin_tensor = torch.tensor(origin, dtype=torch.float32)

    for verts in meshes.verts_list():
        scale = torch.FloatTensor(1).uniform_(*scale_range).item()
        centered_verts = verts - origin_tensor
        scaled_centered_verts = centered_verts * scale
        scaled_verts = scaled_centered_verts + origin_tensor
        scaled_verts_list.append(scaled_verts)

    return Meshes(verts=scaled_verts_list, faces=meshes.faces_list())


def jitter_shift_augmentation(meshes: Meshes, jitter_range=(-0.1, 0.1)):
    """
    Apply jitter-shift augmentation to the mesh.

    Args:
    - mesh (Meshes): Input mesh to be jittered.
    - jitter_range (tuple): Range of jitter values to apply.

    Returns:
    - Meshes: Jittered mesh.
    """

    jittered_verts_list = []
    for verts in meshes.verts_list():
        jitter = torch.FloatTensor(3).uniform_(*jitter_range)
        jittered_verts = verts + jitter
        jittered_verts_list.append(jittered_verts)
    return Meshes(verts=jittered_verts_list, faces=meshes.faces_list())


# def normalize_mesh(meshes: Meshes):
#     """
#     Normalize the mesh to be centered at origin and scaled to unit length.

#     Args:
#     - mesh (Meshes): Input mesh to be normalized.

#     Returns:
#     - Meshes: Normalized mesh.
#     """
#     normalized_verts_list = []
#     for verts in meshes.verts_list():
#         min_coords = verts.min(dim=0)[0]
#         max_coords = verts.max(dim=0)[0]
#         max_extent = torch.max(max_coords - min_coords)
#         normalized_verts = (verts - min_coords.unsqueeze(0)) / max_extent
#         normalized_verts_list.append(normalized_verts)
#     return Meshes(verts=normalized_verts_list, faces=meshes.faces_list())


def normalize_mesh(meshes: Meshes):
    """
    Normalize the mesh to be at origin and scaled to unit length.

    Args:
    - mesh (Meshes): Input mesh to be normalized.

    Returns:
    - Meshes: Normalized mesh.
    """
    normalized_verts_list = []
    for verts in meshes.verts_list():
        min_coords = verts.min(dim=0)[0]
        max_coords = verts.max(dim=0)[0]
        max_extent = torch.max(max_coords - min_coords)
        normalized_verts = (verts - min_coords.unsqueeze(0)) / max_extent
        normalized_verts_list.append(normalized_verts)
    return Meshes(verts=normalized_verts_list, faces=meshes.faces_list())


def quantize_mesh(meshes: Meshes, num_bins=128):
    """
    Quantize the mesh vertices to fit into 128 bins between 0 and 1.

    Args:
    - mesh (Meshes): Input mesh to be quantized.
    - num_bins (int): Number of bins to fit the mesh into.

    Returns:
    - Meshes: Quantized mesh.
    """
    normalized_verts_list = []
    for verts in meshes.verts_list():
        # Fit into bins
        scaled_verts = verts * (num_bins - 1)
        indices = scaled_verts.long()
        scaled_indices = indices / (num_bins - 1)

        normalized_verts_list.append(scaled_indices)

    return Meshes(verts=normalized_verts_list, faces=meshes.faces_list())


def rotate_mesh(meshes: Meshes, rotation_range=(0, 2 * np.pi)):
    """
    Apply rotation augmentation to the mesh.
    """
    rotated_verts_list = []
    for verts in meshes.verts_list():
        rotation_angle = torch.FloatTensor(1).uniform_(*rotation_range)
        rotation_matrix = axis_angle_to_matrix(
            verts.new_tensor([0, 1, 0], dtype=torch.float32), rotation_angle
        )
        rotated_verts = torch.einsum("nxy,nxyz->nxyz", rotation_matrix, verts)
        rotated_verts_list.append(rotated_verts)
    return Meshes(verts=rotated_verts_list, faces=meshes.faces_list())


def sort_mesh_faces(meshes: Meshes) -> Meshes:
    sorted_faces_list = []
    for mesh in meshes:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        # Calculate face centroids
        face_centroids = verts[faces].mean(dim=1)

        # Sort faces based on z-y-x coordinates of centroids
        _, sorted_indices = torch.sort(
            face_centroids.flatten(), dim=0, descending=False
        )
        sorted_faces = faces[sorted_indices // 3]

        # Sort faces based on z-y-x coordinates of centroids using stable sort
        _, x_sorted = torch.sort(face_centroids[:, 0], stable=True)
        _, y_sorted = torch.sort(face_centroids[x_sorted, 1], stable=True)
        _, z_sorted = torch.sort(face_centroids[x_sorted][y_sorted, 2], stable=True)

        sorted_indices = x_sorted[y_sorted][z_sorted]
        sorted_faces = faces[sorted_indices]

        # Get vertices for all sorted faces
        face_verts = verts[sorted_faces]

        # Stable sort for vertices within each face
        _, x_min = torch.sort(face_verts[:, :, 0], dim=1, stable=True)
        _, y_min = torch.sort(
            torch.gather(face_verts[:, :, 1], 1, x_min), dim=1, stable=True
        )
        _, z_min = torch.sort(
            torch.gather(torch.gather(face_verts[:, :, 2], 1, x_min), 1, y_min),
            dim=1,
            stable=True,
        )

        min_indices = torch.gather(torch.gather(x_min, 1, y_min), 1, z_min)[:, 0]

        # Create a tensor to roll each face
        roll_indices = torch.arange(3).unsqueeze(0).repeat(len(sorted_faces), 1)
        roll_indices = (roll_indices - min_indices.unsqueeze(1)) % 3

        # Apply the roll to each face
        reordered_faces = torch.gather(sorted_faces, 1, roll_indices)

        sorted_faces_list.append(reordered_faces)

    # Create a new mesh with sorted and reordered faces
    sorted_meshes = Meshes(verts=meshes.verts_list(), faces=sorted_faces_list)

    return sorted_meshes


def augment_mesh(mesh: Meshes):
    mesh = normalize_mesh(mesh)
    mesh = scale_augmentation(mesh)
    mesh = jitter_shift_augmentation(mesh)


def augment_mesh(mesh: Meshes):
    mesh = scale_augmentation(mesh)
    mesh = jitter_shift_augmentation(mesh)
    mesh = normalize_mesh(mesh)
    mesh = sort_mesh_faces(mesh)
    return mesh
