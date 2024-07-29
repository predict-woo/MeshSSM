import trimesh
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj


def decimate_mesh(mesh, target_faces):
    """
    Decimate the mesh to a target number of faces using trimesh.

    Args:
    - mesh (Meshes): Input mesh to be decimated.
    - target_faces (int): Target number of faces for the decimated mesh.

    Returns:
    - Meshes: Decimated mesh.
    """
    trimesh_mesh = trimesh.Trimesh(
        vertices=mesh.verts_list()[0].cpu().numpy(),
        faces=mesh.faces_list()[0].cpu().numpy(),
    )
    decimated_mesh = trimesh_mesh.simplify_quadratic_decimation(target_faces)
    return Meshes(
        verts=[torch.tensor(decimated_mesh.vertices).float()],
        faces=[torch.tensor(decimated_mesh.faces).long()],
    )


def hausdorff_distance(mesh1, mesh2):
    """
    Calculate the Hausdorff distance between two meshes.

    Args:
    - mesh1 (Meshes): First mesh.
    - mesh2 (Meshes): Second mesh.

    Returns:
    - float: Hausdorff distance.
    """
    verts1 = mesh1.verts_list()[0].cpu().numpy()
    verts2 = mesh2.verts_list()[0].cpu().numpy()
    d1 = np.max(np.min(np.linalg.norm(verts1[:, None] - verts2, axis=2), axis=1))
    d2 = np.max(np.min(np.linalg.norm(verts2[:, None] - verts1, axis=2), axis=1))
    return max(d1, d2)


def select_meshes(mesh_list, hausdorff_threshold, max_faces):
    """
    Select meshes based on Hausdorff distance and face count.

    Args:
    - mesh_list (list of Meshes): List of input meshes.
    - hausdorff_threshold (float): Hausdorff distance threshold.
    - max_faces (int): Maximum number of faces for the decimated mesh.

    Returns:
    - list of Meshes: List of selected meshes.
    """
    selected_meshes = []
    for mesh in mesh_list:
        decimated_mesh = decimate_mesh(mesh, max_faces)
        hd = hausdorff_distance(mesh, decimated_mesh)
        if hd < hausdorff_threshold:
            selected_meshes.append(decimated_mesh)
    return selected_meshes


# Example usage:
if __name__ == "__main__":
    # Create a sample mesh list (replace this with actual loading code)
    verts, faces, aux = load_obj("Horse2.obj")
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    # Selection parameters
    hausdorff_threshold = 0.05
    max_faces = 800

    # Select meshes
    selected_meshes = select_meshes(mesh, hausdorff_threshold, max_faces)

    # save the selected mesh
    save_obj(
        "selected_mesh.obj",
        selected_meshes.verts_list()[0],
        selected_meshes.faces_list()[0],
    )

    # Print the number of selected meshes
    print(f"Number of selected meshes: {len(selected_meshes)}")
