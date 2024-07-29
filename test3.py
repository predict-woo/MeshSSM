from mesh_ssm.utils.agument import (
    scale_augmentation,
    jitter_shift_augmentation,
    normalize_mesh,
    sort_mesh_faces,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.io import load_obj

# Example usage:
if __name__ == "__main__":

    # Create a sample mesh (replace this with actual loading code)
    for i in range(1):
        verts, faces, aux = load_obj("Horse2.obj")
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

        # Apply scale augmentation
        mesh_scaled = scale_augmentation(mesh)

        # Apply jitter shift augmentation
        mesh_jittered = jitter_shift_augmentation(mesh_scaled)

        # Normalize the mesh
        mesh_normalized = normalize_mesh(mesh_jittered)

        # Sort the mesh faces
        mesh_sorted = sort_mesh_faces(mesh_normalized)

        # Final mesh
        final_mesh = mesh_sorted

        # save the final mesh
        save_obj(
            f"final_mesh_{i}.obj",
            final_mesh.verts_list()[0],
            final_mesh.faces_list()[0],
        )

        # Print the final mesh vertices to verify
        final_mesh.verts_list()
