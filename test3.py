from mesh_ssm.utils.augment import normalize_mesh, augment_mesh
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.io import load_obj
from mesh_ssm.utils.render import render_mesh
from mesh_ssm.utils.render import _deconstruct_mesh, reconstruct_mesh

# Example usage:
if __name__ == "__main__":

    # Create a sample mesh (replace this with actual loading code)
    verts, faces, aux = load_obj("final_mesh_0.obj")
    meshes = Meshes(verts=[verts] * 10, faces=[faces.verts_idx] * 10)

    deconstructed_mesh = _deconstruct_mesh(meshes)
    print(deconstructed_mesh.shape)
    reconstructed_mesh = reconstruct_mesh(deconstructed_mesh)

    render_mesh(reconstructed_mesh)
# # Apply scale augmentation
# meshes = augment_mesh(meshes)

# # save each mesh
# for i, mesh in enumerate(meshes):
#     save_obj(f"Horse2_augmented_{i}.obj", mesh.verts_packed(), mesh.faces_packed())
