from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
from pytorch3d.io import load_obj
from mesh_ssm.utils.mesh import FaceFeatureExtractor
from pytorch3d.structures import Meshes


if __name__ == "__main__":
    verts, faces, aux = load_obj("Horse.obj", load_textures=False)
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    autoencoder = MeshAutoencoder()
    autoencoder.training_step(meshes, 0)
