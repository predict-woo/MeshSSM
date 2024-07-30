from mesh_ssm.utils.render import render_mesh
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from mesh_ssm.utils.augment import augment_mesh


verts, faces, aux = load_obj("cone.txt", load_textures=False)

meshes = Meshes(verts=[verts], faces=[faces.verts_idx])

image = render_mesh(meshes)
