import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
import numpy as np
import os
import bpy
import bmesh
import igl

# set torch device to cpu
device = torch.device("cpu")


# class ShapeNetV2Dataset(Dataset):
#     def __init__(
#         self,
#         root_dir,
#         categories,
#         split="train",
#         max_faces=800,
#         hausdorff_threshold=0.01,
#     ):
#         self.root_dir = root_dir
#         self.categories = categories
#         self.split = split
#         self.max_faces = max_faces
#         self.hausdorff_threshold = hausdorff_threshold

#         self.shapes = self._load_shapes()

#     def _load_shapes(self):
#         shapes = []
#         for category in self.categories:
#             category_dir = os.path.join(self.root_dir, category)
#             shape_files = os.listdir(category_dir)

#             if self.split == "train":
#                 shape_files = shape_files[: int(0.9 * len(shape_files))]
#             elif self.split == "test":
#                 shape_files = shape_files[int(0.9 * len(shape_files)) :]

#             for shape_file in shape_files:
#                 shape_path = os.path.join(category_dir, shape_file)
#                 decimated_shape = self._decimate_shape(shape_path)

#                 if decimated_shape is not None:
#                     shapes.append(decimated_shape)

#         return shapes

#     def _decimate_shape(self, shape_path):
#         # Load original mesh
#         verts, faces, _ = load_obj(shape_path)
#         original_verts = verts.numpy()
#         original_faces = faces.verts_idx.numpy()

#         # Decimate mesh using Blender
#         bpy.ops.import_scene.obj(filepath=shape_path)
#         obj = bpy.context.selected_objects[0]
#         bpy.context.view_layer.objects.active = obj

#         best_mesh = None
#         best_hausdorff = float("inf")

#         for angle in range(1, 61):
#             bpy.ops.object.modifier_add(type="DECIMATE")
#             decimate = obj.modifiers["Decimate"]
#             decimate.decimate_type = "DISSOLVE"
#             decimate.angle_limit = angle * np.pi / 180

#             bpy.ops.object.modifier_apply(modifier="Decimate")

#             # Get decimated mesh data
#             bm = bmesh.new()
#             bm.from_mesh(obj.data)
#             bm.verts.ensure_lookup_table()
#             bm.faces.ensure_lookup_table()

#             decimated_verts = np.array([v.co for v in bm.verts])
#             decimated_faces = np.array([[v.index for v in f.verts] for f in bm.faces])

#             # Calculate Hausdorff distance using igl
#             hausdorff_dist, _ = igl.hausdorff(
#                 original_verts, original_faces, decimated_verts, decimated_faces
#             )

#             if (
#                 hausdorff_dist < self.hausdorff_threshold
#                 and hausdorff_dist < best_hausdorff
#                 and len(decimated_faces) <= self.max_faces
#             ):
#                 best_mesh = Meshes(
#                     verts=[torch.tensor(decimated_verts)],
#                     faces=[torch.tensor(decimated_faces)],
#                 )
#                 best_hausdorff = hausdorff_dist

#             bm.free()
#             bpy.ops.object.mode_set(mode="EDIT")
#             bpy.ops.mesh.select_all(action="SELECT")
#             bpy.ops.mesh.delete(type="VERT")
#             bpy.ops.object.mode_set(mode="OBJECT")

#         bpy.ops.object.delete()

#         if best_mesh is not None:
#             # Normalize mesh
#             center = best_mesh.verts_packed().mean(dim=0)
#             verts = best_mesh.verts_packed() - center
#             scale = verts.abs().max()
#             verts = verts / scale

#             return Meshes(verts=[verts], faces=best_mesh.faces_list())

#         return None

#     def __len__(self):
#         return len(self.shapes)

#     def __getitem__(self, idx):
#         return self.shapes[idx]

# def collate_fn(batch):
#     return Meshes.from_data_list(batch)


def decimate_shape(shape_path, hausdorff_threshold=0.01):
    # Load original mesh
    verts, faces, _ = load_obj(shape_path)
    original_verts = verts.numpy()
    original_faces = faces.verts_idx.numpy()

    # Decimate mesh using Blender
    bpy.ops.import_scene.obj(filepath=shape_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    best_mesh = None
    best_hausdorff = float("inf")

    for angle in range(1, 61):
        bpy.ops.object.modifier_add(type="DECIMATE")
        decimate = obj.modifiers["Decimate"]
        decimate.decimate_type = "DISSOLVE"
        decimate.angle_limit = angle * np.pi / 180

        bpy.ops.object.modifier_apply(modifier="Decimate")

        # Get decimated mesh data
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        decimated_verts = np.array([v.co for v in bm.verts])
        decimated_faces = np.array([[v.index for v in f.verts] for f in bm.faces])

        # Calculate Hausdorff distance using igl
        hausdorff_dist, _ = igl.hausdorff(
            original_verts, original_faces, decimated_verts, decimated_faces
        )

        if hausdorff_dist < hausdorff_threshold and hausdorff_dist < best_hausdorff:
            best_mesh = Meshes(
                verts=[torch.tensor(decimated_verts)],
                faces=[torch.tensor(decimated_faces)],
            )
            best_hausdorff = hausdorff_dist

        bm.free()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.delete()

    if best_mesh is not None:
        # Normalize mesh
        center = best_mesh.verts_packed().mean(dim=0)
        verts = best_mesh.verts_packed() - center
        scale = verts.abs().max()
        verts = verts / scale

        return Meshes(verts=[verts], faces=best_mesh.faces_list())

    return None


# Usage example
root_dir = "/path/to/ShapeNetV2"
categories = ["Chair", "Table", "Bench", "Lamp"]
train_dataset = ShapeNetV2Dataset(root_dir, categories, split="train")

res = train_dataset._decimate_shape("a.obj")

save_obj("b.obj", faces=res.faces_packed(), verts=res.verts_packed())

# train_loader = DataLoader(
#     train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True
# )

# test_dataset = ShapeNetV2Dataset(root_dir, categories, split="test")
# test_loader = DataLoader(
#     test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False
# )
