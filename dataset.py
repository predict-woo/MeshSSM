# Standard library imports
from os import path
from typing import Dict

# Third-party imports
import bpy
import bmesh
import igl
import numpy as np
import torch
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from tqdm import tqdm


class ShapeNetCoreWithPath(ShapeNetCore):
    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index and include its file path.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
            - model_path (str): full path to the model file.
        """
        model = self._get_item_ids(idx)
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        model["label"] = self.synset_dict[model["synset_id"]]
        model["model_path"] = model_path
        return model


shapenet_dataset = ShapeNetCoreWithPath("/home/andyye/ShapeNetCore", version=2)


def normalize_mesh(verts):
    # Compute the bounding box
    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)

    # Compute the range of the bounding box
    bbox_range = max_coords - min_coords

    # Normalize the vertices to [0, 1]
    normalized_verts = (verts - min_coords) / bbox_range

    return normalized_verts


def import_mesh(shape_path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=shape_path)

    obj = bpy.context.selected_objects[0]

    # triangulate
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Normalize the mesh
    verts = np.array([v.co for v in obj.data.vertices])
    normalized_verts = normalize_mesh(verts)

    # Update the object with normalized vertices
    for i, v in enumerate(obj.data.vertices):
        v.co = normalized_verts[i]

    return obj


def decimate_and_triangulate(obj, angle):
    # Create a copy of the object to work on
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    bpy.context.collection.objects.link(new_obj)

    # Select only the new object
    bpy.ops.object.select_all(action="DESELECT")
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj

    # Create and apply decimate modifier
    bpy.ops.object.modifier_add(type="DECIMATE")
    decimate = new_obj.modifiers["Decimate"]
    decimate.decimate_type = "DISSOLVE"
    decimate.angle_limit = angle * np.pi / 180
    bpy.ops.object.modifier_apply(modifier="Decimate")

    # Triangulate the mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(new_obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.update_edit_mesh(new_obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Get mesh data
    verts = np.array([v.co for v in new_obj.data.vertices])
    faces = np.array([f.vertices for f in new_obj.data.polygons])

    # Remove the temporary object
    bpy.data.objects.remove(new_obj, do_unlink=True)

    return verts, faces


def find_optimal_decimation(shape_path, hausdorff_threshold, min_angle=1, max_angle=60):
    # Import the original mesh
    original_obj = import_mesh(shape_path)
    original_verts = np.array([v.co for v in original_obj.data.vertices])
    original_faces = np.array([f.vertices for f in original_obj.data.polygons])
    original_face_count = len(original_faces)

    best_angle = min_angle
    best_verts = None
    best_faces = None

    for angle in range(min_angle, max_angle + 1, 2):
        decimated_verts, decimated_faces = decimate_and_triangulate(original_obj, angle)

        # Check if decimation at angle 10 has any effect
        if angle > 20 and len(decimated_faces) == original_face_count:
            break

        hausdorff_dist = igl.hausdorff(
            original_verts, original_faces, decimated_verts, decimated_faces
        )

        if hausdorff_dist > hausdorff_threshold:
            break

        best_angle = angle
        best_verts = decimated_verts
        best_faces = decimated_faces

    # Remove the original object from the scene
    bpy.data.objects.remove(original_obj, do_unlink=True)

    return best_angle, best_verts, best_faces


def save_mesh(verts, faces, filepath):
    igl.write_obj(filepath, verts, faces)


hausdorff_threshold = 0.01  # Adjust this value as needed

count = 0

with open("data_paths.txt", "w") as f:
    progress = tqdm(range(len(shapenet_dataset)), desc="Processing ShapeNet dataset")
    for i in progress:
        shape_path = shapenet_dataset[i]["model_path"]
        output_path = f"dataset/{count}_{shapenet_dataset[i]['synset_id']}.obj"
        best_angle, best_verts, best_faces = find_optimal_decimation(
            shape_path, hausdorff_threshold
        )
        # stdout.seek(0)
        # progress.set_description(stdout.read())
        if best_verts is None or best_faces is None:
            continue

        if best_faces.shape[0] > 800:
            continue

        save_mesh(best_verts, best_faces, output_path)
        count += 1


print(f"Total processed files: {count}")
