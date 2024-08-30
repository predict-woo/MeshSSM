import os
from PIL import Image
import igl
from project.render import render_mesh
from tools import Tools
import numpy as np
import torch

# Define the directory containing the OBJ files


tools = Tools(device="cuda:3")

tools.init_autoencoder(
    "checkpoints/chair-final-bs[128]-lr[0.001]-ks[3]/epoch=987-val_loss=0.0000.ckpt"
)


def downsample_image(image, factor):
    """Downsample an image by the given factor."""
    width, height = image.size
    return image.resize((width // factor, height // factor), Image.Resampling.LANCZOS)


def create_grid_image(grid_size=10, downsample_factor=2):
    images = []

    # Select the first 100 files
    dataset = np.load("encoded_chair_dataset.npy", allow_pickle=True)

    dataset = dataset[: grid_size**2]

    for enc in dataset:
        enc = torch.tensor(enc, device="cuda:3")
        mesh = tools.decode_mesh(enc)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        img = render_mesh(verts, faces)
        img_downsampled = downsample_image(img, downsample_factor)
        images.append(img_downsampled)

    # Determine the size of the grid image
    img_width, img_height = images[0].size
    grid_img_width = img_width * grid_size
    grid_img_height = img_height * grid_size

    # Create a new blank image with the calculated dimensions
    grid_image = Image.new("RGB", (grid_img_width, grid_img_height))

    # Paste each image into the grid
    for i, img in enumerate(images):
        x = (i % grid_size) * img_width
        y = (i // grid_size) * img_height
        grid_image.paste(img, (x, y))

    return grid_image


if __name__ == "__main__":
    grid_image = create_grid_image()
    grid_image.show()  # Display the grid image
    grid_image.save("grid_render.png")  # Optionally save the final grid image


# import os
# import random
# from mesh_ssm.models.mesh_autoencoder import MeshAutoencoder
# from mesh_ssm.utils.augment import augment_mesh
# from mesh_ssm.utils.mesh import FaceFeatureExtractor
# from pytorch3d.io import load_obj
# from pytorch3d.structures import Meshes, join_meshes_as_batch
# from torch.utils.data import Dataset
# import torch
# from tqdm import tqdm  # Import tqdm for the progress bar
# from mesh_ssm.utils.render import reconstruct_mesh
# from pytorch3d.io import save_obj
# from data import EncodedChairDataset
# from tools import Tools
# from project.render import render_mesh
# import numpy as np

# tools = Tools(device="cuda:3")

# tools.init_autoencoder(
#     "checkpoints/chair-final-bs[128]-lr[0.001]-ks[3]/epoch=987-val_loss=0.0000.ckpt"
# )

# dataset = np.load("encoded_chair_dataset.npy", allow_pickle=True)
# first = dataset[0]
# first = torch.tensor(first)
# print(first)

# mesh = tools.decode_mesh(first)

# verts = mesh.verts_list()[0]
# faces = mesh.faces_list()[0]

# verts = verts.cpu().numpy()
# faces = faces.cpu().numpy()

# image = render_mesh(verts, faces)
# image.save("test.png")

# # Add tqdm progress bar
# for i in tqdm(range(len(dataset)), desc="decoding meshes"):
#     all_min_encoding_indices = dataset[i]
#     print(all_min_encoding_indices)
#     all_min_encoding_indices = all_min_encoding_indices.to("cuda:3")

#     z_s = []
#     for j, quantizer in enumerate(model.rq.vq_layers):
#         # Get the quantized vectors for the i-th quantizer
#         indices = all_min_encoding_indices[:, j]
#         z_res = quantizer.get_codebook_entry(indices)
#         # Accumulate the quantized vectors
#         z_s.append(z_res)

#     z_s = torch.stack(z_s, dim=1)
#     z_q = torch.sum(z_s, dim=1)

#     # encode the mesh
#     recons = model.decode(z_q)

#     mesh = reconstruct_mesh(recons)
#     save_obj(
#         f"decoded_meshes/{i}.obj",
#         mesh.verts_list()[0],
#         mesh.faces_list()[0],
#     )
